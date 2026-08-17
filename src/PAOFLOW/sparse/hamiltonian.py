"""Bond-list representation of the PAO real-space Hamiltonian.

:class:`SparseHamiltonian` stores ``H(R)`` as a flat list of "bonds"
``(i, j, R, value)`` shared across all lattice vectors R, thresholded on
``|value|``.  The Bloch Hamiltonian at any k is assembled as

.. math::

    H_{ij}(\\mathbf{k}) = \\sum_{\\mathbf{R}} H_{ij}(\\mathbf{R})
        \\, e^{s \\, 2\\pi i \\, \\mathbf{k} \\cdot \\mathbf{R}}

with ``s = +1`` matching the dense band-path convention
(``spectrum.do_bands.band_loop_H``) and ``s = -1`` matching the dense
FFT mesh convention (``scipy.fftpack.fftn`` in
``hamiltonian.do_double_grid`` / ``do_gradient``).

The CSR sparsity pattern (the union of bond index pairs over all R) is
built once and only the numerical values are refilled per k-point, so
per-k assembly is a vectorized phase multiply plus one
``np.add.reduceat`` — the structure is never re-sorted.

For even R grids the folded Nyquist plane (``m = -nk/2``) has no
``+nk/2`` partner, which would make the Fourier sum slightly
non-Hermitian at k-points off the original grid.  The assembly plan
therefore splits every Nyquist-plane bond into two half-weight bonds at
``-nk/2`` and ``+nk/2`` — the bond-list equivalent of what
``utils.zero_pad`` does for the dense pipeline.  At original-grid
k-points the two half phases coincide, so grid values are unchanged;
off the grid, ``H(k)`` becomes exactly Hermitian, as the iterative
solver requires.  The *stored* bond list stays on the folded grid
(doubling operates on it directly); the split lives only inside the
plan.

The gradient ``dH(k)/dk_l`` is assembled on the same pattern with the
per-bond coefficient ``1j * (alat * Rcart_l + Dnm_l)``, replicating
``hamiltonian.do_gradient`` (R term) plus its diagonal tight-binding
correction (``Dnm`` term).  ``Dnm`` is stored per bond, never as a dense
``(nawf, nawf, 3)`` array.
"""

import numpy as np
from scipy.sparse import csr_matrix


def _r_offsets(nk_grid):
    """Per-axis shift mapping folded components onto ``[0, nk)``.

    A folded component lies in ``[-(nk//2), (nk-1)//2]`` for both parities,
    so adding ``nk//2`` lands it in ``[0, nk-1]`` exactly.
    """
    return tuple(int(nk) // 2 for nk in nk_grid)


def encode_R(triples, nk_grid):
    """Pack folded lattice triples into the linear FFT-box index.

    Ascending code order is *identical* to the lexicographic order of
    ``(m1, m2, m3)``, which is what ``np.unique(..., axis=0)`` produces —
    the property that lets the encoded path stay bit-identical to the
    void-view lexsort it replaces.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    o1, o2, o3 = _r_offsets(nk_grid)
    t = np.asarray(triples, dtype=np.int64)
    return ((t[:, 0] + o1) * n2 + (t[:, 1] + o2)) * n3 + (t[:, 2] + o3)


def decode_R(codes, nk_grid):
    """Inverse of :func:`encode_R`."""
    n1, n2, n3 = (int(n) for n in nk_grid)
    o1, o2, o3 = _r_offsets(nk_grid)
    codes = np.asarray(codes, dtype=np.int64)
    out = np.empty((len(codes), 3), dtype=np.int64)
    out[:, 2] = codes % n3 - o3
    out[:, 1] = (codes // n3) % n2 - o2
    out[:, 0] = (codes // (n3 * n2)) % n1 - o1
    return out


def unique_R(triples, nk_grid):
    """``np.unique(triples, axis=0, return_inverse=True)`` in O(nnz).

    Lattice triples take at most ``nk1*nk2*nk3`` distinct values, so the
    sort is replaced by a presence mask plus a lookup table.  Output is
    bit-identical to the ``axis=0`` form (same lexicographic ordering).
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    box = n1 * n2 * n3
    code = encode_R(triples, nk_grid)
    if len(code) and (code.min() < 0 or code.max() >= box):
        raise AssertionError(
            'unique_R: lattice triple outside the folded %dx%dx%d window' % (n1, n2, n3)
        )
    present = np.zeros(box, dtype=bool)
    present[code] = True
    uniq_codes = np.flatnonzero(present)
    lut = np.empty(box, dtype=np.int64)
    lut[uniq_codes] = np.arange(len(uniq_codes), dtype=np.int64)
    return decode_R(uniq_codes, nk_grid), lut[code]


def encode_bond(rows, cols, triples, nawf, nk_grid):
    """Pack ``(row, col, m1, m2, m3)`` into one int64 key.

    Row/column are the most significant digits, so ascending code order
    reproduces the lexicographic ordering of the 5-column key array that
    ``np.unique(key, axis=0)`` returns — the bond list therefore comes out
    in exactly the same order as the void-view implementation, down to the
    floating-point summation order of the assembly plan.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    box = n1 * n2 * n3
    if int(nawf) ** 2 * box >= 2**62:
        raise NotImplementedError(
            'encode_bond: nawf=%d on a %dx%dx%d R grid overflows the int64 bond key; '
            'the encoded-key path needs nawf^2 * nR < 2^62.' % (nawf, n1, n2, n3)
        )
    rc = np.asarray(rows, dtype=np.int64) * int(nawf) + np.asarray(cols, dtype=np.int64)
    return rc * box + encode_R(triples, nk_grid)


def folded_R_triples(nk1, nk2, nk3):
    """Integer lattice triples of the FFT R grid, folded around zero.

    Returns an ``(nk1*nk2*nk3, 3)`` int array whose row ``n`` is the
    folded triple ``(m1, m2, m3)`` for linear index
    ``n = k + j*nk3 + i*nk2*nk3``, replicating the fold used by
    ``utils.get_R_grid_fft`` (component ``i/nk >= 0.5`` mapped to
    ``i - nk``).
    """
    i = np.arange(nk1)
    j = np.arange(nk2)
    k = np.arange(nk3)
    m1 = np.where(i >= (nk1 + 1) // 2, i - nk1, i)
    m2 = np.where(j >= (nk2 + 1) // 2, j - nk2, j)
    m3 = np.where(k >= (nk3 + 1) // 2, k - nk3, k)
    triples = np.empty((nk1, nk2, nk3, 3), dtype=np.int32)
    triples[..., 0] = m1[:, None, None]
    triples[..., 1] = m2[None, :, None]
    triples[..., 2] = m3[None, None, :]
    return triples.reshape(nk1 * nk2 * nk3, 3)


class SparseHamiltonian:
    """Thresholded bond list for H(R) with fixed-pattern k-space assembly.

    Attributes
    ----------
    nawf, nspin : int
    alat : float
        Lattice constant in Bohr.
    a_vectors : np.ndarray, shape (3, 3)
        Current-cell lattice vectors in units of ``alat`` (rows).  Updated
        by doubling.
    R_int : np.ndarray, shape (nR, 3), int32
        Folded integer lattice triples in the current-cell basis.
    rows, cols : np.ndarray, shape (nnz,), int32
        Orbital indices of each bond.
    ridx : np.ndarray, shape (nnz,), int32
        Index into ``R_int`` for each bond.
    vals : np.ndarray, shape (nnz, nspin), complex128
    dnm : np.ndarray, shape (nnz, 3), float64
        Intra-cell orbital position difference (Bohr) on the bond
        pattern; zero for bonds connecting different cell replicas.
    threshold : float
        Magnitude below which H(R) entries were dropped (eV).
    drop_report : dict
        Truncation statistics; ``eig_bound`` is a rigorous upper bound on
        the eigenvalue shift caused by thresholding (max row sum of
        dropped magnitudes, valid at every k).
    """

    def __init__(
        self,
        nawf,
        nspin,
        alat,
        a_vectors,
        nk_grid,
        R_int,
        rows,
        cols,
        ridx,
        vals,
        dnm,
        threshold=0.0,
        drop_report=None,
    ):
        self.nawf = int(nawf)
        self.nk_grid = tuple(int(n) for n in nk_grid)
        self.nspin = int(nspin)
        self.alat = float(alat)
        self.a_vectors = np.array(a_vectors, dtype=float)
        self.R_int = np.asarray(R_int, dtype=np.int32)
        self.rows = np.asarray(rows, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32)
        self.ridx = np.asarray(ridx, dtype=np.int32)
        self.vals = np.asarray(vals, dtype=np.complex128).reshape(len(self.rows), self.nspin)
        self.dnm = np.asarray(dnm, dtype=np.float64).reshape(len(self.rows), 3)
        self.threshold = float(threshold)
        self.drop_report = drop_report if drop_report is not None else {}
        self._plan = None
        self._doubled = False
        self._compact_nnz = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data_controller(cls, data_controller, threshold, rcut=None):
        """Threshold the dense ``HRs`` in a DataController into a bond list.

        This is the single sanctioned dense-to-sparse boundary of the
        sparse pipeline, intended for the base (pre-doubling) cell where
        ``HRs`` is small.  The caller is responsible for deleting
        ``arrays['HRs']`` afterwards.  Deterministic on every rank
        (``HRs`` is broadcast by ``pao_hamiltonian``), so no MPI
        communication is needed.

        ``rcut`` (Bohr, optional) additionally drops bonds whose physical
        length ``|alat*R + tau_i - tau_j|`` exceeds it.  It is applied as
        part of ``keep``, not as a post-filter, so the reported
        ``eig_bound`` covers both truncations.  It **must** be applied
        here, at the base cell: ``doubling.double_axis`` zeroes ``dnm`` on
        cross-replica blocks (replicating the dense ``block_diag(Dnm,
        Dnm)`` semantics), after which the true bond vector is no longer
        recoverable from the container.

        Note that ``rcut`` is a physically *different* truncation axis
        from ``threshold`` (bond length vs matrix-element magnitude) and
        the two interact; the default is ``None``.
        """
        arry, attr = data_controller.data_dicts()
        HRs = arry['HRs']
        nawf, _, nk1, nk2, nk3, nspin = HRs.shape

        flat = HRs.reshape(nawf, nawf, nk1 * nk2 * nk3, nspin)
        mag = np.abs(flat).max(axis=3)  # (nawf, nawf, nR); bond kept if any spin passes
        keep = mag > threshold
        if rcut is not None:
            # bond vector alat*R_cart + Dnm, exactly the 'rcoef' of the gradient
            R_int = folded_R_triples(nk1, nk2, nk3)
            Rcart = (R_int.astype(float) @ arry['a_vectors']) * attr['alat']
            Dnm = arry['Dnm'] if 'Dnm' in arry else np.zeros((nawf, nawf, 3))
            dist = np.linalg.norm(Dnm[:, :, None, :] + Rcart[None, None, :, :], axis=3)
            # The kept set has to be closed under the Hermitian pairing
            # (i,j,R) -> (j,i,-R) or the bond list stops being Hermitian.
            # Off the Nyquist plane that is automatic, since -R is a distinct
            # grid point and |-R + tau_j - tau_i| = |R + tau_i - tau_j|.  On
            # the folded Nyquist plane -R maps onto R itself, so the two
            # partners get *different* lengths (measured: tens of Bohr apart)
            # and a plain mask would silently break Hermiticity there.  Keep
            # a bond if either partner is inside the cutoff.
            minus = _minus_R_index(R_int, (nk1, nk2, nk3))
            dist = np.minimum(dist, dist.transpose(1, 0, 2)[:, :, minus])
            keep &= dist <= float(rcut)
        rows, cols, ridx = np.nonzero(keep)

        # Gershgorin-type bound on the eigenvalue shift from truncation:
        # for any k, |eig(H) - eig(H_kept)| <= ||dH(k)||_2 <= max_i sum_{j,R} |dropped_ij(R)|
        dropped = np.where(keep, 0.0, mag)
        row_sums = dropped.sum(axis=(1, 2))
        drop_report = {
            'threshold': float(threshold),
            'rcut': None if rcut is None else float(rcut),
            'nnz': int(len(rows)),
            'density': len(rows) / max(mag.size, 1),
            'mbytes': (len(rows) * (4 + 4 + 4 + 16 * nspin + 24)) / 1024**2,
            'dense_mbytes': HRs.nbytes / 1024**2,
            'frob_dropped': float(np.sqrt((dropped**2).sum())),
            'eig_bound': float(row_sums.max()) if len(row_sums) else 0.0,
        }

        vals = flat[rows, cols, ridx, :]
        if 'Dnm' in arry:
            dnm = arry['Dnm'][rows, cols, :]
        else:
            dnm = np.zeros((len(rows), 3))

        return cls(
            nawf=nawf,
            nspin=nspin,
            alat=attr['alat'],
            a_vectors=arry['a_vectors'].copy(),
            nk_grid=(nk1, nk2, nk3),
            R_int=folded_R_triples(nk1, nk2, nk3),
            rows=rows,
            cols=cols,
            ridx=ridx,
            vals=vals,
            dnm=dnm,
            threshold=threshold,
            drop_report=drop_report,
        )

    # ------------------------------------------------------------------
    # Fixed-pattern assembly plan
    # ------------------------------------------------------------------

    def _nyquist_split(self):
        """Expand bonds on the folded Nyquist planes into half-weight pairs.

        Returns per-bond arrays ``(rows, cols, triples, vals, dnm)`` where
        every bond whose R has a component at ``-nk/2`` (even nk only) is
        duplicated into ``-nk/2`` and ``+nk/2`` at half weight, per axis —
        the bond-list equivalent of ``utils.zero_pad``.  The stored bond
        list is not modified.
        """
        rows = self.rows
        cols = self.cols
        triples = self.R_int[self.ridx].astype(np.int64)
        vals = self.vals
        dnm = self.dnm
        for axis in range(3):
            nk = self.nk_grid[axis]
            if nk % 2 != 0:
                continue
            if (triples[:, axis] == nk // 2).any():
                # +nk/2 already present (hermitized list): the plane is
                # explicitly paired, splitting again would skew weights
                continue
            hit = triples[:, axis] == -(nk // 2)
            if not hit.any():
                continue
            vals = vals.copy()
            vals[hit] *= 0.5
            mirrored = triples[hit].copy()
            mirrored[:, axis] = nk // 2
            rows = np.concatenate((rows, rows[hit]))
            cols = np.concatenate((cols, cols[hit]))
            triples = np.concatenate((triples, mirrored))
            vals = np.concatenate((vals, vals[hit]))
            dnm = np.concatenate((dnm, dnm[hit]))
        return rows, cols, triples, vals, dnm

    def _build_plan(self):
        """Sort bonds into CSR order once; per-k assembly only refills data."""
        rows, cols, triples, vals, dnm = self._nyquist_split()

        # compact unique-R list so per-k phases are computed once per R
        R_uniq, ridx = np.unique(triples, axis=0, return_inverse=True)

        pair = rows.astype(np.int64) * self.nawf + cols
        order = np.argsort(pair, kind='stable')
        pair = pair[order]

        seg_starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
        upair = pair[seg_starts]
        indices = (upair % self.nawf).astype(np.int32)
        counts = np.bincount((upair // self.nawf).astype(np.intp), minlength=self.nawf)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)

        Rcart = R_uniq.astype(float) @ self.a_vectors  # units of alat
        ridx = ridx[order]
        rcoef = self.alat * Rcart[ridx] + dnm[order]  # (nnz', 3), Bohr

        self._plan = {
            'seg_starts': seg_starts,
            'indices': indices,
            'indptr': indptr,
            'R_uniq': R_uniq,
            'Rcart': Rcart,
            'ridx': ridx,
            'vals': vals[order],
            'rcoef': rcoef,
        }

    @property
    def plan(self):
        if self._plan is None:
            self._build_plan()
        return self._plan

    def invalidate_plan(self):
        """Call after mutating geometry (a_vectors) so coefficients rebuild."""
        self._plan = None

    def compact(self):
        """Build the assembly plan and release the raw bond arrays.

        The plan already carries everything per-k assembly needs (ordered
        values, CSR indices, gradient coefficients), so after it is built
        ``rows``/``cols``/``dnm``/``vals`` are dead weight — roughly half
        the steady-state bond memory.  Call this once the bond list is
        final, i.e. after the last ``double_axis`` and ``hermitize``.

        Irreversible: anything that mutates or inspects the bond list
        (``hermitize``, ``hermiticity_error``, ``double_axis``) raises
        afterwards.  Assembly, and therefore every property, is unaffected.
        """
        self._require_bonds('compact')
        self.plan  # force the build while the raw arrays are still here
        self._compact_nnz = self.nnz
        self.rows = None
        self.cols = None
        self.dnm = None
        self.vals = None
        return self

    @property
    def compacted(self):
        return self._compact_nnz is not None

    def _require_bonds(self, caller):
        if self.compacted:
            raise RuntimeError(
                'SparseHamiltonian.%s needs the raw bond arrays, which compact() '
                'released. Do all doubling and hermitization before compact().' % caller
            )

    # ------------------------------------------------------------------
    # k-space assembly
    # ------------------------------------------------------------------

    def _phase_arg(self, kvec, cart):
        """Dimensionless k.R per lattice vector, shape (nR,).

        ``cart=False``: k in crystal coordinates, k.R = kfrac . m.
        ``cart=True``: k Cartesian in units of 2 pi / alat (the dense
        band-path convention after the ``b_vectors`` rotation in
        ``do_bands``); k.R uses the Cartesian R of the *current* cell,
        replicating ``band_loop_H`` exactly (including doubled cells,
        where ``b_vectors`` remain those of the original cell).
        """
        p = self.plan
        if cart:
            return p['Rcart'] @ np.asarray(kvec, dtype=float)
        return p['R_uniq'] @ np.asarray(kvec, dtype=float)

    def assemble_hk(self, kvec, ispin=0, sign=-1, cart=False):
        """Assemble H(k) as a CSR matrix on the fixed union pattern."""
        p = self.plan
        phase = np.exp((sign * 2.0j * np.pi) * self._phase_arg(kvec, cart))
        vp = p['vals'][:, ispin] * phase[p['ridx']]
        data = np.add.reduceat(vp, p['seg_starts'])
        return csr_matrix((data, p['indices'], p['indptr']), shape=(self.nawf, self.nawf))

    def assemble_hk_dhk(self, kvec, ispin=0, sign=-1, cart=False):
        """Assemble H(k) and its gradient [dH/dk_0, dH/dk_1, dH/dk_2].

        The gradient replicates ``hamiltonian.do_gradient``: per bond,
        ``dH_l = 1j * (alat * Rcart_l + Dnm_l) * H_ij(R) * phase``.
        """
        p = self.plan
        phase = np.exp((sign * 2.0j * np.pi) * self._phase_arg(kvec, cart))
        vp = p['vals'][:, ispin] * phase[p['ridx']]
        hk = csr_matrix(
            (np.add.reduceat(vp, p['seg_starts']), p['indices'], p['indptr']),
            shape=(self.nawf, self.nawf),
        )
        dhk = []
        for l in range(3):
            data = np.add.reduceat(1j * p['rcoef'][:, l] * vp, p['seg_starts'])
            dhk.append(csr_matrix((data, p['indices'], p['indptr']), shape=(self.nawf, self.nawf)))
        return hk, dhk

    # ------------------------------------------------------------------
    # Hermitization
    # ------------------------------------------------------------------

    def hermitize(self):
        """Return the bond-level Hermitian average ``(B + B^dagger)/2``.

        ``B^dagger`` is the transposed, conjugated bond list at ``-R``,
        with ``-R`` folded back into the R window (``+nk/2 -> -nk/2`` on
        even axes), so the result stays in folded-grid coordinates and
        the assembly plan's Nyquist split applies uniformly.  Combined
        with that split, this is *exactly* equivalent to replacing
        ``H(k)`` by ``(H(k) + H(k)^dagger)/2`` at every k-point — the
        operation the dense pipeline applies to ``Hksp``/``dHksp`` in
        ``gradient_and_momenta`` — done once, in real space.

        Needed because the dense doubling kernel (which sparse doubling
        replicates bond-for-bond) maps the base cell's self-paired
        Nyquist plane asymmetrically, leaving the doubled ``H(R)``
        slightly non-Hermitian; the dense pipeline mops this up with
        per-k Hermitizations and one-triangle ``eigh`` reads, while the
        sparse solver requires an exactly Hermitian operator.  On an
        already Hermitian bond list this is an exact no-op.
        """
        self._require_bonds('hermitize')
        triples = self.R_int[self.ridx].astype(np.int64)
        mirrored = -triples
        for axis in range(3):
            nk = self.nk_grid[axis]
            if nk % 2 == 0:
                comp = mirrored[:, axis]
                mirrored[:, axis] = np.where(comp == nk // 2, -(nk // 2), comp)
        rows_all = np.concatenate((self.rows, self.cols))
        cols_all = np.concatenate((self.cols, self.rows))
        triples_all = np.concatenate((triples, mirrored))
        vals = np.concatenate((0.5 * self.vals, 0.5 * np.conj(self.vals)))
        dnm = np.concatenate((self.dnm, -self.dnm))  # Dnm_ij = -Dnm_ji

        # One int64 key per bond instead of np.unique's void-view lexsort over
        # a (2*nnz, 5) array: same ordering, ~17x faster and 8 B/row not 40.
        code = encode_bond(rows_all, cols_all, triples_all, self.nawf, self.nk_grid)
        uniq_code, first, inv = np.unique(code, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        vals_m = np.zeros((len(uniq_code), self.nspin), dtype=np.complex128)
        np.add.at(vals_m, inv, vals)
        dnm_m = np.zeros((len(uniq_code), 3))
        dnm_m[inv] = dnm  # group members carry identical dnm (not a weight)

        R_uniq, ridx = unique_R(triples_all[first], self.nk_grid)
        out = SparseHamiltonian(
            nawf=self.nawf,
            nspin=self.nspin,
            alat=self.alat,
            a_vectors=self.a_vectors.copy(),
            nk_grid=self.nk_grid,
            R_int=R_uniq.astype(np.int32),
            rows=rows_all[first],
            cols=cols_all[first],
            ridx=ridx,
            vals=vals_m,
            dnm=dnm_m,
            threshold=self.threshold,
            drop_report=dict(self.drop_report),
        )
        out._doubled = self._doubled
        return out

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def nnz(self):
        return self._compact_nnz if self.compacted else len(self.rows)

    def density(self):
        return self.nnz / (self.nawf * self.nawf * len(self.R_int))

    def hermiticity_error(self):
        """Max |H_ij(R) - conj(H_ji(-R))| over stored bonds (diagnostic).

        Bonds whose (j, i, -R) partner was dropped by thresholding
        contribute |H_ij(R)| directly.  Fully vectorized: usable as a
        production assertion at tens of millions of bonds.
        """
        self._require_bonds('hermiticity_error')
        if self.nnz == 0:
            return 0.0
        nR = len(self.R_int)
        minus = _minus_R_index(self.R_int, self.nk_grid)
        rows = self.rows.astype(np.int64)
        cols = self.cols.astype(np.int64)
        ridx = self.ridx.astype(np.int64)
        code = (rows * self.nawf + cols) * nR + ridx
        mr = minus[ridx]
        partner_code = (cols * self.nawf + rows) * nR + np.maximum(mr, 0)

        order = np.argsort(code)
        sorted_code = code[order]
        pos = np.searchsorted(sorted_code, partner_code)
        np.clip(pos, 0, len(sorted_code) - 1, out=pos)
        found = (sorted_code[pos] == partner_code) & (mr >= 0)
        partner = order[pos]

        # where the partner is missing the bond is its own error
        err = np.where(
            found,
            np.abs(self.vals - np.conj(self.vals[partner])).max(axis=1),
            np.abs(self.vals).max(axis=1),
        )
        return float(err.max())

    def stats_line(self):
        mbytes = self.nnz * (4 + 4 + 4 + 16 * self.nspin + 24) / 1024**2
        dense_mbytes = (self.nawf**2 * np.prod(self.nk_grid) * self.nspin * 16) / 1024**2
        return (
            'Sparse H(R): nawf=%d, nR=%d, nnz=%.3gM, density=%.2e, mem=%.1f MB '
            '(dense equivalent %.1f MB), truncation eig-shift bound=%.2e eV'
            % (
                self.nawf,
                len(self.R_int),
                self.nnz / 1e6,
                self.density(),
                mbytes,
                dense_mbytes,
                self.drop_report.get('eig_bound', 0.0),
            )
        )


def _minus_R_index(R_int, nk_grid):
    """Index of -R for each row of R_int, folding Nyquist components onto
    themselves (for even grids, -(-nk/2) folds back to -nk/2).

    Returns ``-1`` where -R is not present in ``R_int`` at all, which a
    thresholded bond list can produce.  The window comes from ``nk_grid``
    rather than from the extent of ``R_int``, so a sparse R list folds
    correctly.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    box = n1 * n2 * n3
    R = np.asarray(R_int, dtype=np.int64)
    m = -R
    for axis, nk in enumerate((n1, n2, n3)):
        if nk % 2 == 0:
            m[:, axis] = np.where(m[:, axis] == nk // 2, -(nk // 2), m[:, axis])
    lut = np.full(box, -1, dtype=np.int64)
    lut[encode_R(R, nk_grid)] = np.arange(len(R), dtype=np.int64)
    return lut[encode_R(m, nk_grid)]
