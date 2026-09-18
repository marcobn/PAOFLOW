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

``HERMITIZE_GROWTH`` is the bond-count growth factor across
:meth:`SparseHamiltonian.hermitize`, used for size projection only.  That
operation unions the list with its mirrored conjugate, so the true factor
lies in ``[1, 2]``: 1 when every ``(j,i,-R)`` partner is already present, 2
when none is.  Thresholding drops partners asymmetrically, so it sits above
1; measured 1.19 on example01 at nx=1, rounded up here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from PAOFLOW.DataController import DataController

HERMITIZE_GROWTH = 1.25


def _r_offsets(nk_grid: Sequence[int]) -> tuple[int, ...]:
    """Per-axis shift mapping folded components onto ``[0, nk)``.

    Parameters
    ----------
    nk_grid : sequence of int
        The three R-grid dimensions.

    Returns
    -------
    tuple of int
        The shift to add to each axis component.

    Notes
    -----
    A folded component lies in ``[-(nk//2), (nk-1)//2]`` for both parities,
    so adding ``nk//2`` lands it in ``[0, nk-1]`` exactly.
    """
    return tuple(int(nk) // 2 for nk in nk_grid)


def encode_R(triples: np.ndarray, nk_grid: Sequence[int]) -> np.ndarray:
    """Pack folded lattice triples into the linear FFT-box index.

    Parameters
    ----------
    triples : np.ndarray, shape (m, 3)
        Folded integer lattice triples.
    nk_grid : sequence of int
        The three R-grid dimensions.

    Returns
    -------
    np.ndarray, shape (m,), int64
        One integer per triple, in ``[0, nk1*nk2*nk3)``.

    Notes
    -----
    Each triple is shifted into the non-negative box and then read as a
    three-digit number in mixed radix, most significant axis first.  Because
    the shift is monotonic per axis, ascending code order is *identical* to
    the lexicographic order of ``(m1, m2, m3)``, which is what
    ``np.unique(..., axis=0)`` produces — the property that lets the encoded
    path stay bit-identical to the void-view lexsort it replaces.
    """
    _, n2, n3 = (int(n) for n in nk_grid)
    o1, o2, o3 = _r_offsets(nk_grid)
    t = np.asarray(triples, dtype=np.int64)
    return ((t[:, 0] + o1) * n2 + (t[:, 1] + o2)) * n3 + (t[:, 2] + o3)


def decode_R(codes: np.ndarray, nk_grid: Sequence[int]) -> np.ndarray:
    """Unpack linear FFT-box indices back into folded lattice triples.

    Parameters
    ----------
    codes : np.ndarray, shape (m,)
        Codes as produced by :func:`encode_R`.
    nk_grid : sequence of int
        The three R-grid dimensions.

    Returns
    -------
    np.ndarray, shape (m, 3), int64
        The folded lattice triples.

    Notes
    -----
    Exact inverse of :func:`encode_R`: successive division and remainder
    peel off the mixed-radix digits, and the per-axis offsets are subtracted
    to return to folded coordinates.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    o1, o2, o3 = _r_offsets(nk_grid)
    codes = np.asarray(codes, dtype=np.int64)
    out = np.empty((len(codes), 3), dtype=np.int64)
    out[:, 2] = codes % n3 - o3
    out[:, 1] = (codes // n3) % n2 - o2
    out[:, 0] = (codes // (n3 * n2)) % n1 - o1
    return out


def unique_R(triples: np.ndarray, nk_grid: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """``np.unique(triples, axis=0, return_inverse=True)`` in O(nnz).

    Parameters
    ----------
    triples : np.ndarray, shape (nnz, 3)
        Folded lattice triples, one per bond, with repeats.
    nk_grid : sequence of int
        The three R-grid dimensions.

    Returns
    -------
    (R_uniq, inverse) : tuple of np.ndarray
        The distinct triples in lexicographic order, ``(nR, 3)``, and the
        index into them for each input row, ``(nnz,)``.

    Raises
    ------
    AssertionError
        If any triple lies outside the folded window of the grid, which
        would mean the caller produced a lattice vector the R grid cannot
        represent.

    Notes
    -----
    Lattice triples take at most ``nk1*nk2*nk3`` distinct values, a number
    fixed by the grid and typically far smaller than the bond count.  So the
    generic sort-based unique — ``O(nnz log nnz)`` on a structured array — is
    replaced by a presence mask over the whole box plus a lookup table:
    every bond is touched twice, and nothing is sorted.  Output is
    bit-identical to the ``axis=0`` form, since the code order of
    :func:`encode_R` reproduces lexicographic order.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    box = n1 * n2 * n3
    code = encode_R(triples, nk_grid)
    if len(code) and (code.min() < 0 or code.max() >= box):
        raise AssertionError(f'unique_R: lattice triple outside the folded {n1}x{n2}x{n3} window')
    present = np.zeros(box, dtype=bool)
    present[code] = True
    uniq_codes = np.flatnonzero(present)
    lut = np.empty(box, dtype=np.int64)
    lut[uniq_codes] = np.arange(len(uniq_codes), dtype=np.int64)
    return decode_R(uniq_codes, nk_grid), lut[code]


def encode_bond(
    rows: np.ndarray,
    cols: np.ndarray,
    triples: np.ndarray,
    nawf: int,
    nk_grid: Sequence[int],
) -> np.ndarray:
    """Pack ``(row, col, m1, m2, m3)`` into one int64 key.

    Parameters
    ----------
    rows, cols : np.ndarray, shape (nnz,)
        Orbital indices of each bond.
    triples : np.ndarray, shape (nnz, 3)
        Folded lattice triple of each bond.
    nawf : int
        Number of orbitals, the radix of the row/column digits.
    nk_grid : sequence of int
        The three R-grid dimensions.

    Returns
    -------
    np.ndarray, shape (nnz,), int64
        One key per bond, unique across the whole ``(orbital, orbital, R)``
        space.

    Raises
    ------
    NotImplementedError
        If the key space does not fit in int64.

    Notes
    -----
    Row and column are the most significant digits, so ascending code order
    reproduces the lexicographic ordering of the 5-column key array that
    ``np.unique(key, axis=0)`` returns — the bond list therefore comes out
    in exactly the same order as the void-view implementation, down to the
    floating-point summation order of the assembly plan.  A single integer
    key also sorts an order of magnitude faster than a structured array and
    costs 8 bytes per bond instead of 40.
    """
    n1, n2, n3 = (int(n) for n in nk_grid)
    box = n1 * n2 * n3
    if int(nawf) ** 2 * box >= 2**62:
        raise NotImplementedError(
            f'encode_bond: nawf={nawf} on a {n1}x{n2}x{n3} R grid overflows the int64 bond '
            'key; the encoded-key path needs nawf^2 * nR < 2^62.'
        )
    rc = np.asarray(rows, dtype=np.int64) * int(nawf) + np.asarray(cols, dtype=np.int64)
    return rc * box + encode_R(triples, nk_grid)


def folded_R_triples(nk1: int, nk2: int, nk3: int) -> np.ndarray:
    """Integer lattice triples of the FFT R grid, folded around zero.

    Parameters
    ----------
    nk1, nk2, nk3 : int
        R-grid dimensions.

    Returns
    -------
    np.ndarray, shape (nk1*nk2*nk3, 3), int32
        Row ``n`` is the folded triple ``(m1, m2, m3)`` for linear index
        ``n = k + j*nk3 + i*nk2*nk3``.

    Notes
    -----
    An FFT indexes its real-space grid from 0 to ``nk-1``, but the second
    half of that range represents *negative* lattice vectors: index
    ``i >= nk/2`` means the cell at ``i - nk``.  Folding makes that explicit,
    so a lattice triple can be used directly as a displacement in the phase
    factor and in the bond length.  This replicates the fold used by
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

    Parameters
    ----------
    nawf : int
        Number of atomic orbitals in the current cell.
    nspin : int
        Number of spin channels.
    alat : float
        Lattice constant in Bohr.
    a_vectors : array_like, shape (3, 3)
        Lattice vectors in units of ``alat`` (rows).
    nk_grid : sequence of int
        The three R-grid dimensions; fixed by the original DFT k-mesh and
        unchanged by doubling.
    R_int : array_like, shape (nR, 3)
        Distinct folded lattice triples referenced by the bonds.
    rows, cols : array_like, shape (nnz,)
        Orbital indices of each bond.
    ridx : array_like, shape (nnz,)
        Index into ``R_int`` for each bond.
    vals : array_like, shape (nnz, nspin)
        Hamiltonian matrix element of each bond, in eV.
    dnm : array_like, shape (nnz, 3)
        Intra-cell orbital position difference of each bond, in Bohr.
    threshold : float, optional
        Magnitude below which H(R) entries were dropped (eV).
    drop_report : dict or None, optional
        Truncation statistics, carried alongside the data.

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

    Notes
    -----
    A tight-binding Hamiltonian in real space is a list of hoppings: orbital
    ``i`` in the home cell couples to orbital ``j`` in the cell displaced by
    lattice vector ``R``, with some amplitude.  Almost all of those
    amplitudes are negligible — the interaction decays with distance — so
    storing the full ``(nawf, nawf, nR)`` array wastes memory that grows
    quadratically with the cell.  Here only the surviving hoppings are
    stored, as parallel arrays of row, column, R index and value.  Cell
    doubling then costs a factor of two in memory instead of four.

    The one operation the whole pipeline repeats is building
    :math:`H(\\mathbf{k})` from these bonds, millions of times over a
    k-mesh.  Each bond always lands in the same matrix entry regardless of
    k, and only its phase factor changes, so the sparsity structure is
    computed once (:attr:`plan`) and every k-point reuses it: multiply each
    bond by its phase, sum the bonds that share a matrix entry, done.  No
    sorting, no allocation of index arrays, no dense intermediate.
    """

    def __init__(
        self,
        nawf: int,
        nspin: int,
        alat: float,
        a_vectors: np.ndarray,
        nk_grid: Sequence[int],
        R_int: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        ridx: np.ndarray,
        vals: np.ndarray,
        dnm: np.ndarray,
        threshold: float = 0.0,
        drop_report: dict[str, Any] | None = None,
    ) -> None:
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
    def from_data_controller(
        cls, data_controller: DataController, threshold: float, rcut: float | None = None
    ) -> SparseHamiltonian:
        """Threshold the dense ``HRs`` in a DataController into a bond list.

        Parameters
        ----------
        data_controller : DataController
            Run state holding the dense ``arrays['HRs']`` produced by
            ``pao_hamiltonian``, plus ``a_vectors``, ``alat`` and, when
            present, ``Dnm``.
        threshold : float
            Magnitude in eV below which a hopping is dropped.
        rcut : float or None, optional
            Additional bond-length cutoff in Bohr.

        Returns
        -------
        SparseHamiltonian
            The thresholded bond list for the base cell.

        Notes
        -----
        This is the single sanctioned dense-to-sparse boundary of the sparse
        pipeline, intended for the base (pre-doubling) cell where ``HRs`` is
        small.  The caller is responsible for deleting ``arrays['HRs']``
        afterwards.  Deterministic on every rank (``HRs`` is broadcast by
        ``pao_hamiltonian``), so no MPI communication is needed.

        A hopping is kept if it passes the threshold in *any* spin channel,
        so both channels keep a common sparsity pattern and the assembly
        plan is shared between them.

        Truncation shifts the eigenvalues, and by how much is bounded rather
        than guessed.  The dropped part of the Hamiltonian is itself a
        Hermitian matrix at every k, and Gershgorin's theorem bounds its
        spectral norm by its largest absolute row sum; a perturbation of
        that norm can move an eigenvalue by at most that much.  Summing the
        dropped magnitudes per row therefore gives ``eig_bound``, a rigorous
        upper bound on the error introduced, valid at every k-point.

        ``rcut`` drops bonds whose physical length ``|alat*R + tau_i -
        tau_j|`` exceeds it.  It is applied as part of the keep mask, not as
        a post-filter, so the reported ``eig_bound`` covers both truncations.
        The kept set has to be closed under the Hermitian pairing ``(i,j,R)
        -> (j,i,-R)`` or the bond list stops being Hermitian.  Off the
        Nyquist plane that is automatic, since ``-R`` is a distinct grid
        point at the same length.  On the folded Nyquist plane ``-R`` maps
        onto ``R`` itself, so the two partners get *different* lengths
        (measured: tens of Bohr apart) and a plain mask would silently break
        Hermiticity there; a bond is therefore kept when either partner is
        inside the cutoff.

        ``rcut`` **must** be applied here, at the base cell:
        ``doubling.double_axis`` zeroes ``dnm`` on cross-replica blocks
        (replicating the dense ``block_diag(Dnm, Dnm)`` semantics), after
        which the true bond vector is no longer recoverable from the
        container.  Note also that ``rcut`` is a physically *different*
        truncation axis from ``threshold`` (bond length vs matrix-element
        magnitude) and the two interact; the default is ``None``.
        """
        arry, attr = data_controller.data_dicts()
        HRs = arry['HRs']
        nawf, _, nk1, nk2, nk3, nspin = HRs.shape

        flat = HRs.reshape(nawf, nawf, nk1 * nk2 * nk3, nspin)
        mag = np.abs(flat).max(axis=3)
        keep = mag > threshold
        if rcut is not None:
            R_int = folded_R_triples(nk1, nk2, nk3)
            Rcart = (R_int.astype(float) @ arry['a_vectors']) * attr['alat']
            Dnm = arry['Dnm'] if 'Dnm' in arry else np.zeros((nawf, nawf, 3))
            dist = np.linalg.norm(Dnm[:, :, None, :] + Rcart[None, None, :, :], axis=3)
            minus = _minus_R_index(R_int, (nk1, nk2, nk3))
            dist = np.minimum(dist, dist.transpose(1, 0, 2)[:, :, minus])
            keep &= dist <= float(rcut)
        rows, cols, ridx = np.nonzero(keep)

        dropped = np.where(keep, 0.0, mag)
        row_sums = dropped.sum(axis=(1, 2))
        drop_report = {
            'threshold': float(threshold),
            'rcut': None if rcut is None else float(rcut),
            'nnz': len(rows),
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

    def _nyquist_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Expand bonds on the folded Nyquist planes into half-weight pairs.

        Returns
        -------
        (rows, cols, triples, vals, dnm) : tuple of np.ndarray
            Per-bond arrays in which every bond whose R has a component at
            ``-nk/2`` (even ``nk`` only) has been duplicated into ``-nk/2``
            and ``+nk/2`` at half weight, per axis.

        Notes
        -----
        On an even grid the highest-frequency lattice plane is its own
        mirror image: ``-nk/2`` and ``+nk/2`` label the same folded index, so
        the plane has no distinct partner to pair with under ``R -> -R``.
        Left alone, its contribution to the Fourier sum is not Hermitian at
        k-points away from the original mesh, and the iterative eigensolver
        requires an exactly Hermitian operator.

        Splitting the plane's weight evenly between ``-nk/2`` and ``+nk/2``
        restores the symmetry: the two phases are complex conjugates, so
        their average is real-symmetric in the right way.  At original-grid
        k-points the two phases coincide and the split changes nothing, so
        grid values are bit-for-bit what the dense pipeline gets.  This is
        the bond-list equivalent of ``utils.zero_pad``.  The stored bond
        list is not modified — the split lives only inside the plan, so
        doubling continues to operate on folded-grid coordinates.

        A list that has already been hermitized carries an explicit
        ``+nk/2`` plane; splitting again would halve weights that are
        already paired, so that case is detected and skipped.
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

    def _build_plan(self) -> None:
        """Sort bonds into CSR order once; per-k assembly only refills data.

        Notes
        -----
        The plan is everything about ``H(k)`` that does not depend on k.
        Bonds are Nyquist-split, grouped by the matrix entry they land in,
        and sorted into compressed-row order; the boundaries between groups
        are recorded so that summing a group is one segmented reduction.
        The distinct lattice vectors are collected separately, so a k-point
        evaluates one phase per lattice vector rather than one per bond.

        The gradient coefficients ``alat * Rcart + dnm`` are precomputed on
        the same ordering, since they too are k-independent: the derivative
        of the phase factor with respect to k brings down exactly this bond
        displacement.

        The sort is stable, so bonds landing in the same matrix entry are
        summed in a fixed order and the assembly is reproducible to the last
        bit across runs.
        """
        rows, cols, triples, vals, dnm = self._nyquist_split()

        R_uniq, ridx = np.unique(triples, axis=0, return_inverse=True)

        pair = rows.astype(np.int64) * self.nawf + cols
        order = np.argsort(pair, kind='stable')
        pair = pair[order]

        seg_starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
        upair = pair[seg_starts]
        indices = (upair % self.nawf).astype(np.int32)
        counts = np.bincount((upair // self.nawf).astype(np.intp), minlength=self.nawf)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)

        Rcart = R_uniq.astype(float) @ self.a_vectors
        ridx = ridx[order]
        rcoef = self.alat * Rcart[ridx] + dnm[order]

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
    def plan(self) -> dict[str, np.ndarray]:
        """The k-independent assembly plan, built on first access.

        Notes
        -----
        Keys are ``seg_starts`` (segment boundaries of the reduction),
        ``indices`` and ``indptr`` (the CSR pattern), ``R_uniq`` and
        ``Rcart`` (distinct lattice vectors, fractional and Cartesian in
        units of ``alat``), ``ridx`` (lattice vector of each ordered bond),
        ``vals`` (bond values in plan order) and ``rcoef`` (gradient
        coefficients in Bohr).
        """
        if self._plan is None:
            self._build_plan()
        return self._plan

    def invalidate_plan(self) -> None:
        """Discard the cached plan so geometry changes take effect.

        Notes
        -----
        Call after mutating ``a_vectors``: the Cartesian lattice vectors and
        the gradient coefficients are baked into the plan, so a stale plan
        would silently keep using the old cell.
        """
        self._plan = None

    def compact(self) -> SparseHamiltonian:
        """Build the assembly plan and release the raw bond arrays.

        Returns
        -------
        SparseHamiltonian
            ``self``, for chaining.

        Raises
        ------
        RuntimeError
            If already compacted.

        Notes
        -----
        The plan already carries everything per-k assembly needs (ordered
        values, CSR indices, gradient coefficients), so after it is built
        ``rows``/``cols``/``dnm``/``vals`` are dead weight — roughly half
        the steady-state bond memory.  Call this once the bond list is
        final, i.e. after the last ``double_axis`` and ``hermitize``.

        Irreversible: anything that mutates or inspects the bond list
        (:meth:`hermitize`, :meth:`hermiticity_error`,
        :func:`~PAOFLOW.sparse.doubling.double_axis`) raises afterwards.
        Assembly, and therefore every property, is unaffected.
        """
        self._require_bonds('compact')
        self.plan
        self._compact_nnz = self.nnz
        self.rows = None
        self.cols = None
        self.dnm = None
        self.vals = None
        return self

    @property
    def compacted(self) -> bool:
        """Whether :meth:`compact` has released the raw bond arrays."""
        return self._compact_nnz is not None

    def _require_bonds(self, caller: str) -> None:
        """Raise if the raw bond arrays needed by ``caller`` are gone."""
        if self.compacted:
            raise RuntimeError(
                f'SparseHamiltonian.{caller} needs the raw bond arrays, which compact() '
                'released. Do all doubling and hermitization before compact().'
            )

    # ------------------------------------------------------------------
    # k-space assembly
    # ------------------------------------------------------------------

    def _phase_arg(self, kvec: np.ndarray, cart: bool) -> np.ndarray:
        """Dimensionless ``k.R`` per lattice vector.

        Parameters
        ----------
        kvec : np.ndarray, shape (3,)
            The k-point, in the frame selected by ``cart``.
        cart : bool
            ``False``: k in crystal coordinates, so ``k.R = kfrac . m``.
            ``True``: k Cartesian in units of ``2 pi / alat`` (the dense
            band-path convention after the ``b_vectors`` rotation in
            ``do_bands``), so ``k.R`` uses the Cartesian R of the *current*
            cell.

        Returns
        -------
        np.ndarray, shape (nR,)
            The dot product for each distinct lattice vector.

        Notes
        -----
        The Cartesian branch replicates ``band_loop_H`` exactly, including
        doubled cells, where ``b_vectors`` remain those of the original cell
        and the doubling shows up in the Cartesian R instead.
        """
        p = self.plan
        if cart:
            return p['Rcart'] @ np.asarray(kvec, dtype=float)
        return p['R_uniq'] @ np.asarray(kvec, dtype=float)

    def assemble_hk(
        self, kvec: np.ndarray, ispin: int = 0, sign: int = -1, cart: bool = False
    ) -> csr_matrix:
        """Assemble ``H(k)`` as a CSR matrix on the fixed union pattern.

        Parameters
        ----------
        kvec : np.ndarray, shape (3,)
            The k-point, in the frame selected by ``cart``.
        ispin : int, optional
            Spin channel.
        sign : {-1, +1}, optional
            Sign of the Fourier phase: ``-1`` is the FFT mesh convention,
            ``+1`` the band-path convention.
        cart : bool, optional
            Whether ``kvec`` is Cartesian; see :meth:`_phase_arg`.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (nawf, nawf)
            The Bloch Hamiltonian at this k-point, in eV.

        Notes
        -----
        Each lattice vector contributes a phase :math:`e^{s 2\\pi i
        \\mathbf{k} \\cdot \\mathbf{R}}`; every bond is multiplied by the
        phase of its own lattice vector, and bonds sharing a matrix entry are
        summed with one segmented reduction over the precomputed group
        boundaries.  The CSR index arrays come straight from the plan, so no
        sorting or index construction happens per k-point.
        """
        p = self.plan
        phase = np.exp((sign * 2.0j * np.pi) * self._phase_arg(kvec, cart))
        vp = p['vals'][:, ispin] * phase[p['ridx']]
        data = np.add.reduceat(vp, p['seg_starts'])
        return csr_matrix((data, p['indices'], p['indptr']), shape=(self.nawf, self.nawf))

    def assemble_hk_dhk(
        self, kvec: np.ndarray, ispin: int = 0, sign: int = -1, cart: bool = False
    ) -> tuple[csr_matrix, list[csr_matrix]]:
        """Assemble ``H(k)`` and its gradient with respect to k.

        Parameters
        ----------
        kvec : np.ndarray, shape (3,)
            The k-point, in the frame selected by ``cart``.
        ispin : int, optional
            Spin channel.
        sign : {-1, +1}, optional
            Sign of the Fourier phase; see :meth:`assemble_hk`.
        cart : bool, optional
            Whether ``kvec`` is Cartesian; see :meth:`_phase_arg`.

        Returns
        -------
        (hk, dhk) : tuple
            The Bloch Hamiltonian, and a list of its three Cartesian
            derivatives ``[dH/dk_0, dH/dk_1, dH/dk_2]``, all CSR matrices of
            shape ``(nawf, nawf)``.

        Notes
        -----
        Differentiating the Fourier sum with respect to k brings down a
        factor of the bond displacement, so the gradient shares the phase
        factors and the sparsity pattern of :math:`H(k)` and costs only three
        more segmented reductions.  Per bond the coefficient is
        ``1j * (alat * Rcart_l + Dnm_l)``, replicating
        ``hamiltonian.do_gradient``: the ``Rcart`` term is the displacement
        between cells, and the ``Dnm`` term is the tight-binding correction
        for the offset between the two orbitals inside the cell.  Both are
        precomputed in the plan as ``rcoef``.
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

    def hermitize(self) -> SparseHamiltonian:
        """Return the bond-level Hermitian average ``(B + B^dagger)/2``.

        Returns
        -------
        SparseHamiltonian
            A new bond list, exactly Hermitian at every k-point.

        Raises
        ------
        RuntimeError
            If the bond arrays have been released by :meth:`compact`.

        Notes
        -----
        A Hamiltonian must be Hermitian, which in real space means the
        hopping from orbital ``j`` in cell ``-R`` to orbital ``i`` in the
        home cell is the complex conjugate of the hopping from ``i`` to
        ``j`` at ``R``.  This method enforces that by averaging the bond list
        with its own transposed conjugate: each bond ``(i, j, R, v)`` is
        paired with ``(j, i, -R, conj(v))``, both at half weight, and bonds
        that then coincide are added together.  The mirrored ``-R`` is folded
        back into the R window (``+nk/2 -> -nk/2`` on even axes), so the
        result stays in folded-grid coordinates and the assembly plan's
        Nyquist split applies uniformly.

        Combined with that split, this is *exactly* equivalent to replacing
        ``H(k)`` by ``(H(k) + H(k)^dagger)/2`` at every k-point — the
        operation the dense pipeline applies to ``Hksp``/``dHksp`` in
        ``gradient_and_momenta`` — done once, in real space, instead of at
        every k-point.

        It is needed because the dense doubling kernel (which sparse
        doubling replicates bond-for-bond) maps the base cell's self-paired
        Nyquist plane asymmetrically, leaving the doubled ``H(R)`` slightly
        non-Hermitian; the dense pipeline mops this up with per-k
        Hermitizations and one-triangle ``eigh`` reads, while the sparse
        solver requires an exactly Hermitian operator.  On an already
        Hermitian bond list this is an exact no-op.

        The merge groups bonds by a single int64 key (:func:`encode_bond`)
        rather than by ``np.unique`` over a ``(2*nnz, 5)`` array: same
        ordering, roughly 17x faster, and 8 bytes per row instead of 40 —
        which matters because this step is the memory high-water mark of the
        whole doubling stage.  The position differences ``dnm`` are a per-pair
        property rather than a weight, so members of a group carry identical
        values and are assigned rather than summed; the antisymmetry
        ``Dnm_ij = -Dnm_ji`` is applied to the mirrored half.
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
        dnm = np.concatenate((self.dnm, -self.dnm))

        code = encode_bond(rows_all, cols_all, triples_all, self.nawf, self.nk_grid)
        uniq_code, first, inv = np.unique(code, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        vals_m = np.zeros((len(uniq_code), self.nspin), dtype=np.complex128)
        np.add.at(vals_m, inv, vals)
        dnm_m = np.zeros((len(uniq_code), 3))
        dnm_m[inv] = dnm

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
    def nnz(self) -> int:
        """Number of stored bonds, before or after :meth:`compact`."""
        return self._compact_nnz if self.compacted else len(self.rows)

    def density(self) -> float:
        """Fraction of the full ``(nawf, nawf, nR)`` array that is stored."""
        return self.nnz / (self.nawf * self.nawf * len(self.R_int))

    def hermiticity_error(self) -> float:
        """Max ``|H_ij(R) - conj(H_ji(-R))|`` over stored bonds.

        Returns
        -------
        float
            The largest Hermiticity violation, in eV; 0.0 for an empty list.

        Raises
        ------
        RuntimeError
            If the bond arrays have been released by :meth:`compact`.

        Notes
        -----
        Diagnostic counterpart of :meth:`hermitize`: it measures how far the
        stored list is from Hermitian without changing it.  Each bond is
        looked up against its ``(j, i, -R)`` partner through a sorted search
        on the packed bond key.  A bond whose partner was dropped by
        thresholding has nothing to cancel against, so it contributes its own
        magnitude to the error.

        Fully vectorized: usable as a production assertion at tens of
        millions of bonds.
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

        err = np.where(
            found,
            np.abs(self.vals - np.conj(self.vals[partner])).max(axis=1),
            np.abs(self.vals).max(axis=1),
        )
        return float(err.max())

    def bytes_per_bond(self) -> tuple[int, int, int]:
        """Bytes per bond for the three states a bond list passes through.

        Returns
        -------
        (container, plan, hermitize_peak) : tuple of int
            All per bond of the list being described.

        Notes
        -----
        Derived from the actual allocations, so these track the
        implementation rather than a remembered constant:

        - ``container``: ``rows`` + ``cols`` + ``ridx`` (int32) + ``vals``
          (complex128 per spin) + ``dnm`` (3 float64).
        - ``plan``: what survives :meth:`compact` — ``ridx`` (int64),
          ``vals`` (complex128 per spin), ``rcoef`` (3 float64), plus the
          CSR pattern (``seg_starts`` int64 + ``indices`` int32, at most
          one per bond).
        - ``hermitize_peak``: everything :meth:`hermitize` holds at once,
          counted per *input* bond — it works on ``2*nnz`` rows, so this is
          the high-water mark of the whole doubling stage and the number
          that decides whether a run fits.  Terms: ``triples`` +
          ``mirrored`` (24 each), ``rows_all``/``cols_all`` (16),
          ``triples_all`` (48), ``vals`` (32/spin), ``dnm`` (48), ``code``
          (16), ``np.unique``'s sorted copy and argsort (32), ``inv`` +
          ``first`` + ``uniq_code`` (48), the output arrays (~112), and the
          input container that stays alive throughout.
        """
        container = 4 + 4 + 4 + 16 * self.nspin + 24
        plan = 8 + 16 * self.nspin + 24 + 8 + 4
        hermitize_peak = 24 + 24 + 16 + 48 + 32 * self.nspin + 48 + 16 + 32 + 48 + 112 + container
        return container, plan, hermitize_peak

    def project_doubling(self, nx: int, ny: int, nz: int) -> dict[str, Any]:
        """Project the cost of ``doubling_Hamiltonian(nx, ny, nz)``.

        Parameters
        ----------
        nx, ny, nz : int
            Number of doublings along each axis.

        Returns
        -------
        dict
            ``d`` (total doublings), ``N`` (cell multiplier ``2**d``),
            ``nawf`` and ``nnz`` after doubling, ``steady_bytes`` (the
            resident cost of the final plan), ``peak_bytes`` (the
            high-water mark during hermitization) and ``dense_hk_bytes``
            (one dense ``H(k)`` at the final size, for comparison).

        Raises
        ------
        ValueError
            If any doubling count is negative.

        Notes
        -----
        Doubling is pure index arithmetic that replicates every bond
        exactly twice per step (see :mod:`PAOFLOW.sparse.doubling`), so the
        final bond count is ``nnz * 2**(nx+ny+nz)`` — exact, not a
        heuristic.  ``nawf`` scales the same way.  The only estimated
        quantity is the peak, which assumes ``hermitize`` is the
        high-water mark of the stage (it is: the per-step ``double_axis``
        transients act on lists at most half as long).

        ``nnz`` is the count *entering* hermitization, which is what the
        peak scales with.  Hermitization then takes the union of the list
        with its mirrored conjugate, so the steady-state count lands
        between ``nnz`` (list already Hermitian, every partner present)
        and ``2*nnz`` (no partner present at all); thresholding drops
        partners asymmetrically, so growth is the norm — measured 1.19x on
        example01 at nx=1.  ``steady_bytes`` therefore carries
        ``HERMITIZE_GROWTH``.

        Running this before the doubling itself is the point: it lets a run
        fail immediately with a number, instead of after hours, in the
        out-of-memory killer.
        """
        d = int(nx) + int(ny) + int(nz)
        if d < 0:
            raise ValueError('project_doubling: negative doubling counts')
        N = 1 << d
        nnz_final = self.nnz * N
        nawf_final = self.nawf * N
        _, plan_b, peak_b = self.bytes_per_bond()
        return {
            'd': d,
            'N': N,
            'nawf': nawf_final,
            'nnz': nnz_final,
            'steady_bytes': int(nnz_final * HERMITIZE_GROWTH) * plan_b,
            'peak_bytes': nnz_final * peak_b,
            'dense_hk_bytes': 16 * nawf_final * nawf_final,
        }

    def stats_line(self) -> str:
        """One-line summary of size, sparsity and truncation quality.

        Returns
        -------
        str
            Orbital count, number of lattice vectors, bond count, density,
            memory, the dense-array equivalent, and the rigorous bound on
            the eigenvalue shift caused by thresholding.
        """
        mbytes = self.nnz * (4 + 4 + 4 + 16 * self.nspin + 24) / 1024**2
        dense_mbytes = (self.nawf**2 * np.prod(self.nk_grid) * self.nspin * 16) / 1024**2
        return (
            f'Sparse H(R): nawf={self.nawf}, nR={len(self.R_int)}, nnz={self.nnz / 1e6:.3g}M, '
            f'density={self.density():.2e}, mem={mbytes:.1f} MB '
            f'(dense equivalent {dense_mbytes:.1f} MB), '
            f'truncation eig-shift bound={self.drop_report.get("eig_bound", 0.0):.2e} eV'
        )


def _minus_R_index(R_int: np.ndarray, nk_grid: Sequence[int]) -> np.ndarray:
    """Index of ``-R`` for each row of ``R_int``.

    Parameters
    ----------
    R_int : np.ndarray, shape (nR, 3)
        Folded lattice triples.
    nk_grid : sequence of int
        The three R-grid dimensions, which define the folding window.

    Returns
    -------
    np.ndarray, shape (nR,), int64
        Row index of ``-R`` within ``R_int``, or ``-1`` where ``-R`` is not
        present at all, which a thresholded bond list can produce.

    Notes
    -----
    Negating a folded lattice vector can leave the window, so components at
    ``+nk/2`` on an even axis are folded back onto ``-nk/2``; the Nyquist
    plane is therefore its own negative.  The window comes from ``nk_grid``
    rather than from the extent of ``R_int``, so a sparse R list folds
    correctly.  The lookup itself is a table over the whole FFT box, built
    from the packed codes of :func:`encode_R`.
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
