"""Bond-list representation of the PAO real-space Hamiltonian.

:class:`SparseHamiltonian` stores ``H(R)`` as a flat list of bonds ``(i, j, R, value)`` shared
across all lattice vectors, thresholded on ``|value|``, and assembles

.. math::

    H_{ij}(\\mathbf{k}) = \\sum_{\\mathbf{R}} H_{ij}(\\mathbf{R})
        \\, e^{s \\, 2\\pi i \\, \\mathbf{k} \\cdot \\mathbf{R}}

with ``s = +1`` the dense band-path convention (``spectrum.do_bands.band_loop_H``) and
``s = -1`` the dense FFT mesh convention (``scipy.fftpack.fftn`` in
``hamiltonian.do_double_grid`` / ``do_gradient``).  The CSR pattern is built once
(:meth:`SparseHamiltonian._build_plan`); per k only the values are refilled.  ``Dnm`` is kept
per bond, never as a dense ``(nawf, nawf, 3)`` array.

Nyquist split: on an even grid the folded plane ``m = -nk/2`` has no ``+nk/2`` partner, so the
Fourier sum is slightly non-Hermitian off the original grid, while the iterative solver needs
an exactly Hermitian ``H(k)``.  The plan splits every such bond into ``-nk/2`` and ``+nk/2`` at
half weight (the bond-list ``utils.zero_pad``): grid values are unchanged, off-grid ``H(k)`` is
exactly Hermitian.  The stored list stays on the folded grid, which doubling operates on.

``HERMITIZE_GROWTH``: bond-count growth across :meth:`SparseHamiltonian.hermitize`, for size
projection only.  The union with the mirrored conjugate puts it in ``[1, 2]``; thresholding
drops partners asymmetrically, measured 1.19 on example01 at nx=1, rounded up here.
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
    """Per-axis shift ``nk//2`` mapping folded ``[-(nk//2), (nk-1)//2]`` onto ``[0, nk)``."""
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
        One code per triple, in ``[0, nk1*nk2*nk3)``.

    Notes
    -----
    Mixed radix, most significant axis first, so code order equals the lexicographic order of
    ``np.unique(..., axis=0)`` and results stay bit-identical to the sort-based path.
    """
    _, n2, n3 = (int(n) for n in nk_grid)
    o1, o2, o3 = _r_offsets(nk_grid)
    t = np.asarray(triples, dtype=np.int64)
    return ((t[:, 0] + o1) * n2 + (t[:, 1] + o2)) * n3 + (t[:, 2] + o3)


def decode_R(codes: np.ndarray, nk_grid: Sequence[int]) -> np.ndarray:
    """Exact inverse of :func:`encode_R`.

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
        Distinct triples in lexicographic order ``(nR, 3)``, and each row's index into them.

    Raises
    ------
    AssertionError
        If a triple lies outside the folded window, i.e. is not representable on the R grid.

    Notes
    -----
    A presence mask over the FFT box replaces sorting; output is bit-identical to the ``axis=0``
    form because :func:`encode_R` preserves lexicographic order.
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
        One key per bond, unique over ``(orbital, orbital, R)``.

    Raises
    ------
    NotImplementedError
        If ``nawf**2 * nR >= 2**62`` (key overflows int64).

    Notes
    -----
    Row and column are the most significant digits, so key order matches
    ``np.unique(key, axis=0)`` on the 5-column array, down to the floating-point summation order of
    the plan.  8 bytes per bond instead of 40, and an order of magnitude faster to sort.
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
        Row ``n = k + j*nk3 + i*nk2*nk3`` is the folded triple ``(m1, m2, m3)``.

    Notes
    -----
    FFT index ``i >= nk/2`` is the cell ``i - nk``, as in ``utils.get_R_grid_fft``, so a triple is
    directly the displacement used in the phase and the bond length.
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


def nyquist_plane(triples: np.ndarray, nk: int, axis: int) -> np.ndarray | None:
    """Mask of the bonds on the unsplit Nyquist plane ``-nk/2`` of ``axis``.

    ``None`` when there is nothing to split: an odd ``nk``, no bond on the
    plane, or a list that already carries the ``+nk/2`` plane (hermitized
    with an explicit partner, or split before).
    """
    if nk % 2 != 0 or (triples[:, axis] == nk // 2).any():
        return None
    hit = triples[:, axis] == -(nk // 2)
    return hit if hit.any() else None


def split_nyquist_axis(rows, cols, triples, vals, dnm, nk, axis):
    """Split the ``-nk/2`` plane of ``axis`` into ``-nk/2`` and ``+nk/2`` at half weight (see
    the module docstring); arrays come back unchanged when :func:`nyquist_plane` finds nothing."""
    hit = nyquist_plane(triples, nk, axis)
    if hit is None:
        return rows, cols, triples, vals, dnm
    vals = vals.copy()
    vals[hit] *= 0.5
    mirrored = triples[hit].copy()
    mirrored[:, axis] = nk // 2
    return (
        np.concatenate((rows, rows[hit])),
        np.concatenate((cols, cols[hit])),
        np.concatenate((triples, mirrored)),
        np.concatenate((vals, vals[hit])),
        np.concatenate((dnm, dnm[hit])),
    )


class SparseHamiltonian:
    """Thresholded bond list for H(R) with fixed-pattern k-space assembly.

    Only the surviving hoppings are stored, as parallel per-bond arrays, so doubling costs 2x memory
    instead of the dense 4x.  The k-independent structure is built once (:attr:`plan`) and reused at
    every k-point.

    Parameters
    ----------
    nawf : int
        Number of atomic orbitals in the current cell.
    nspin : int
        Number of spin channels.
    alat : float
        Lattice constant in Bohr.
    a_vectors : array_like, shape (3, 3)
        Current-cell lattice vectors (rows) in units of ``alat``; updated by doubling.
    nk_grid : sequence of int
        The three R-grid dimensions; fixed by the DFT k-mesh, unchanged by doubling.
    R_int : array_like, shape (nR, 3)
        Distinct folded lattice triples, in the current-cell basis.
    rows, cols : array_like, shape (nnz,)
        Orbital indices of each bond.
    ridx : array_like, shape (nnz,)
        Index into ``R_int`` for each bond.
    vals : array_like, shape (nnz, nspin)
        Hamiltonian matrix element of each bond, in eV.
    dnm : array_like, shape (nnz, 3)
        Intra-cell orbital position difference of each bond, in Bohr (one value per orbital pair;
        for doubled cells see :func:`~PAOFLOW.sparse.doubling.double_axis`).
    threshold : float, optional
        Magnitude below which H(R) entries were dropped (eV).
    drop_report : dict or None, optional
        Truncation statistics; ``eig_bound`` is a rigorous bound (eV) on the eigenvalue shift from
        truncation at every k (see :meth:`from_data_controller`).

    Attributes
    ----------
    R_int, rows, cols, ridx : np.ndarray, int32
    vals : np.ndarray, complex128
    dnm : np.ndarray, float64
        The parameters, converted; ``rows``/``cols``/``vals``/``dnm`` are ``None`` after
        :meth:`compact`.
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
        cls,
        data_controller: DataController,
        threshold: float | None = None,
        rcut: float | None = None,
        bond_order: int | None = None,
        distance_tol: float = 1.0e-3,
    ) -> SparseHamiltonian:
        """Threshold the dense ``HRs`` in a DataController into a bond list.

        Parameters
        ----------
        data_controller : DataController
            Holds ``arrays['HRs']`` from ``pao_hamiltonian``, ``a_vectors``, ``alat``, ``tau`` and,
            when present, ``Dnm``.
        threshold : float or None, optional
            Drop magnitude in eV.  ``None``: 1e-3 eV without a real-space cutoff, no element cut
            with one; a positive value is exclusive with ``rcut``/``bond_order``
            (:func:`PAOFLOW.sparse.config.resolve_threshold`).
        rcut : float or None, optional
            Bond-length cutoff in Bohr, snapped into the gap above the outermost shell it reaches
            (:func:`PAOFLOW.sparse.shells.snap_cutoff`); ``drop_report`` keeps ``rcut`` and
            ``rcut_requested``.
        bond_order : int or None, optional
            Keep bonds up to this neighbour shell (``1`` = nearest), resolved by
            :func:`PAOFLOW.sparse.shells.shell_cutoff`.  Exclusive with ``rcut``.
        distance_tol : float, optional
            Shell-merging tolerance (Bohr) for ``bond_order`` and the ``rcut`` snap.

        Returns
        -------
        SparseHamiltonian
            The thresholded bond list for the base cell.

        Raises
        ------
        ValueError
            If both ``rcut`` and ``bond_order`` are given, a positive ``threshold`` is combined with
            either, ``bond_order`` exceeds the shells the grid can represent, ``rcut`` exceeds the
            aliasing-safe radius, or either cutoff reaches a bond the grid stores at a longer image
            than its shortest one (see :mod:`PAOFLOW.sparse.shells`).

        Notes
        -----
        The single dense-to-sparse boundary, for the small base cell; the caller deletes
        ``arrays['HRs']`` afterwards.  Deterministic on every rank (``HRs`` is broadcast), no MPI.

        A hopping is kept if any spin channel passes, so both spins share one pattern and plan.

        ``eig_bound``: the dropped part is Hermitian at every k, and Gershgorin bounds its norm by
        the largest absolute row sum, a rigorous bound on any eigenvalue shift.  Cutoffs are part of
        the keep mask, so the bound covers them.

        The element threshold breaks symmetry: equivalent bonds rotate orbitals into each other, so
        the cut keeps different elements of each and protected degeneracies split.  A length cutoff
        keeps or drops whole atom-pair blocks, which space-group operations preserve; hence they
        are exclusive.

        On the folded Nyquist plane ``-R`` maps onto ``R``, so the Hermitian partners ``(i,j,R)``
        and ``(j,i,-R)`` have different lengths (tens of Bohr apart); a bond is kept if either is
        inside the cutoff, or Hermiticity breaks.  ``double_axis`` preserves bond lengths, so a
        base-cell cutoff carries over to doubled cells unchanged.
        """
        from .config import resolve_threshold
        from .shells import (
            aliasing_safe_radius,
            misplaced_bond_length,
            shell_cutoff,
            snap_cutoff,
        )

        arry, attr = data_controller.data_dicts()
        HRs = arry['HRs']
        nawf, _, nk1, nk2, nk3, nspin = HRs.shape

        if bond_order is not None and rcut is not None:
            raise ValueError('Give either rcut (Bohr) or bond_order (shells), not both.')
        threshold = resolve_threshold(threshold, rcut, bond_order)
        rcut_requested = None if rcut is None else float(rcut)
        if bond_order is not None:
            rcut, shells = shell_cutoff(data_controller, bond_order, distance_tol)
        lattice = np.asarray(arry['a_vectors'], dtype=float) * attr['alat']
        safe_radius = aliasing_safe_radius(lattice, (nk1, nk2, nk3))
        grid_name = f'{nk1}x{nk2}x{nk3}'

        flat = HRs.reshape(nawf, nawf, nk1 * nk2 * nk3, nspin)
        mag = np.abs(flat).max(axis=3)
        keep = mag > threshold
        if rcut is not None:
            R_int = folded_R_triples(nk1, nk2, nk3)
            Rcart = (R_int.astype(float) @ arry['a_vectors']) * attr['alat']
            Dnm = arry['Dnm'] if 'Dnm' in arry else _orbital_offsets(data_controller)
            dist = np.linalg.norm(Dnm[:, :, None, :] + Rcart[None, None, :, :], axis=3)
            minus = _minus_R_index(R_int, (nk1, nk2, nk3))
            dist = np.minimum(dist, dist.transpose(1, 0, 2)[:, :, minus])
            if bond_order is None:
                # shells are only counted inside the safe radius, so only
                # an explicit radius can get past it
                if rcut_requested > safe_radius:
                    raise ValueError(
                        f'rcut = {rcut_requested:.3f} Bohr exceeds the aliasing-safe radius '
                        f'{safe_radius:.3f} Bohr of the {grid_name} grid. Beyond it one grid '
                        'value of H(R) mixes bonds of different lengths, so no length cut is '
                        'defined there. Use a smaller rcut or a denser k-grid.'
                    )
                rcut = snap_cutoff(dist, rcut, distance_tol)
            limit = misplaced_bond_length(Dnm, lattice, (nk1, nk2, nk3), distance_tol)
            if rcut >= limit:
                what = (
                    f'rcut = {rcut_requested:.3f} Bohr'
                    if bond_order is None
                    else f'bond_order = {bond_order} (cutoff {rcut:.3f} Bohr)'
                )
                fix = 'a smaller rcut'
                if bond_order is not None:
                    fix = f'bond_order <= {int(np.sum(shells + distance_tol < limit))}'
                raise ValueError(
                    f'{what} reaches bonds of length {limit:.3f} Bohr that the {grid_name} '
                    'grid stores at a longer periodic image than their shortest one. The cut '
                    'would keep only part of their neighbour shell and break the crystal '
                    f'symmetry. Keep the cutoff below {limit:.3f} Bohr ({fix}) or use a '
                    'denser k-grid.'
                )
            keep &= dist <= float(rcut)
        rows, cols, ridx = np.nonzero(keep)

        dropped = np.where(keep, 0.0, mag)
        row_sums = dropped.sum(axis=(1, 2))
        drop_report = {
            'threshold': float(threshold),
            'rcut': None if rcut is None else float(rcut),
            'rcut_requested': rcut_requested,
            'bond_order': None if bond_order is None else int(bond_order),
            'aliasing_safe_radius': safe_radius,
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
        """Return ``(rows, cols, triples, vals, dnm)`` with every ``-nk/2`` plane split.

        The split and its reason are in the module docstring; it applies per even axis and leaves
        the stored list untouched.  A hermitized list already carries ``+nk/2`` and is not split
        again (:func:`nyquist_plane`).
        """
        rows = self.rows
        cols = self.cols
        triples = self.R_int[self.ridx].astype(np.int64)
        vals = self.vals
        dnm = self.dnm
        for axis in range(3):
            rows, cols, triples, vals, dnm = split_nyquist_axis(
                rows, cols, triples, vals, dnm, self.nk_grid[axis], axis
            )
        return rows, cols, triples, vals, dnm

    def _build_plan(self) -> None:
        """Turn the bond list into the k-independent :attr:`plan`, once.

        Bonds are Nyquist-split and grouped by matrix entry ``(i, j)``; distinct R are kept apart,
        so a k-point costs one phase per R.  Per spin ``V[e, R] = H_e(R)``.  The gradient
        coefficient ``alat * Rcart + dnm`` splits into an R-only part, carried as extra right-hand
        sides of ``V @ phase``, and ``dnm``, one value per entry, applied after the sum; so ``H(k)``
        and its derivatives come from one pass over ``V``.
        """
        rows, cols, triples, vals, dnm = self._nyquist_split()

        R_uniq, ridx = np.unique(triples, axis=0, return_inverse=True)
        ridx = ridx.reshape(-1)

        pair = rows.astype(np.int64) * self.nawf + cols
        order = np.argsort(pair, kind='stable')
        pair = pair[order]

        new_entry = np.r_[True, pair[1:] != pair[:-1]]
        seg_starts = np.flatnonzero(new_entry)
        entry = np.cumsum(new_entry) - 1
        upair = pair[seg_starts]
        indices = (upair % self.nawf).astype(np.int32)
        counts = np.bincount((upair // self.nawf).astype(np.intp), minlength=self.nawf)
        indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)

        dnm = dnm[order]
        entry_dnm = dnm[seg_starts]
        if not np.array_equal(dnm, entry_dnm[entry]):
            raise ValueError(
                'SparseHamiltonian: dnm differs between bonds of the same orbital pair; '
                'it is the intra-cell offset of the pair and must be one value per (i, j).'
            )

        shape = (len(seg_starts), len(R_uniq))
        ridx = ridx[order]
        V = [csr_matrix((vals[order, s], (entry, ridx)), shape=shape) for s in range(self.nspin)]

        self._plan = {
            'indices': indices,
            'indptr': indptr,
            'R_uniq': R_uniq,
            'Rcart': R_uniq.astype(float) @ self.a_vectors,
            'V': V,
            'dnm': entry_dnm,
        }

    @property
    def plan(self) -> dict[str, np.ndarray]:
        """The k-independent assembly plan, built on first access.

        Keys: ``indices``, ``indptr`` (CSR pattern of ``H(k)``); ``R_uniq``, ``Rcart`` (distinct
        lattice vectors, fractional and Cartesian in ``alat``); ``V`` (per spin, sparse
        ``(n_entries, nR)`` values, entries in pattern order); ``dnm`` (per-entry intra-cell offset,
        Bohr).
        """
        if self._plan is None:
            self._build_plan()
        return self._plan

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
        Frees ``rows``/``cols``/``dnm``/``vals``, about two thirds of the steady-state bond memory
        (:meth:`bytes_per_bond`).  Call after the last ``double_axis`` and ``hermitize``:
        irreversibly, :meth:`hermitize`, :meth:`hermiticity_error`, :meth:`to_dense_HRs` and
        :func:`~PAOFLOW.sparse.doubling.double_axis` raise afterwards.  Assembly is unaffected.
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
        """Dimensionless ``k.R`` per distinct lattice vector, shape ``(nR,)``.

        ``cart=False``: ``kvec`` in crystal coordinates, ``k.R = kfrac . m``.  ``cart=True``:
        ``kvec`` Cartesian in ``2 pi / alat`` (dense band path after the ``b_vectors`` rotation in
        ``do_bands``), dotted with the Cartesian R of the *current* cell.  This replicates
        ``band_loop_H`` on doubled cells, where ``b_vectors`` stay those of the original cell.
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
            Fourier phase sign: ``-1`` FFT mesh convention, ``+1`` band-path convention.
        cart : bool, optional
            Whether ``kvec`` is Cartesian; see :meth:`_phase_arg`.

        Returns
        -------
        scipy.sparse.csr_matrix, shape (nawf, nawf)
            The Bloch Hamiltonian at this k-point, in eV.
        """
        p = self.plan
        phase = np.exp((sign * 2.0j * np.pi) * self._phase_arg(kvec, cart))
        data = p['V'][ispin] @ phase
        return csr_matrix((data, p['indices'], p['indptr']), shape=(self.nawf, self.nawf))

    def assemble_derivatives(
        self,
        kvec: np.ndarray,
        ispin: int = 0,
        sign: int = -1,
        cart: bool = False,
        order: int = 1,
        with_dnm: bool = True,
    ) -> tuple[csr_matrix, ...]:
        """Assemble ``H(k)`` and its k-derivatives up to second order.

        Parameters
        ----------
        kvec, ispin, sign, cart
            As in :meth:`assemble_hk`.
        order : {1, 2}, optional
            Highest derivative returned.
        with_dnm : bool, optional
            Include the intra-cell ``Dnm`` terms.  ``False`` gives the pure lattice-sum derivatives
            the dense kernels build from ``Rfft`` alone (``do_berry_curvature``, the second
            derivative of ``do_topology``).

        Returns
        -------
        (hk, dhk) or (hk, dhk, d2hk) : tuple
            ``H(k)``; ``[dH/dk_0, dH/dk_1, dH/dk_2]``; for ``order=2`` the six
            :math:`d^2H/dk_i dk_j` in the order ``xx, yy, zz, xy, xz, yz``
            (``hamiltonian.do_d2Hd2k.IJ_PAIRS``).  All CSR, shape ``(nawf, nawf)``.

        Raises
        ------
        ValueError
            If ``order`` is not 1 or 2.

        Notes
        -----
        Per bond, ``dH/dk_l`` carries ``1j * (alat * Rcart_l + Dnm_l)`` as in
        ``hamiltonian.do_gradient`` (cell displacement plus intra-cell orbital offset), and
        ``d2H/dk_i dk_j`` carries ``-(alat * Rcart_i + Dnm_i) * (alat * Rcart_j + Dnm_j)``, the
        four-term FFT of ``do_d2Hd2k.do_d2Hd2k_ij``.  One pass, see :meth:`_build_plan`.
        """
        from ..hamiltonian.do_d2Hd2k import IJ_PAIRS

        if order not in (1, 2):
            raise ValueError(f'assemble_derivatives: order must be 1 or 2, got {order!r}')
        p = self.plan
        phase = np.exp((sign * 2.0j * np.pi) * self._phase_arg(kvec, cart))
        ncol = 4 if order == 1 else 10
        rhs = np.empty((len(phase), ncol), dtype=np.complex128)
        rhs[:, 0] = phase
        R = self.alat * p['Rcart']
        np.multiply(R, phase[:, None], out=rhs[:, 1:4])
        if order == 2:
            for ij, (i, j) in enumerate(IJ_PAIRS):
                rhs[:, 4 + ij] = R[:, i] * R[:, j] * phase
        sums = p['V'][ispin] @ rhs

        shape = (self.nawf, self.nawf)
        h = sums[:, 0]
        dnm = p['dnm'] if with_dnm else np.zeros_like(p['dnm'])
        hk = csr_matrix((h, p['indices'], p['indptr']), shape=shape)
        dhk = [
            csr_matrix(
                (1j * (sums[:, 1 + l] + dnm[:, l] * h), p['indices'], p['indptr']),
                shape=shape,
            )
            for l in range(3)
        ]
        if order == 1:
            return hk, dhk
        d2hk = [
            csr_matrix(
                (
                    -(
                        sums[:, 4 + ij]
                        + dnm[:, i] * sums[:, 1 + j]
                        + dnm[:, j] * sums[:, 1 + i]
                        + dnm[:, i] * dnm[:, j] * h
                    ),
                    p['indices'],
                    p['indptr'],
                ),
                shape=shape,
            )
            for ij, (i, j) in enumerate(IJ_PAIRS)
        ]
        return hk, dhk, d2hk

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
        Each bond ``(i, j, R, v)`` is paired with ``(j, i, -R, conj(v))`` at half weight and
        coinciding bonds are summed; ``-R`` is folded back (``+nk/2 -> -nk/2``) so the plan's
        Nyquist split applies.  With that split this equals ``(H(k) + H(k)^dagger)/2`` at every k,
        which dense ``gradient_and_momenta`` applies per k.  Needed because doubling (bond-for-bond
        the dense kernel) maps the self-paired Nyquist plane asymmetrically; an exact no-op on a
        Hermitian list.  ``dnm`` is per pair, so it is assigned, not summed, with
        ``Dnm_ji = -Dnm_ij`` on the mirror.  Grouping uses the int64 key of :func:`encode_bond`,
        as this is the memory peak of doubling.
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
        A bond whose partner was thresholded away counts its own magnitude.  Vectorized, usable as a
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

        err = np.where(
            found,
            np.abs(self.vals - np.conj(self.vals[partner])).max(axis=1),
            np.abs(self.vals).max(axis=1),
        )
        return float(err.max())

    def to_dense_HRs(self) -> np.ndarray:
        """Scatter the bond list back onto the dense FFT real-space grid.

        Returns
        -------
        np.ndarray, shape (nawf, nawf, nk1, nk2, nk3, nspin), complex128
            The truncated ``H(R)``; elements outside the stored bonds are exactly zero.

        Raises
        ------
        RuntimeError
            If the bond arrays have been released by :meth:`compact`.

        Notes
        -----
        Validation tool for the base cell (truncation convergence checks): it allocates the
        ``O(nawf^2 * nR)`` tensor the sparse pipeline never forms.  Never call on a doubled cell.
        """
        self._require_bonds('to_dense_HRs')
        HRs = np.zeros((self.nawf, self.nawf) + self.nk_grid + (self.nspin,), dtype=np.complex128)
        cell = np.mod(self.R_int[self.ridx], self.nk_grid)
        HRs[self.rows, self.cols, cell[:, 0], cell[:, 1], cell[:, 2], :] = self.vals
        return HRs

    def bytes_per_bond(self) -> tuple[int, int, int]:
        """Bytes per bond for the three states a bond list passes through.

        Returns
        -------
        (container, plan, hermitize_peak) : tuple of int
            All per bond of the list being described.

        Notes
        -----
        - ``container``: ``rows`` + ``cols`` + ``ridx`` (int32), ``vals`` (complex128 per spin),
          ``dnm`` (3 float64).
        - ``plan``: what survives :meth:`compact`: per spin ``V`` (complex128 + int32 index), plus
          CSR ``indices`` (int32, at most one per bond); per-entry and per-R arrays are not per
          bond.
        - ``hermitize_peak``: all :meth:`hermitize` holds at once per *input* bond, the high-water
          mark of doubling: ``triples`` + ``mirrored`` (24 each), ``rows_all``/``cols_all`` (16),
          ``triples_all`` (48), ``vals`` (32/spin), ``dnm`` (48), ``code`` (16), ``np.unique`` sort
          copy + argsort (32), ``inv`` + ``first`` + ``uniq_code`` (48), outputs (~112), plus the
          input container.
        """
        container = 4 + 4 + 4 + 16 * self.nspin + 24
        plan = (16 + 4) * self.nspin + 4
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
            ``d`` (total doublings), ``N`` (cell multiplier ``2**d``), ``nawf`` and ``nnz`` after
            doubling, ``steady_bytes`` (resident final plan), ``peak_bytes`` (hermitization
            high-water mark) and ``dense_hk_bytes`` (one dense ``H(k)`` at the final size).

        Raises
        ------
        ValueError
            If any doubling count is negative.

        Notes
        -----
        Run before doubling, to fail fast with a number rather than in the OOM killer.  ``nnz`` is
        exact: each doubling replicates every bond twice, and the first doubling of an axis first
        splits its ``-nk/2`` plane, doubling the bonds on it.  It counts bonds *entering*
        hermitization; ``steady_bytes`` applies ``HERMITIZE_GROWTH``.  Only the peak is estimated,
        assuming ``hermitize`` is the high-water mark (``double_axis`` transients act on lists at
        most half as long).
        """
        d = int(nx) + int(ny) + int(nz)
        if d < 0:
            raise ValueError('project_doubling: negative doubling counts')
        N = 1 << d
        triples = self.R_int[self.ridx]
        copies = np.ones(self.nnz, dtype=np.int64)
        for axis, reps in enumerate((nx, ny, nz)):
            nk = self.nk_grid[axis]
            hit = nyquist_plane(triples, nk, axis) if reps and nk >= 4 else None
            if hit is not None:
                copies += copies * hit
        nnz_final = int(copies.sum()) * N
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

    def stats(self) -> dict:
        """Size, sparsity and truncation quality of the bond list.

        Returns
        -------
        dict
            ``nawf``, ``nR``, ``nnz``, ``total`` (elements of dense ``H(R)``), ``density``,
            ``mbytes`` (bond list), ``dense_mbytes`` (dense ``H(R)``) and ``eig_bound``, the
            rigorous bound (eV) on the truncation eigenvalue shift at any k-point.
        """
        nR = len(self.R_int)
        return {
            'nawf': self.nawf,
            'nR': nR,
            'nnz': self.nnz,
            'total': self.nawf**2 * nR,
            'density': self.density(),
            'mbytes': self.nnz * (4 + 4 + 4 + 16 * self.nspin + 24) / 1024**2,
            'dense_mbytes': (self.nawf**2 * np.prod(self.nk_grid) * self.nspin * 16) / 1024**2,
            'eig_bound': float(self.drop_report.get('eig_bound', 0.0)),
        }

    def stats_line(self) -> str:
        """One-line summary of :meth:`stats`."""
        st = self.stats()
        return (
            f'Sparse H(R): nawf={st["nawf"]}, nR={st["nR"]}, nnz={st["nnz"] / 1e6:.3g}M, '
            f'density={st["density"]:.2e}, mem={st["mbytes"]:.1f} MB '
            f'(dense equivalent {st["dense_mbytes"]:.1f} MB), '
            f'truncation eig-shift bound={st["eig_bound"]:.2e} eV'
        )


def _minus_R_index(R_int: np.ndarray, nk_grid: Sequence[int]) -> np.ndarray:
    """Row index of ``-R`` in ``R_int`` for each row, ``-1`` where absent.

    A thresholded list can lack ``-R``.  ``+nk/2`` on an even axis folds back to ``-nk/2``, so the
    Nyquist plane is its own negative; the window comes from ``nk_grid``, not from ``R_int``.
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


def _orbital_offsets(data_controller: DataController) -> np.ndarray:
    """``tau_i - tau_j`` per orbital pair, shape ``(nawf, nawf, 3)``, in Bohr.

    Needs ``tau`` and an orbital map :func:`PAOFLOW.sparse.io.build_orbital_basis_table` can
    expand.  Geometry only, for the bond-length cutoff when the controller has no ``Dnm``; the
    per-bond ``dnm`` of the gradient then stays zero.
    """
    from .io import build_orbital_basis_table

    tau = np.asarray(data_controller.data_arrays['tau'], dtype=float)
    centres = tau[build_orbital_basis_table(data_controller)['orbital_atom']]
    return centres[:, None, :] - centres[None, :, :]
