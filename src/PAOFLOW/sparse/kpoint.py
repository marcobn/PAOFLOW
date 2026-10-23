"""Per-k context handed to every mesh-pass consumer.

A :class:`KPoint` holds one k-point's solve while it is live — the
eigenpairs, the assembled ``H(k)`` and ``dH/dk``, and the band-diagonal
velocities and widths the mesh stores — and computes everything else a
consumer may ask for on first use, once.  Several consumers at the same
k-point therefore share one momentum tensor, one second-derivative
assembly, one set of degeneracies.

Nothing here may outlive the k-point: the mesh pass drops the object
before the next one, and a consumer must reduce whatever it needs inside
``on_k`` (see :mod:`PAOFLOW.sparse.properties`).  The cached members are
per-k dense scratch, bounded by the solve: ``(nawf, m)`` for ``V``,
``(3, m, m)`` for ``pksp`` with ``m`` the number of states solved.  A
consumer that needs interband sums asks for the full spectrum
(``needs = {'full_spectrum'}``), which makes ``m = nawf`` and is only
admitted while ``nawf <= DENSE_N_MAX``.

When the mesh pass decides that ``dH/dk`` is full enough to be cheaper
dense (``mesh.dense_operators``), :attr:`KPoint.dhk` and :attr:`KPoint.d2hk`
are :class:`DenseOnDemand` sequences: each derivative is densified on first
use, so every kernel multiplies through BLAS instead of scipy's
single-threaded sparse product.  That adds up to three cached
``(nawf, nawf)`` arrays for ``dhk`` plus one transient for a ``d2hk``
component, all freed with the k-point.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from .hamiltonian import SparseHamiltonian


def degenerate_groups(E: np.ndarray, bnd: int) -> list[np.ndarray]:
    """Degenerate groups among the first ``bnd`` states, each state in one group.

    Parameters
    ----------
    E : np.ndarray, shape (m,)
        Eigenvalues at one k-point (eV), ascending.
    bnd : int
        Only groups entirely below ``bnd`` are returned.

    Returns
    -------
    list of np.ndarray
        Contiguous index ranges, disjoint and ascending.

    Notes
    -----
    :func:`~PAOFLOW.spectrum.do_eigh.get_degeneracies` collects, for every
    rounded energy that starts a close pair, all states within tolerance of
    it.  When two rounded energies of one cluster both start a pair, it lists
    the same group twice, or two overlapping groups; ``perturb_split`` then
    rotates those states twice, and the basis no longer diagonalizes
    ``dH/dk`` in the group.  A folded cell hits this whenever a pair split
    by less than the tolerance has an exact copy (on doubled example01, the
    worst k-point of the d=1 dielectric).  The groups are merged here into
    one range per cluster.
    """
    from ..spectrum.do_eigh import get_degeneracies

    merged: list[list[int]] = []
    for g in sorted(get_degeneracies(E[None, :, None], bnd)[0][0], key=lambda g: g[0]):
        if merged and g[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], int(g[-1]))
        else:
            merged.append([int(g[0]), int(g[-1])])
    return [np.arange(lo, hi + 1) for lo, hi in merged]


class DenseOnDemand(Sequence):
    """A list of sparse ``(n, n)`` operators handed out as dense arrays.

    Parameters
    ----------
    operators : sequence of scipy.sparse matrix
        The operators, e.g. the three ``dH/dk_l``.
    cache : bool, optional
        Keep each dense array after its first use.  Without it every
        access densifies again, which suits operators read once (the
        six ``d2H/dk_i dk_j`` of the band curvature) and keeps at most one
        of them alive.

    Notes
    -----
    The per-k kernels shared with the dense pipeline (``perturb_split``,
    ``momentum_k``, ``band_curvature_k``, ...) accept dense and sparse
    operators alike, and only the dense ones reach a threaded BLAS.  On a
    near-dense PAO ``dH/dk`` that is the difference between minutes and
    hours per k-point (see :func:`PAOFLOW.sparse.mesh._times_block`).
    """

    def __init__(self, operators: Sequence[Any], cache: bool = True) -> None:
        self._operators = list(operators)
        self._cache = cache
        self._dense: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self._operators)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = range(len(self))[index]
        dense = self._dense.get(index)
        if dense is None:
            dense = self._operators[index].toarray()
            if self._cache:
                self._dense[index] = dense
        return dense


class KPoint:
    """One k-point of a mesh or path pass, with lazily computed extras.

    Parameters
    ----------
    sparse_h : SparseHamiltonian
        Bond list; reassembled here only for :attr:`d2hk`.
    ik : int
        Local k-point index (into this rank's share of the k-points).
    ispin : int
        Spin channel.
    kvec : np.ndarray, shape (3,)
        The k-point, crystal (``cart=False``) or Cartesian in units of
        ``2 pi / alat`` (``cart=True``).
    E : np.ndarray, shape (m,)
        Every eigenvalue solved at this k-point (eV), ascending.  With a
        ``full_spectrum`` consumer ``m = nawf``.
    V : np.ndarray, shape (nawf, m)
        Matching orthonormal eigenvectors (columns).
    hk : scipy.sparse.csr_matrix, shape (nawf, nawf)
        ``H(k)``.
    dhk : list of scipy.sparse.csr_matrix, or DenseOnDemand
        ``[dH/dk_x, dH/dk_y, dH/dk_z]``.
    vel : np.ndarray, shape (3, bnd)
        Band-diagonal velocities of the window bands.
    delta : np.ndarray, shape (bnd,)
        Adaptive smearing widths of the window bands.
    bnd : int
        Window width: ``E[:bnd]`` are the bands the stored mesh arrays keep.
    afac, dk : float
        Adaptive-smearing prefactor and mean k spacing, for :attr:`delta2`.
    sign : {-1, +1}, optional
        Fourier phase convention of the pass (mesh ``-1``, path ``+1``).
    cart : bool, optional
        Whether ``kvec`` is Cartesian.
    kglobal : int or None, optional
        Index of this k-point in the full mesh (FFT-grid order) or path, for
        consumers that address per-k data of the DFT run.
    with_dnm : bool, optional
        Whether ``dhk`` and :attr:`d2hk` carry the intra-cell ``Dnm`` terms
        (see ``SparseHamiltonian.assemble_derivatives``).
    dense_operators : bool, optional
        Hand out ``dhk`` and :attr:`d2hk` as :class:`DenseOnDemand`.
    resolved : sequence of (np.ndarray, list of np.ndarray), optional
        ``(D, parts)`` for the degenerate groups whose columns of ``V`` the
        mesh pass already rotated (see ``mesh._band_diagonal_velocities``):
        into one eigenbasis of all three ``dH/dk_l``, except on the index
        ranges ``parts``, which still take the per-direction rule.
        :attr:`degen` lists ``parts`` in place of ``D``.

    Notes
    -----
    The dense pipeline computes the same quantities as k-indexed tensors
    (``v_k``, ``pksp``, ``d2Hksp``, ``deltakp2``); here each exists for one
    k-point at a time.  The conventions are the dense ones, so a consumer
    can call the extracted per-k body of a dense kernel on these members
    unchanged: ``degen`` is :func:`~PAOFLOW.spectrum.do_eigh.get_degeneracies`
    restricted to ``bnd`` (merged by :func:`degenerate_groups`, with the
    ``resolved`` groups reduced to their parts), ``pksp`` is
    :func:`~PAOFLOW.hamiltonian.do_momentum.momentum_k`, ``delta2`` is
    :func:`~PAOFLOW.spectrum.do_adaptive_smearing.adaptive_widths` on the
    diagonal of ``pksp``.
    """

    def __init__(
        self,
        sparse_h: SparseHamiltonian,
        ik: int,
        ispin: int,
        kvec: np.ndarray,
        E: np.ndarray,
        V: np.ndarray,
        hk: csr_matrix,
        dhk: list[csr_matrix],
        vel: np.ndarray,
        delta: np.ndarray,
        bnd: int,
        afac: float,
        dk: float,
        sign: int = -1,
        cart: bool = False,
        kglobal: int | None = None,
        with_dnm: bool = True,
        dense_operators: bool = False,
        resolved: Sequence[tuple[np.ndarray, list[np.ndarray]]] = (),
    ) -> None:
        self._sparse_h = sparse_h
        self.ik = ik
        self.ispin = ispin
        self.kvec = kvec
        self.E = E
        self.V = V
        self.hk = hk
        if dense_operators and not isinstance(dhk, DenseOnDemand):
            dhk = DenseOnDemand(dhk)
        self.dhk = dhk
        self.dense_operators = dense_operators
        self.vel = vel
        self.delta = delta
        self.bnd = bnd
        self.afac = afac
        self.dk = dk
        self.sign = sign
        self.cart = cart
        self.kglobal = kglobal
        self.with_dnm = with_dnm
        self.resolved = resolved
        self._projected: dict[int, tuple[Any, np.ndarray]] = {}

    @property
    def nstates(self) -> int:
        """Number of states solved at this k-point (``len(E)``)."""
        return len(self.E)

    @cached_property
    def degen(self) -> list[np.ndarray]:
        """Degenerate groups among the window bands (dense ``degen[ispin][ik]``).

        A group in :attr:`resolved` is replaced by its parts.  Elsewhere
        ``V`` already holds one common eigenbasis of the three ``dH/dk_l``
        there, and the kernels that take ``degen`` (``perturb_split``) would
        rotate each direction on its own again, which on a folded cell pairs
        the matrix elements of one state with the velocities of another.
        """
        groups = degenerate_groups(self.E, self.bnd)
        if not self.resolved:
            return groups
        parts = {tuple(D): p for D, p in self.resolved}
        return [part for g in groups for part in parts.get(tuple(g), [g])]

    @cached_property
    def pksp(self) -> np.ndarray:
        """Momentum matrix ``(3, m, m)`` over every solved state (dense ``pksp[ik]``)."""
        from ..hamiltonian.do_momentum import momentum_k

        return momentum_k(self.dhk, self.V, self.degen)

    @cached_property
    def d2hk(self) -> list[csr_matrix] | DenseOnDemand:
        """The six ``d2H/dk_i dk_j`` (``xx, yy, zz, xy, xz, yz``), assembled on first use."""
        d2hk = self._sparse_h.assemble_derivatives(
            self.kvec,
            ispin=self.ispin,
            sign=self.sign,
            cart=self.cart,
            order=2,
            with_dnm=self.with_dnm,
        )[2]
        return DenseOnDemand(d2hk, cache=False) if self.dense_operators else d2hk

    @cached_property
    def delta_all(self) -> np.ndarray:
        """Adaptive widths ``(m,)`` of every solved state (dense ``deltakp[ik]``).

        ``delta`` covers the window only; interband properties that weight
        every state by an occupation need all of them, from the diagonal of
        ``pksp`` as ``do_adaptive_smearing`` computes them.
        """
        from ..spectrum.do_adaptive_smearing import adaptive_widths

        diagonal = np.ascontiguousarray(np.einsum('lnn->ln', self.pksp))
        return adaptive_widths(diagonal, self.afac, self.dk)

    @cached_property
    def delta2(self) -> np.ndarray:
        """Interband adaptive widths ``(m, m)`` (dense ``deltakp2[ik]``)."""
        from ..spectrum.do_adaptive_smearing import adaptive_widths

        diagonal = np.ascontiguousarray(np.einsum('lnn->ln', self.pksp))
        return adaptive_widths(diagonal, self.afac, self.dk, pairs=True)[1]

    def project(self, operator: Any) -> np.ndarray:
        """``V^dagger O V`` over the solved states, degeneracies resolved by ``O``.

        Parameters
        ----------
        operator : np.ndarray or scipy.sparse matrix, shape (nawf, nawf)
            Operator in the PAO basis (e.g. one Cartesian component of the
            spin operator).

        Returns
        -------
        np.ndarray, shape (m, m)
            The projection by :func:`~PAOFLOW.utils.perturb_split.perturb_split`
            with ``operator`` as both arguments, the convention of the dense
            texture kernels.  Cached per operator object for this k-point.
        """
        from ..utils.perturb_split import perturb_split

        key = id(operator)
        hit = self._projected.get(key)
        if hit is not None and hit[0] is operator:
            return hit[1]
        projected, _ = perturb_split(operator, operator, self.V, self.degen)
        self._projected[key] = (operator, projected)
        return projected
