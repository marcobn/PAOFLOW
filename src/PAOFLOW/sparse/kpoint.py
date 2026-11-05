"""Per-k context handed to every mesh-pass consumer.

A :class:`KPoint` holds one k-point's solve and computes the extras on first use, once, so
consumers at the same k share one momentum tensor, ``d2hk`` assembly and degeneracy list.

It must not outlive the k-point: consumers reduce inside ``on_k``. Cached scratch is bounded by
the solve, ``(nawf, m)`` for ``V`` and ``(3, m, m)`` for ``pksp``; ``needs = {'full_spectrum'}``
makes ``m = nawf``, admitted only while ``nawf <= DENSE_N_MAX``. With dense operators
(``mesh.dense_operators``) ``dhk``/``d2hk`` are :class:`DenseOnDemand`: up to three cached
``(nawf, nawf)`` arrays plus one transient ``d2hk`` component, freed with the k-point.
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
    """Degenerate groups of ascending ``E`` within the first ``bnd`` states, as disjoint
    ascending index ranges.

    :func:`~PAOFLOW.spectrum.do_eigh.get_degeneracies` can list one cluster twice or as
    overlapping groups (a sub-tolerance pair with an exact copy on a folded cell), and
    ``perturb_split`` would then rotate it twice and lose the ``dH/dk`` eigenbasis (worst k of the
    d=1 dielectric on doubled example01). Overlaps are merged into one range per cluster.
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
    """Sparse ``(n, n)`` operators handed out as dense arrays, so the shared per-k kernels
    (``perturb_split``, ``momentum_k``, ...) reach threaded BLAS (see
    :func:`~PAOFLOW.sparse.mesh._times_block`).

    ``cache=False`` densifies on every access, keeping at most one alive: for operators read once,
    like the six ``d2H/dk_i dk_j``.
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
    ik, ispin : int
        Local k index on this rank, spin channel.
    kvec : np.ndarray, shape (3,)
        Crystal (``cart=False``) or Cartesian in ``2 pi / alat`` (``cart=True``).
    E : np.ndarray, shape (m,)
        Every solved eigenvalue (eV), ascending; ``m = nawf`` with ``full_spectrum``.
    V : np.ndarray, shape (nawf, m)
        Orthonormal eigenvectors.
    hk : scipy.sparse.csr_matrix, shape (nawf, nawf)
    dhk : list of scipy.sparse.csr_matrix, or DenseOnDemand
        ``[dH/dk_x, dH/dk_y, dH/dk_z]``.
    vel : np.ndarray, shape (3, bnd)
    delta : np.ndarray, shape (bnd,)
        Velocities and adaptive widths of the window bands.
    bnd : int
        ``E[:bnd]`` are the bands the stored mesh arrays keep.
    afac, dk : float
        Adaptive-width prefactor and mean k spacing.
    min_width, max_width : float or None, optional
        Bounds of the widths ``delta_all`` and ``delta2`` form (``adaptive_smearing``).
    sign : {-1, +1}, optional
        Fourier phase convention (mesh ``-1``, path ``+1``).
    cart : bool, optional
        Whether ``kvec`` is Cartesian.
    kglobal : int or None, optional
        Index in the full mesh (FFT-grid order) or path, to address per-k DFT data.
    with_dnm : bool, optional
        Whether ``dhk``/:attr:`d2hk` carry the intra-cell ``Dnm`` terms
        (``SparseHamiltonian.assemble_derivatives``).
    dense_operators : bool, optional
        Hand out ``dhk``/:attr:`d2hk` as :class:`DenseOnDemand`.
    resolved : sequence of (D, parts), optional
        Degenerate groups whose ``V`` columns the mesh pass already rotated into a joint
        ``dH/dk_l`` eigenbasis, except on ``parts`` (``mesh._band_diagonal_velocities``).

    Notes
    -----
    Members follow the dense conventions, one k at a time, so dense per-k kernel bodies apply
    unchanged: ``degen`` ~ ``degen[ispin][ik]``, ``pksp`` ~
    :func:`~PAOFLOW.hamiltonian.do_momentum.momentum_k`, ``delta2`` ~ ``deltakp2[ik]``.
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
        min_width: float | None = None,
        max_width: float | None = None,
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
        self.width_bounds = {'min_width': min_width, 'max_width': max_width}

    @property
    def nstates(self) -> int:
        """Number of states solved at this k-point (``len(E)``)."""
        return len(self.E)

    @cached_property
    def degen(self) -> list[np.ndarray]:
        """Degenerate groups among the window bands (dense ``degen[ispin][ik]``).

        Groups in :attr:`resolved` are replaced by their ``parts``: ``V`` already holds a joint
        ``dH/dk_l`` eigenbasis there, and ``perturb_split`` re-rotating per direction would pair
        one state's matrix elements with another's velocities.
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
        """Adaptive widths ``(m,)`` of every solved state, from the ``pksp`` diagonal
        (dense ``deltakp[ik]``); ``delta`` covers only the window."""
        from ..spectrum.do_adaptive_smearing import adaptive_widths

        diagonal = np.ascontiguousarray(np.einsum('lnn->ln', self.pksp))
        return adaptive_widths(diagonal, self.afac, self.dk, **self.width_bounds)

    @cached_property
    def delta2(self) -> np.ndarray:
        """Interband adaptive widths ``(m, m)`` (dense ``deltakp2[ik]``)."""
        from ..spectrum.do_adaptive_smearing import adaptive_widths

        diagonal = np.ascontiguousarray(np.einsum('lnn->ln', self.pksp))
        return adaptive_widths(diagonal, self.afac, self.dk, pairs=True, **self.width_bounds)[1]
