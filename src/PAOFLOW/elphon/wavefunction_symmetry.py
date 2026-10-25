"""Full-grid Bloch functions from an nscf on the irreducible wedge.

An nscf with ``K_POINTS automatic`` stores the wavefunctions of the irreducible k-points
only.  The native electron-phonon vertex (:mod:`PAOFLOW.elphon.dfpt_vertex`) and the PAO
projections need every k of the full Γ-centred grid, which this module reconstructs by
symmetry: for the space-group operation ``g: x -> R x - t`` (crystal coordinates of
positions, :class:`~PAOFLOW.elphon.dfpt_io.SpaceGroup`) the rotated state
``(O_g psi_k)(r) = psi_k(g^-1 r)`` is the Bloch function at ``R k`` with coefficients

.. math::

    c_{Rk}(R G) = e^{2\\pi i (\\kappa' + m')\\cdot t}\\, c_k(G),

``kappa'`` and ``m'`` being the crystal coordinates of ``R k`` and ``R G``.  Spinors are
rotated with the SU(2) matrix of the proper part of ``R``; time reversal is complex
conjugation (``k -> -k``, ``G -> -G``), combined with ``i sigma_y`` for spinors.  The
rotated states are the same states as an nscf at ``R k`` would give, up to a unitary
mixing inside degenerate groups, so everything that treats the bands as bra and ket of the
same block (the vertex, the projections computed from the rotated block) is unchanged.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .dfpt_io import SpaceGroup, read_nscf_plane_wave_setup, read_space_group
from .plane_waves import PlaneWaveBlock, read_plane_wave_block


def _rotation_helpers():
    from .dfpt_vertex import cartesian_rotation, rotate_q

    return cartesian_rotation, rotate_q


_PAULI = (
    np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
)


def su2_rotation(m_cart: NDArray[np.float64]) -> NDArray[np.complex128]:
    """SU(2) matrix of a Cartesian rotation (QE ``find_u``: improper operations use ``-R``).

    Parameters
    ----------
    m_cart : ndarray ``(3, 3)``
        Orthogonal matrix.

    Returns
    -------
    ndarray ``(2, 2)``, complex
        ``cos(theta/2) - i sin(theta/2) n.sigma`` for the proper rotation by ``theta`` about
        ``n`` (right-handed); the overall sign is a global phase.
    """
    r = np.asarray(m_cart, dtype=float)
    if np.linalg.det(r) < 0:
        r = -r
    cos_t = np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.eye(2, dtype=complex)
    axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if np.linalg.norm(axis) > 1e-8:
        n = axis / np.linalg.norm(axis)
    else:  # theta = pi: (R + 1)/2 = n n^T
        sym = 0.5 * (r + np.eye(3))
        i = int(np.argmax(np.diag(sym)))
        n = sym[:, i] / np.sqrt(sym[i, i])
    return np.cos(theta / 2) * np.eye(2, dtype=complex) - 1j * np.sin(theta / 2) * (
        n[0] * _PAULI[0] + n[1] * _PAULI[1] + n[2] * _PAULI[2]
    )


@dataclass(frozen=True)
class FullGridMap:
    """How every point of the full grid derives from the irreducible k-points.

    Attributes
    ----------
    grid : tuple of int
    kpts_cryst : ndarray ``(N, 3)``
        Full grid in ``[0, 1)``, label order (third index fastest).
    source : ndarray ``(N,)`` int
        Index of the irreducible point.
    isym : ndarray ``(N,)`` int
        Operation (0 = identity for the representatives themselves).
    time_reversed : ndarray ``(N,)`` bool
    umklapp : ndarray ``(N, 3)`` int
        ``kpts_cryst = (+/-) R k_irr + umklapp``.
    """

    grid: tuple[int, int, int]
    kpts_cryst: NDArray[np.float64]
    source: NDArray[np.int_]
    isym: NDArray[np.int_]
    time_reversed: NDArray[np.bool_]
    umklapp: NDArray[np.int_]

    @property
    def nk(self) -> int:
        """Number of full-grid points."""
        return int(self.kpts_cryst.shape[0])


def full_grid_map(
    kpts_irr_cryst: NDArray[np.float64],
    group: SpaceGroup,
    grid: Sequence[int],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
    time_reversal: bool = True,
) -> FullGridMap:
    """Assign every full-grid point to an irreducible point and an operation.

    Parameters
    ----------
    kpts_irr_cryst : ndarray ``(nk_irr, 3)``
    group : SpaceGroup
    grid : sequence of int
        ``(nk1, nk2, nk3)``, Γ-centred.
    at, bg : ndarray ``(3, 3)``
    time_reversal : bool, optional
        Also use ``k -> -k``.

    Raises
    ------
    ValueError
        If the stars of the irreducible points do not cover the grid (wrong grid,
        symmetries or time-reversal flag).
    """
    _, rotate_q = _rotation_helpers()
    ng = np.asarray(grid, dtype=int)
    nk = int(np.prod(ng))
    labels = np.stack(np.meshgrid(*[np.arange(n) for n in ng], indexing='ij'), axis=-1).reshape(
        -1, 3
    )
    assigned: dict[tuple[int, int, int], tuple[int, int, bool, NDArray[np.int_]]] = {}
    for i, k_irr in enumerate(np.asarray(kpts_irr_cryst, dtype=float)):
        for trev in [False, True] if time_reversal else [False]:
            for isym in range(group.nsym):
                sk = rotate_q(group.rotations[isym], k_irr, at, bg)
                if trev:
                    sk = -sk
                lab = np.rint(sk * ng).astype(int)
                if np.abs(lab - sk * ng).max() > 1e-6:
                    raise ValueError('R k = %s is not on the %s grid' % (sk, tuple(ng)))
                key = tuple(int(x) for x in lab % ng)
                if key not in assigned:
                    g0 = np.array(key) - lab  # a multiple of ng by construction
                    assigned[key] = (i, isym, trev, (g0 // ng).astype(int))
    if len(assigned) != nk:
        raise ValueError(
            'the %d irreducible k-points cover %d of the %d grid points (grid, symmetries or '
            'time reversal do not match the nscf)' % (len(kpts_irr_cryst), len(assigned), nk)
        )
    src = np.empty(nk, dtype=int)
    isyms = np.empty(nk, dtype=int)
    trevs = np.empty(nk, dtype=bool)
    umk = np.empty((nk, 3), dtype=int)
    for n, lab in enumerate(labels):
        i, isym, trev, g0 = assigned[tuple(int(x) for x in lab)]
        src[n], isyms[n], trevs[n], umk[n] = i, isym, trev, g0
    return FullGridMap(tuple(int(n) for n in ng), labels / ng, src, isyms, trevs, umk)


def rotate_block(
    block: PlaneWaveBlock,
    group: SpaceGroup,
    isym: int,
    time_reversed: bool,
    target_cryst: NDArray[np.float64],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
) -> PlaneWaveBlock:
    """The Bloch functions at the full-grid point ``target_cryst`` from an irreducible block.

    Parameters
    ----------
    block : PlaneWaveBlock
        Stored block at the irreducible ``k``.
    group : SpaceGroup
    isym : int
    time_reversed : bool
    target_cryst : ndarray ``(3,)``
        The grid representative (``[0, 1)``); must equal ``(+/-) R k`` up to a reciprocal
        lattice vector, which is applied to the Miller indices (the stored ``k`` of the block
        may carry a different lattice shift than the folded k-point list).
    at, bg : ndarray ``(3, 3)``

    Returns
    -------
    PlaneWaveBlock
        At ``target_cryst``, with the rotated Miller indices and phased coefficients.
    """
    cartesian_rotation, rotate_q = _rotation_helpers()
    rot = group.rotations[isym]
    t = group.translations[isym]
    k_cryst = np.linalg.solve(np.asarray(bg, dtype=float).T, block.xk_cart)
    k_rot = rotate_q(rot, k_cryst, at, bg)
    w = np.array([rotate_q(rot, e, at, bg) for e in np.eye(3)]).T  # W e_i as columns
    m_rot = np.rint(w @ block.miller.astype(float)).astype(int)  # (3, npw)
    if np.abs(w @ block.miller - m_rot).max() > 1e-6:
        raise ValueError('rotated Miller indices are not integers (symmetry/lattice mismatch)')
    phase = np.exp(2j * np.pi * ((k_rot[:, None] + m_rot) * t[:, None]).sum(0))  # (npw,)
    npw = block.npw
    coef = block.coefficients
    if block.npol == 2:
        u = su2_rotation(cartesian_rotation(rot, at))
        up, dn = coef[:, :npw], coef[:, npw:]
        coef = np.concatenate([u[0, 0] * up + u[0, 1] * dn, u[1, 0] * up + u[1, 1] * dn], axis=1)
    coef = coef * np.tile(phase, block.npol)[None, :]
    if time_reversed:
        k_rot = -k_rot
        m_rot = -m_rot
        coef = np.conj(coef)
        if block.npol == 2:
            coef = np.concatenate([coef[:, npw:], -coef[:, :npw]], axis=1)  # i sigma_y K
    target = np.asarray(target_cryst, dtype=float)
    shift = np.rint(target - k_rot).astype(int)
    if np.abs(k_rot + shift - target).max() > 1e-6:
        raise ValueError('inconsistent map: (+/-) R k = %s, target %s' % (k_rot, target))
    miller = m_rot - shift[:, None]
    return PlaneWaveBlock(
        target @ np.asarray(bg, dtype=float),
        np.ascontiguousarray(miller),
        np.ascontiguousarray(coef),
        block.npol,
    )


def stores_irreducible_wedge(save_dir: str) -> bool:
    """True when a QE save holds the irreducible wedge of a Gamma-centred automatic grid.

    Parameters
    ----------
    save_dir : str
        QE save directory (``data-file-schema.xml``).

    Returns
    -------
    bool
        A ``monkhorst_pack`` element without shift and fewer stored k-points than
        ``nk1 * nk2 * nk3``; ``False`` for explicit k lists and complete grids.  The
        electron-phonon drivers pass this to ``unfold_kpoints``.
    """
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    mp = root.find('output/band_structure/starting_k_points/monkhorst_pack')
    if mp is None:
        return False
    a = mp.attrib
    if any(int(a[s]) for s in ('k1', 'k2', 'k3')):
        return False
    nks = int(root.findtext('output/band_structure/nks'))
    return nks < int(a['nk1']) * int(a['nk2']) * int(a['nk3'])


def monkhorst_pack_grid(save_dir: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """``((nk1, nk2, nk3), (k1, k2, k3))`` of an automatic nscf from ``data-file-schema.xml``.

    Raises
    ------
    ValueError
        For an explicit k-point list (no ``monkhorst_pack`` element).
    """
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    mp = root.find('output/band_structure/starting_k_points/monkhorst_pack')
    if mp is None:
        raise ValueError(
            '%s: the nscf used an explicit k-point list; the irreducible-wedge unfolding needs K_POINTS automatic'
            % save_dir
        )
    a = mp.attrib
    return (int(a['nk1']), int(a['nk2']), int(a['nk3'])), (int(a['k1']), int(a['k2']), int(a['k3']))


def _time_reversal_flag(save_dir: str) -> bool:
    root = ET.parse(os.path.join(save_dir, 'data-file-schema.xml')).getroot()
    noinv = root.find('input/symmetry_flags/noinv')
    if noinv is not None and (noinv.text or '').strip().lower() == 'true':
        return False
    mag = root.find('output/magnetization')
    if mag is not None:
        noncolin = (mag.findtext('noncolin') or 'false').strip().lower() == 'true'
        domag = (mag.findtext('do_magnetization') or 'false').strip().lower() == 'true'
        if noncolin and domag:
            return False
    return True


class WavefunctionProvider:
    """Full-grid plane-wave blocks and k-points from an irreducible-wedge nscf save.

    Parameters
    ----------
    save_dir : str
        QE save of an nscf with ``K_POINTS automatic`` (Γ-centred).
    nspin : int, optional
        1 (or 4 for noncollinear) reads ``wfc<ik>``; 2 reads ``wfcup``/``wfcdw``.
    cache_blocks : int, optional
        Number of rotated blocks kept in memory (the irreducible blocks are always kept).
    """

    def __init__(self, save_dir: str, nspin: int = 1, cache_blocks: int = 256):
        self.save_dir = save_dir
        self.nspin = int(nspin)
        self.setup = read_nscf_plane_wave_setup(save_dir)
        self.group = read_space_group(save_dir)
        self.at, self.bg, self.alat = self.setup['at'], self.setup['bg'], self.setup['alat']
        grid, shift = monkhorst_pack_grid(save_dir)
        if any(shift):
            raise ValueError(
                'the irreducible-wedge unfolding needs a Gamma-centred grid (k1 = k2 = k3 = 0)'
            )
        from .elph_bloch import read_nscf

        self.kpts_irr_cryst = read_nscf(save_dir)['kpts_cryst']
        self.time_reversal = _time_reversal_flag(save_dir)
        self.map = full_grid_map(
            self.kpts_irr_cryst, self.group, grid, self.at, self.bg, self.time_reversal
        )
        self.grid = self.map.grid
        self.kpts_cryst = self.map.kpts_cryst
        self.nk = self.map.nk
        self._irr: dict[tuple[int, int], PlaneWaveBlock] = {}
        self._cache: OrderedDict[tuple[int, int], PlaneWaveBlock] = OrderedDict()
        self.cache_blocks = int(cache_blocks)

    def irreducible_block(self, i: int, ispin: int = 0) -> PlaneWaveBlock:
        """Stored block of irreducible point ``i``."""
        key = (int(i), int(ispin))
        if key not in self._irr:
            self._irr[key] = read_plane_wave_block(
                self.save_dir, int(i), self.alat, self.nspin, ispin
            )
        return self._irr[key]

    def block(self, ik: int, ispin: int = 0) -> PlaneWaveBlock:
        """Block at full-grid point ``ik`` (label order), rotated from its irreducible source."""
        key = (int(ik), int(ispin))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        m = self.map
        blk = rotate_block(
            self.irreducible_block(m.source[ik], ispin),
            self.group,
            int(m.isym[ik]),
            bool(m.time_reversed[ik]),
            m.kpts_cryst[ik],
            self.at,
            self.bg,
        )
        self._cache[key] = blk
        if len(self._cache) > self.cache_blocks:
            self._cache.popitem(last=False)
        return blk

    def expand(self, values: NDArray, axis: int) -> NDArray:
        """Expand an array indexed by irreducible k along ``axis`` to the full grid."""
        return np.take(np.asarray(values), self.map.source, axis=axis)

    def expand_weights(self, weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Uniform full-grid weights with the total of the irreducible ones."""
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != self.kpts_irr_cryst.shape[0]:
            raise ValueError(
                'expected %d weights, got %d' % (self.kpts_irr_cryst.shape[0], w.shape[0])
            )
        return np.full(self.nk, float(w.sum()) / self.nk)

    def kpts_cart(self) -> NDArray[np.float64]:
        """Full-grid k-points, Cartesian ``2 pi / alat``."""
        return self.kpts_cryst @ self.bg

    def gkspace_wfc(self, ik: int, ispin: int = 0) -> tuple[dict, dict]:
        """The ``(gkspace, wfc)`` pair of :func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc` (raw coefficients).

        ``xk`` and ``bg`` are in bohr^-1 as in the ``wfc*.dat`` headers (``bg`` columns).
        """
        blk = self.block(ik, ispin)
        tpiba = 2.0 * np.pi / self.alat
        gkspace = {
            'xk': blk.xk_cart * tpiba,
            'igwx': blk.npw,
            'mill': blk.miller,
            'bg': (np.asarray(self.bg, dtype=float) * tpiba).T,
            'gamma_only': False,
        }
        return gkspace, {
            'wfc': blk.coefficients,
            'npol': blk.npol,
            'nbnd': blk.nbnd,
            'ispin': ispin,
        }


_PROVIDERS: dict[tuple[str, int], WavefunctionProvider] = {}


def provider_for(save_dir: str, nspin: int = 1) -> WavefunctionProvider:
    """Per-process cache of :class:`WavefunctionProvider` keyed by save directory."""
    key = (os.path.abspath(save_dir), int(nspin))
    if key not in _PROVIDERS:
        _PROVIDERS[key] = WavefunctionProvider(save_dir, nspin)
    return _PROVIDERS[key]


def irreducible_representatives(
    kpts_full_cryst: NDArray[np.float64],
    group: SpaceGroup,
    grid: Sequence[int],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
    time_reversal: bool = True,
) -> NDArray[np.int_]:
    """Indices of a full-grid list that QE's ``kpoint_grid`` would keep (first of each star in grid order).

    Used to emulate an irreducible-wedge save from a full-grid one in tests.
    """
    _, rotate_q = _rotation_helpers()
    ng = np.asarray(grid, dtype=int)
    kint = np.rint(np.asarray(kpts_full_cryst) * ng).astype(int) % ng
    order = np.lexsort((kint[:, 2], kint[:, 1], kint[:, 0]))  # grid order, third index fastest
    seen: set[tuple[int, int, int]] = set()
    reps = []
    for ik in order:
        key = tuple(int(x) for x in kint[ik])
        if key in seen:
            continue
        reps.append(int(ik))
        k = kpts_full_cryst[ik]
        for trev in [False, True] if time_reversal else [False]:
            for isym in range(group.nsym):
                sk = rotate_q(group.rotations[isym], k, at, bg)
                if trev:
                    sk = -sk
                seen.add(tuple(int(x) for x in np.rint(sk * ng).astype(int) % ng))
    return np.array(reps, dtype=int)


def expand_data_controller_kpoints(data_controller) -> WavefunctionProvider:
    """Replace the irreducible k-point data of a :class:`~PAOFLOW.DataController.DataController` by the full grid.

    Called on the reading rank right after the QE XML has been parsed (``unfold_kpoints=True`` in
    :class:`PAOFLOW.PAOFLOW`): ``kpnts``, ``kpnts_wght``, ``my_eigsmat`` and ``nkpnts`` become those of
    the full Γ-centred ``nk1 x nk2 x nk3`` grid in ``K_POINTS automatic`` order, so that the
    projections, the Hamiltonian (``expand_wedge=False``) and the electron-phonon vertex see an
    ordinary full-grid nscf.  The wavefunctions themselves are rotated on demand by
    :func:`PAOFLOW.projection.do_atwfc_proj.read_QE_wfc`.

    Returns
    -------
    WavefunctionProvider
        The provider used for the map (also cached per process, see :func:`provider_for`).

    Raises
    ------
    ValueError
        If the save already holds the full grid, or if it is not a Γ-centred automatic grid.
    """
    arrays, attr = data_controller.data_dicts()
    provider = provider_for(attr['fpath'], attr['nspin'])
    nk_irr = int(attr['nkpnts'])
    if nk_irr == provider.nk:
        raise ValueError(
            '%s already holds the full %s grid; unfold_kpoints is for an nscf on the irreducible wedge'
            % (attr['fpath'], provider.grid)
        )
    if nk_irr != provider.kpts_irr_cryst.shape[0]:
        raise ValueError(
            'k-point count mismatch between the XML (%d) and the provider (%d)'
            % (nk_irr, provider.kpts_irr_cryst.shape[0])
        )
    arrays['kpnts'] = provider.kpts_cart()
    arrays['kpnts_wght'] = provider.expand_weights(arrays['kpnts_wght'])
    arrays['my_eigsmat'] = np.ascontiguousarray(provider.expand(arrays['my_eigsmat'], axis=1))
    attr['nkpnts'] = provider.nk
    attr['nk1'], attr['nk2'], attr['nk3'] = provider.grid
    attr['ok1'] = attr['ok2'] = attr['ok3'] = 0
    attr['unfolded_from_irreducible'] = nk_irr
    return provider
