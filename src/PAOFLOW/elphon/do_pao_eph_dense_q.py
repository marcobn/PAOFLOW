"""SKETCH -- double real-space PAO el-ph vertex g(R_e, R_p) and dense-q Eliashberg.

Prototype extension of the PAO route (:mod:`PAOFLOW.elphon.do_pao_eph`) that
Wigner-Seitz interpolates the phonons *and* the electron-phonon vertex to a dense
q-grid, instead of summing only the coarse DFPT q-points.  The electron side is
unchanged -- it reuses :func:`~PAOFLOW.elphon.elph_bloch.precompute_dense_electrons`
and :func:`~PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws_fast` verbatim.

Idea (EPW / Wannier-Fourier, but in the deterministic PAO gauge)
----------------------------------------------------------------
The current route builds, for one coarse q, the half-transformed vertex

    g_q(R_e)_{ij,c}          # electron in R_e, phonon still Bloch-q, Cartesian c

with :func:`~PAOFLOW.elphon.do_pao_eph.vertex_from_qe_elphmat`.  Collecting this
for *every* q on the full coarse q-grid and Fourier-transforming q -> R_p gives
the double real-space object

    g(R_e, R_p)_{ij,c} = (1/N_q) sum_q e^{-2 pi i q . R_p} g_q(R_e)_{ij,c} .

For any dense q we recover the half-transformed vertex by a Wigner-Seitz sum over
the phonon cells R_p,

    g_q(R_e)_{ij,c} = sum_{R_p} W_p e^{+2 pi i q . R_p} g(R_e, R_p)_{ij,c} ,

which is exactly the input :func:`lambda_q_dense_ws_fast` already consumes, so the
dense-k Fermi-surface double delta is untouched.  The phonon frequencies /
eigenvectors at the same dense q come from the standard q2r/matdyn Wigner-Seitz
interpolation of the dynamical matrix (:mod:`PAOFLOW.elphon.qe_matdyn`).

Working in the **Cartesian displacement** basis (index ``c = 3*kappa + alpha``)
is deliberate: it is smooth in q (the per-q phonon eigenvectors z(q) are applied
later at the dense q), so it is the right representation to Fourier-interpolate.

Deferred / to validate
-----------------------
* Symmetry unfolding: g(R_e, R_p) needs g_q(R_e) on the *full* coarse q-grid.
  Either dump all q from ph.x, or unfold the irreducible set by the star
  operations (rotate R_e, the orbital pair and the Cartesian/atom index).  The
  prototype assumes the full grid is supplied (``TODO: unfold_star``).
* q-phase convention: the e^{i q . tau_kappa} atom-position phase of QE's
  ``el_ph_mat`` must match the convention used by :mod:`qe_matdyn` when it builds
  D(q)/z(q); otherwise the q -> R_p transform mixes cells (``TODO: verify_phase``).
* Eigenvector normalisation: ``_phonon_modes_at_q`` must return z in the SAME
  convention as :func:`~PAOFLOW.elphon.qe_elph_io.read_qe_dyn` so that
  ``zmass = z / sqrt(M)`` matches the coarse-q driver (``TODO: verify_evec``).
* Polar (Frohlich) long-range part is intentionally NOT handled here; add the
  dipole/quadrupole subtract-before / add-back-after step for polar materials.
  For now :func:`~PAOFLOW.elphon.do_pao_eph.check_long_range_terms` refuses
  polar EPW couplings (and warns for non-polar insulators); the plan is the wiki
  page *Long-range electron-phonon interpolation (plan)*.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from ..phonon.qe_force_constants import (
    PHONON_WS_SIGN,  # noqa: F401  (re-exported for backward compatibility)
    phonon_interpolator,
    read_dynamical_matrix_file,
    real_space_force_constants,
)
from .do_pao_eph import (
    check_long_range_terms,
    load_epw_coupling,
    vertex_from_epw,
    vertex_from_qe_ahc,
    vertex_from_qe_elphmat,
)
from .elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    _ws_lattice,
    _ws_lattice_pairs,
    fermi_surface_plasma_frequency,
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
    release_dense_electrons,
)
from .eph_kq import eliashberg_from_modes
from .fermi_surface_coupling import FermiSurfacePairCoupling


# --------------------------------------------------------------------------- #
# 1. Build the double real-space vertex g(R_e, R_p)
# --------------------------------------------------------------------------- #
def build_g_ReRp(g_qRe, q_cryst, qgrid):
    """Fourier-transform the coarse-q half-vertex ``g_q(R_e)`` to ``g(R_e, R_p)``.

    Parameters
    ----------
    g_qRe : ndarray ``(nq, nawf, nawf, ncart, n1e, n2e, n3e)``
        The per-q PAO-gauge vertex ``g_q(R_e)`` (Cartesian displacement basis),
        one slice per coarse q-point -- e.g. stacked outputs of
        :func:`~PAOFLOW.elphon.do_pao_eph.vertex_from_qe_elphmat`.
    q_cryst : ndarray ``(nq, 3)``
        The coarse q-points in crystal coordinates (must tile the full ``qgrid``).
    qgrid : tuple(int, int, int)
        The coarse phonon q-grid ``(nq1, nq2, nq3)`` (``nq == prod(qgrid)``).

    Returns
    -------
    g_ReRp : ndarray ``(nawf, nawf, ncart, n1e, n2e, n3e, nq1, nq2, nq3)``
        The double real-space vertex; the trailing three axes are the phonon
        cells ``R_p`` (grid == ``qgrid``).
    """
    qgrid = tuple(int(n) for n in qgrid)
    nq = g_qRe.shape[0]
    if nq != qgrid[0] * qgrid[1] * qgrid[2]:
        raise ValueError(
            'build_g_ReRp needs the FULL coarse q-grid (%d), got %d q-points; '
            'unfold the irreducible set first (TODO: unfold_star).'
            % (qgrid[0] * qgrid[1] * qgrid[2], nq)
        )

    # Scatter each q onto its integer grid cell, then FFT q -> R_p (same 1/N and
    # sign convention as vertex_pao_R uses for k -> R_e).
    tail = g_qRe.shape[1:]  # (nawf, nawf, ncart, n1e, n2e, n3e)
    qlab = np.round(np.asarray(q_cryst) * np.asarray(qgrid)).astype(int) % np.asarray(qgrid)
    gq_grid = np.zeros(qgrid + tail, dtype=complex)
    gq_grid[qlab[:, 0], qlab[:, 1], qlab[:, 2]] = g_qRe
    # FFT over the three q-axes (axes 0,1,2) -> R_p, move them to the tail.
    g_ReRp = np.fft.fftn(gq_grid, axes=(0, 1, 2)) / nq
    g_ReRp = np.moveaxis(g_ReRp, (0, 1, 2), (-3, -2, -1))
    return np.ascontiguousarray(g_ReRp)


# --------------------------------------------------------------------------- #
# 2. Evaluate the half-vertex g_q(R_e) at an arbitrary dense q
# --------------------------------------------------------------------------- #
def g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p):
    """Wigner-Seitz sum over ``R_p`` -> the half-vertex ``g_q(R_e)`` at one q.

    ``(Nint_p, W_p, Midx_p)`` is :func:`~PAOFLOW.elphon.elph_bloch._ws_lattice`
    applied to the phonon q-grid: ``Nint_p`` are the integer phonon cells (Bloch
    phase ``exp(2 pi i q . n_p)``), ``W_p`` the WS degeneracy weights and
    ``Midx_p = n_p mod qgrid`` the indices into the trailing axes of ``g_ReRp``.
    ``W_p`` may also be pair-resolved, shape ``(nws_p, nawf, nawf, ncart)``
    (:func:`vertex_phonon_ws_weights`).

    Returns
    -------
    gR : ndarray ``(nawf, nawf, ncart, n1e, n2e, n3e)``
        The half-vertex for this q -- the exact input shape expected by
        :func:`~PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws_fast`.
    """
    if np.ndim(W_p) == 1:
        phase = W_p * np.exp(2j * np.pi * (np.asarray(q_cryst) @ Nint_p.T))  # (nws_p,)
        cells = g_ReRp[..., Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]]  # (..., nws_p)
        return np.tensordot(cells, phase, axes=([-1], [0]))  # (nawf, nawf, ncart, n1e,n2e,n3e)
    # Pair-resolved weights (nws_p, nawf, nawf, ncart).  The Wigner-Seitz images of
    # one grid cell share its data, so their phases and weights are summed first:
    # one contiguous pass over g(R_e, R_p) instead of one strided pass per image.
    phase = np.exp(2j * np.pi * (np.asarray(q_cryst) @ Nint_p.T))
    grid_p = g_ReRp.shape[6:]
    image_cell = np.ravel_multi_index(np.asarray(Midx_p).T, grid_p)
    cell_weights = np.zeros((int(np.prod(grid_p)),) + W_p.shape[1:], dtype=complex)
    np.add.at(cell_weights, image_cell, phase[:, None, None, None] * W_p)
    cell_weights = np.moveaxis(cell_weights, 0, -1)  # (nawf, nawf, ncart, nRp)
    rows = g_ReRp.reshape(g_ReRp.shape[:3] + (-1, cell_weights.shape[-1]))
    vertex = np.empty(g_ReRp.shape[:6], dtype=complex)
    for orbital in range(g_ReRp.shape[0]):
        vertex[orbital] = np.einsum('jcrp,jcp->jcr', rows[orbital], cell_weights[orbital]).reshape(
            g_ReRp.shape[1:6]
        )
    return vertex


def vertex_phonon_ws_weights(
    qgrid: Sequence[int],
    at: NDArray[np.float64],
    orbital_positions: NDArray[np.float64],
    atom_positions: NDArray[np.float64],
) -> tuple[NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]]:
    """Pair-resolved Wigner-Seitz images of the phonon cells ``R_p`` of the vertex.

    Parameters
    ----------
    qgrid : sequence of int
        Coarse phonon q-grid.
    at : ndarray, shape ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    orbital_positions : ndarray, shape ``(nawf, 3)``
        PAO orbital centres (crystal coordinates).
    atom_positions : ndarray, shape ``(nat, 3)``
        Atomic positions (crystal coordinates).

    Returns
    -------
    Nint_p : ndarray, shape ``(nws_p, 3)``, int
        Phonon cells.
    W_p : ndarray, shape ``(nws_p, nawf, nawf, 3 nat)``
        Weight of each cell for each vertex element ``(i, j, c)``, input of
        :func:`g_Re_at_q`.
    Midx_p : ndarray, shape ``(nws_p, 3)``, int
        Grid indices ``R_p mod qgrid``.

    Notes
    -----
    The cell ``R_p`` of ``g_{ij, c}`` is kept when the displaced atom
    ``kappa(c)`` at ``R_p`` lies in the Wigner-Seitz cell of the supercell
    around orbital ``i``, as EPW does for ``g(R_e, R_p)`` (electron Wannier
    centre to atom).  Of the four choices of orbital (``i`` or ``j``) and sign,
    this one makes ``lambda_q`` of MgB2 the most symmetric within a q-star
    (spread below 1%, against up to 33% with single-site images).
    """
    mode_positions = np.repeat(np.asarray(atom_positions, dtype=float), 3, axis=0)
    Nint_p, weights, Midx_p = _ws_lattice_pairs(
        qgrid, at, orbital_positions, sign=1, column_positions_cryst=mode_positions
    )  # (nws_p, nawf, ncart): orbital i, displaced-atom coordinate c
    nawf, ncart = weights.shape[1], weights.shape[2]
    weights = np.broadcast_to(weights[:, :, None, :], (len(Nint_p), nawf, nawf, ncart))
    return Nint_p, weights, Midx_p


# --------------------------------------------------------------------------- #
# 3. Phonon modes at a dense q (frequencies + Cartesian eigenvectors)
# --------------------------------------------------------------------------- #
def phonon_interp_from_dyn(dyn_paths, qgrid, bg, at, atom_positions=None, long_range=None):
    """Dense-q phonon interpolator built from the coarse ``*.dyn`` files.

    Reconstructs the mass-weighted dynamical matrix ``D(q)`` on the full coarse
    q-grid from each dyn file's ``(freq, eigenvector)`` spectral pair
    (``D = sum_nu omega_nu^2 e_nu e_nu^dagger``, exact since the QE eigenvectors
    are mass-weighted and orthonormal), Fourier-transforms ``q -> R_p`` and
    returns ``phonon_at_q(q_cryst) -> (freq_thz, z)`` that Wigner-Seitz
    interpolates ``D`` to any q and re-diagonalises it.  ``z`` are the
    mass-weighted Cartesian eigenvectors ``(nmode, ncart)`` -- the same
    convention as :func:`~PAOFLOW.elphon.qe_elph_io.read_qe_dyn`, so the driver's
    ``zmass = z / sqrt(M)`` matches the coarse-q route.

    This avoids ``q2r.x`` (whose star bookkeeping is incompatible with the
    EPW/AHC full-grid dyn dumps) and mirrors the vertex ``q -> R_p`` transform.

    Parameters
    ----------
    dyn_paths : sequence of str
        ``*.dyn`` files whose q-points, including every star member listed in
        each file, cover the full coarse grid: either one file per q of a
        full-grid (``nosym``) run, or the irreducible-q files of a symmetric
        ``ph.x`` run.
    qgrid : tuple(int, int, int)
        Coarse phonon q-grid.
    bg, at : ndarray ``(3, 3)``
        Reciprocal- and real-lattice vectors (rows).
    atom_positions : ndarray ``(nat, 3)``, optional
        Atomic positions (crystal coordinates) for atom-pair Wigner-Seitz images
        (as ``matdyn.x``; recommended with several atoms per cell).  ``None``
        keeps the single-site images.
    long_range : LongRangeModel, optional
        Polar crystals: the dipole (and quadrupole) force constants
        (:meth:`~PAOFLOW.elphon.long_range.LongRangeModel.dynamical_matrix`) are subtracted
        from the coarse ``C(q)`` before the ``q -> R_p`` transform and added back at every
        dense q (LO-TO splitting); the returned callable records it as ``phonon_at_q.long_range``.

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz (nmode,), z (nmode, ncart))``.

    Raises
    ------
    ValueError
        If the files do not cover every q of the grid.
    """
    qgrid = tuple(int(n) for n in qgrid)
    nq = qgrid[0] * qgrid[1] * qgrid[2]

    # Collect the full-precision force-constant matrix C(q) for every star-q in
    # every dyn file (the "Dynamical Matrix in cartesian axes" blocks are the
    # force constants, i.e. eig(C)/M = omega^2), and place them on the full grid.
    Cgrid = None
    masses_ry = None
    for path in dyn_paths:
        parsed = _read_dyn_matrices(path)
        masses_ry = parsed['masses_ry']
        nmode = 3 * masses_ry.size
        if Cgrid is None:
            Cgrid = np.zeros(qgrid + (nmode, nmode), dtype=complex)
            filled = np.zeros(qgrid, dtype=bool)
        for q_cart, C in zip(parsed['q_cart'], parsed['C']):
            qc = np.linalg.solve(bg.T, np.asarray(q_cart, dtype=float))
            lab = tuple(np.round(qc * np.asarray(qgrid)).astype(int) % np.asarray(qgrid))
            if not filled[lab]:
                Cgrid[lab] = C
                filled[lab] = True
    if not filled.all():
        raise ValueError('dyn files cover only %d/%d q of the grid.' % (filled.sum(), nq))
    return _phonon_interp_from_force_constants(
        Cgrid, masses_ry, qgrid, at, atom_positions, long_range=long_range
    )


def phonon_interp_from_epw(
    dynq: NDArray[np.complex128],
    q_cryst: NDArray[np.float64],
    masses_amu: NDArray[np.float64],
    qgrid: tuple[int, int, int],
    at: NDArray[np.float64],
    atom_positions: NDArray[np.float64] | None = None,
    long_range=None,
) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.complex128]]]:
    """Dense-q phonon interpolator from EPW's coarse force constants.

    Parameters
    ----------
    dynq : NDArray[np.complex128], shape ``(3 nat, 3 nat, nq)``
        Force-constant matrices on the full coarse q-grid
        (:func:`~PAOFLOW.elphon.qe_elph_io.read_epw_epb` ``['dynq']``;
        Ry/bohr^2, not divided by the masses).
    q_cryst : NDArray[np.float64], shape ``(nq, 3)``
        The matching q-points in crystal coordinates (any order).
    masses_amu : NDArray[np.float64], shape ``(nat,)``
        Atomic masses (amu).
    qgrid : tuple of int
        Coarse phonon q-grid.
    at : NDArray[np.float64], shape ``(3, 3)``
        Real-lattice vectors (rows, alat).
    atom_positions : NDArray[np.float64], shape ``(nat, 3)``, optional
        Atomic positions (crystal coordinates).  When given, the force
        constants are interpolated with atom-pair Wigner-Seitz images (as QE
        ``matdyn``), which keeps the dense phonons symmetric in cells with
        several atoms.

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz (nmode,), z (nmode, ncart))``, the
        same convention as :func:`phonon_interp_from_dyn`.

    Raises
    ------
    ValueError
        If the q-points do not cover the full grid.
    """
    qgrid = tuple(int(n) for n in qgrid)
    nmode = dynq.shape[0]
    Cgrid = np.zeros(qgrid + (nmode, nmode), dtype=complex)
    filled = np.zeros(qgrid, dtype=bool)
    labels = np.round(np.asarray(q_cryst) * np.asarray(qgrid)).astype(int) % np.asarray(qgrid)
    for iq, lab in enumerate(map(tuple, labels)):
        Cgrid[lab] = dynq[:, :, iq]
        filled[lab] = True
    if not filled.all():
        raise ValueError(
            'EPW q-points cover only %d/%d q of the grid.' % (filled.sum(), filled.size)
        )
    masses_ry = np.asarray(masses_amu, dtype=float) * AMU_RY
    return _phonon_interp_from_force_constants(
        Cgrid, masses_ry, qgrid, at, atom_positions, long_range=long_range
    )


def _phonon_interp_from_force_constants(
    Cgrid: NDArray[np.complex128],
    masses_ry: NDArray[np.float64],
    qgrid: tuple[int, int, int],
    at: NDArray[np.float64],
    atom_positions: NDArray[np.float64] | None = None,
    long_range=None,
) -> Callable[[NDArray[np.float64]], tuple[NDArray[np.float64], NDArray[np.complex128]]]:
    """Wigner-Seitz phonon interpolator from force constants on the full coarse grid.

    Parameters
    ----------
    Cgrid : NDArray[np.complex128], shape ``(nq1, nq2, nq3, 3 nat, 3 nat)``
        Force-constant matrices ``C(q)`` (Ry/bohr^2) placed on the grid.
    masses_ry : NDArray[np.float64], shape ``(nat,)``
        Atomic masses in QE Rydberg units.
    qgrid : tuple of int
        Coarse phonon q-grid.
    at : NDArray[np.float64], shape ``(3, 3)``
        Real-lattice vectors (rows, alat).
    atom_positions : NDArray[np.float64], shape ``(nat, 3)``, optional
        Atomic positions (crystal coordinates) for atom-pair Wigner-Seitz
        images; ``None`` uses single-site images.
    long_range : LongRangeModel, optional
        Subtract its force constants on the grid and restore them at every q
        (:func:`phonon_interp_from_dyn`).

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz (nmode,), z (nmode, ncart))``.
    """
    # q -> R_p (same 1/N, sign convention as build_g_ReRp), simple acoustic sum
    # rule and Wigner-Seitz interpolation: the q2r/matdyn chain of
    # PAOFLOW.phonon.qe_force_constants.
    restore = None
    if long_range is not None:
        Cgrid = np.array(Cgrid, dtype=complex)  # private copy: the short-range remainder
        ng_q = np.asarray(qgrid)
        for lab in np.ndindex(*qgrid):
            Cgrid[lab] -= long_range.dynamical_matrix(np.asarray(lab, dtype=float) / ng_q)
        restore = long_range.dynamical_matrix
    phonon_at_q = phonon_interpolator(
        real_space_force_constants(Cgrid), masses_ry, qgrid, at, atom_positions, long_range=restore
    )
    phonon_at_q.long_range = long_range
    return phonon_at_q


def _read_dyn_matrices(path):
    """Force-constant matrices of every star-q in a QE ``*.dyn`` file.

    Returns
    -------
    dict
        ``{'q_cart': list of (3,), 'C': list of (3*nat, 3*nat), 'masses_ry': (nat,)}``
        (see :func:`PAOFLOW.phonon.qe_force_constants.read_dynamical_matrix_file`).
    """
    parsed = read_dynamical_matrix_file(path)
    return {
        'q_cart': parsed['q_cart'],
        'C': parsed['force_constants'],
        'masses_ry': parsed['masses_ry'],
    }


def _dyn_float(tok):
    return float(tok.replace('D', 'E').replace('d', 'e'))


def _phonon_modes_at_q(q_cryst, phonon_at_q):
    """Return ``(freq_thz (nmode,), z (nmode, ncart))`` for one dense q.

    ``phonon_at_q`` is a user-supplied callable ``q_cryst -> (freq_thz, z)`` that
    Wigner-Seitz interpolates the dynamical matrix (e.g. wrapping
    :func:`PAOFLOW.elphon.qe_matdyn._matrix_at_q_ws` + ``eigh`` + signed
    frequencies).  ``z`` MUST use the same normalisation as
    :func:`~PAOFLOW.elphon.qe_elph_io.read_qe_dyn` (``TODO: verify_evec``).
    """
    freq_thz, z = phonon_at_q(np.asarray(q_cryst, dtype=float))
    return np.asarray(freq_thz, dtype=float), np.asarray(z)


# --------------------------------------------------------------------------- #
# 4. Dense-q driver
# --------------------------------------------------------------------------- #
def _crystal_point_group(s_cryst, tau_cryst, species, tol=1.0e-4):
    """Filter the lattice rotations down to the true crystal point group.

    Keeps each integer rotation ``S`` (crystal axes, acting on real-space crystal
    coordinates as ``r' = S r``) for which some translation ``t`` maps every atom
    onto an atom of the same species -- i.e. ``S`` is the rotational part of a
    space-group operation.  Prevents over-symmetrising the q-grid for crystals
    whose basis lowers the symmetry below the lattice holohedry.
    """
    tau = np.asarray(tau_cryst, dtype=float) % 1.0
    sp = np.asarray(species)
    same0 = np.nonzero(sp == sp[0])[0]
    keep = []
    for S in s_cryst:
        S = np.asarray(S)
        rot = (tau @ S.T) % 1.0  # S r for every atom (row-vector form)
        for b0 in same0:
            t = (tau[b0] - rot[0]) % 1.0
            shifted = (rot + t) % 1.0
            ok = True
            for a in range(len(tau)):
                d = np.abs(((shifted[a] - tau + 0.5) % 1.0) - 0.5)
                if not np.any(np.all(d < tol, axis=1) & (sp == sp[a])):
                    ok = False
                    break
            if ok:
                keep.append(S)
                break
    return keep


def irreducible_qmesh(nq, rots_cryst, at, bg, include_tr=True):
    """Fold a Gamma-centred ``nq^3`` q-grid into the irreducible wedge.

    The rotations (integer crystal-axis matrices) are applied in Cartesian
    (``R = at^T S at^-T``); because they form an orthogonal group, the orbit is
    independent of the ``S`` vs ``S^T`` convention.  Time reversal (``q -> -q``)
    is added by default (``lambda_q = lambda_{-q}``).

    Returns
    -------
    q_reps : ndarray ``(nir, 3)``
        One representative q per star (crystal coordinates).
    weights : ndarray ``(nir,)``
        Star multiplicities (sum to ``nq^3``).
    """
    mesh, representatives, weights, _ = irreducible_mesh(nq, rots_cryst, at, bg, include_tr)
    return mesh[representatives], weights


def irreducible_mesh(
    n: int,
    rots_cryst: Sequence[NDArray],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
    include_tr: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.int_], NDArray[np.float64], NDArray[np.int_]]:
    """Stars of a Gamma-centred ``n^3`` grid under a point group.

    Parameters
    ----------
    n : int
        Grid size along each reciprocal axis.
    rots_cryst : sequence of ndarray, shape ``(3, 3)``
        Point-group rotations (integer, crystal axes), e.g. the output of
        :func:`_crystal_point_group`.
    at, bg : ndarray, shape ``(3, 3)``
        Direct and reciprocal lattice vectors (rows, alat and 2 pi / alat units).
    include_tr : bool, optional
        Add time reversal (``k -> -k``) to the group (default ``True``).

    Returns
    -------
    mesh : ndarray, shape ``(n^3, 3)``
        The full grid in crystal coordinates, flat index ``(i1 n + i2) n + i3``.
    representatives : ndarray, shape ``(nir,)``
        Flat index of one representative point per star.
    weights : ndarray, shape ``(nir,)``
        Star multiplicities (sum to ``n^3``).
    star_index : ndarray, shape ``(n^3,)``
        Star of every grid point (index into ``representatives``).

    Notes
    -----
    The rotations act on Cartesian vectors as :math:`R = A S A^{-1}`, with
    :math:`A` the matrix of direct lattice vectors as columns.
    """
    nq = n
    ax = [np.arange(nq) / nq for _ in range(3)]
    qm = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    A = np.asarray(at, dtype=float).T
    Ainv = np.linalg.inv(A)
    Rs = [A @ np.asarray(S, dtype=float) @ Ainv for S in rots_cryst]
    if include_tr:
        Rs = Rs + [-R for R in Rs]
    BT = np.asarray(bg, dtype=float).T
    Binv = np.linalg.inv(BT)
    lab2idx = {tuple(np.round(q * nq).astype(int) % nq): i for i, q in enumerate(qm)}
    assigned = -np.ones(len(qm), dtype=int)
    reps, wts = [], []
    for i in range(len(qm)):
        if assigned[i] >= 0:
            continue
        qc = BT @ qm[i]
        orbit = set()
        for R in Rs:
            y = Binv @ (R @ qc)
            yl = np.round(y * nq)
            if np.allclose(y * nq, yl, atol=1.0e-6):
                orbit.add(lab2idx[tuple(yl.astype(int) % nq)])
        for j in orbit:
            assigned[j] = len(reps)
        reps.append(i)
        wts.append(len(orbit))
    return qm, np.asarray(reps, dtype=int), np.asarray(wts, dtype=float), assigned


def eliashberg_dense_q(
    A,
    HRs,
    kpts_cryst,
    bg,
    at,
    coupling_dir,
    qgrid_coarse,
    q_cryst_coarse,
    dyn_paths_full,
    ng,
    phonon_at_q,
    nq_dense=None,
    source='elphmat',
    masses_amu=None,
    nk_dense=18,
    sigmas_ry=(0.02,),
    nelec=None,
    mu_star=0.10,
    ispin=0,
    isig=0,
    sigma_w_frac=0.02,
    fs_window=8.0,
    min_freq_thz=0.0,
    sym_rots=None,
    tau_cryst=None,
    species=None,
    comm=None,
    fs_coupling=False,
    fsthick_ev=None,
    n_freq_fs=100,
    orbital_positions=None,
    allow_missing_long_range=False,
    transport=False,
    alat=None,
    cell_volume=None,
    spin_degeneracy=2.0,
    adaptive_smearing=None,
    save_dir=None,
    pseudo_dir=None,
    prefix=None,
    basis='auto',
    unfold='full',
    basis_shells=None,
    unfold_tol=1e-6,
    cache_dir=None,
    cache_tag=None,
    long_range=None,
):
    """SKETCH: Eliashberg properties with BOTH k and q interpolated.

    adaptive_smearing : float, optional
        Adaptive Fermi-surface smearing (Yates et al., 0.7 for Gaussians): the
        delta of every dense state gets the width ``adaptive_smearing * |v_nk| *
        dk``, floored at ``sigmas_ry`` (one series entry per floor); needs
        ``alat`` (the group velocities are computed).  ``None`` keeps the fixed
        ``sigmas_ry``.
    Mirrors :func:`~PAOFLOW.elphon.do_pao_eph.eliashberg_from_qe_coupling`, but
    replaces the coarse-q loop by (a) one build of ``g(R_e, R_p)`` from the full
    coarse-q couplings and (b) a loop over a dense ``nq_dense^3`` q-grid, WS-
    interpolating both the vertex (:func:`g_Re_at_q`) and the phonons
    (``phonon_at_q``) to each dense q.

    Parameters
    ----------
    q_cryst_coarse : ndarray ``(nq_coarse, 3)`` or None
        FULL coarse q-grid in crystal coordinates (``TODO: unfold_star`` if only
        the irreducible set is available).  For ``source='epw'``, ``None`` takes
        EPW's q-points, which already cover the full grid.
    dyn_paths_full : sequence of str or None
        One ``*.dyn`` per full coarse q (used only for ``source='ahc'`` to supply
        the q-point of each dump).
    phonon_at_q : callable or None
        ``q_cryst -> (freq_thz, z)``; see :func:`_phonon_modes_at_q`.  For
        ``source='epw'``, ``None`` builds it from EPW's force constants
        (:func:`phonon_interp_from_epw`).
    source : {'elphmat', 'ahc', 'epw', 'dfpt'}, optional
        Coarse coupling input (see
        :func:`~PAOFLOW.elphon.do_pao_eph.eliashberg_from_qe_coupling`).  Only
        ``'epw'`` and ``'dfpt'`` give vertices whose band phases match ``A`` for
        every q, which the ``q -> R_p`` interpolation requires; ``'dfpt'`` needs
        ``save_dir`` (and ``coupling_dir`` = the ``ph.x`` outputs).
    nq_dense : int, optional
        Dense q-grid size (defaults to ``nk_dense`` so k+q stays commensurate).
    min_freq_thz : float, optional
        Soft-mode guard: modes with ``freq_thz < min_freq_thz`` are dropped from
        ``lambda`` (default 0.0 -> only imaginary ``omega^2 < 0`` modes).  Coarse
        DFPT-grid Fourier interpolation can overshoot the acoustic branch into
        spurious soft/imaginary modes whose ``1/omega^2`` blows up ``lambda``;
        genuine near-Gamma acoustic modes contribute negligibly (their coupling
        vanishes), so raising this to a few kelvin (~0.05-0.1 THz) is safe.
    sym_rots : ndarray ``(nrot, 3, 3)``, optional
        Lattice point-group rotations (crystal axes, integer), e.g. ``read_nscf``
        ``'s_cryst'``.  When given, the dense q-grid is folded to its irreducible
        wedge (with star-multiplicity weights), evaluating only the inequivalent
        q -- an up-to-48x (cubic) speedup with an identical result.  ``tau_cryst``
        and ``species`` filter the holohedry down to the true crystal point group.
    tau_cryst : ndarray ``(natom, 3)``, optional
        Atomic positions (crystal coords) for the point-group basis filter.
    species : sequence, optional
        Per-atom species labels for the basis filter.
    comm : mpi4py communicator, optional
        Distributes the dense-q loop across ranks (``MPI.COMM_WORLD`` by default);
        run ``mpirun -np N python ...`` for an up-to-``nq_dense^3``-fold speedup.
        The coarse ``g(R_e,R_p)`` build and the dense-electron cache are computed
        redundantly on every rank; only the per-q interpolation is parallelised.
    fs_coupling : bool, optional
        Also accumulate the state-resolved Fermi-surface coupling of the
        anisotropic Migdal-Eliashberg equations
        (:mod:`~PAOFLOW.elphon.fermi_surface_coupling`), returned under the key
        ``'fs_coupling'``.  Requires ``sym_rots``: the states are folded to the
        irreducible wedge of the dense k-grid.  Use ``nq_dense = nk_dense``:
        with a coarser q-grid, ``k+q`` reaches only a sublattice of the k-grid
        and the anisotropic equations split into independent sublattice
        problems, each with its own ``Tc``.
    fsthick_ev : float, optional
        Half-width (eV) of the Fermi window of the anisotropic states (EPW
        ``fsthick``); defaults to ``fs_window`` smearings.
    n_freq_fs : int, optional
        Number of phonon-frequency points of the Fermi-surface coupling
        (default 100).
    orbital_positions : ndarray ``(nawf, 3)``, optional
        PAO orbital centres in crystal coordinates
        (:func:`~PAOFLOW.elphon.elph_bloch.pao_orbital_positions`).  When given,
        every Wigner-Seitz sum is pair-resolved: electrons and the vertex
        ``R_e`` over orbital pairs, the vertex ``R_p`` over (orbital, displaced
        atom) pairs and the EPW force constants over atom pairs (``tau_cryst``
        required).  This keeps the dense interpolation symmetric in cells with
        several atoms; ``None`` keeps the single-site images.
    allow_missing_long_range : bool, optional
        For ``source='epw'``, run a polar material without the long-range dipole
        term instead of raising
        (:func:`~PAOFLOW.elphon.do_pao_eph.check_long_range_terms`).
    transport : bool, optional
        Also compute the transport Eliashberg function (``a2F_tr``,
        ``lambda_tr``, ``lambda_tr_qv``), the Fermi-surface plasma-frequency
        tensor ``omega_plasma2_ev2`` (3x3, eV^2) and its isotropic
        ``omega_plasma_ev`` (:math:`\\sqrt{\\mathrm{Tr}/3}`, eV), the inputs of metal transport and
        optics (:class:`~PAOFLOW.boltzmann.scattering_channels.TransportSpectralChannel`,
        :mod:`PAOFLOW.response.drude`).  Needs ``alat`` and ``cell_volume``.
    alat : float, optional
        Lattice parameter (bohr), for the band velocities.
    cell_volume : float, optional
        Unit-cell volume (bohr^3), for the plasma frequency.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.  The band velocities are the
        group velocities of the PAO Hamiltonian (no non-local correction, which
        belongs to interband dipoles only).
    save_dir, pseudo_dir, prefix, basis, unfold, basis_shells, unfold_tol, cache_dir, cache_tag
        ``source='dfpt'`` and cache options of :func:`prepare_dense_vertex`.

    Notes
    -----
    Deferred: q-phase validation and the polar Frohlich term.
    """
    if masses_amu is None:
        raise ValueError('masses_amu is required to mass-weight the phonon eigenvectors')
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY  # (ncart,)
    if nq_dense is None:
        nq_dense = nk_dense  # keep the dense q-grid commensurate with k for k+q

    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    vertex = prepare_dense_vertex(
        A, kpts_cryst, bg, at, coupling_dir, qgrid_coarse, q_cryst_coarse, ng, phonon_at_q,
        source=source, masses_amu=masses_amu, tau_cryst=tau_cryst,
        orbital_positions=orbital_positions, comm=comm,
        allow_missing_long_range=allow_missing_long_range,
        save_dir=save_dir, pseudo_dir=pseudo_dir, prefix=prefix, basis=basis, unfold=unfold,
        basis_shells=basis_shells, unfold_tol=unfold_tol, cache_dir=cache_dir, cache_tag=cache_tag,
        long_range=long_range,
    )  # fmt: skip
    g_ReRp, win, phonon_at_q = vertex['g_ReRp'], vertex['window'], vertex['phonon_at_q']
    Nint_p, W_p, Midx_p = vertex['phonon_ws']
    nmodes = int(mass_flat_ry.size)

    # --- electron cache (dense k), shared by every dense q ----------------- #
    if adaptive_smearing is not None and alat is None:
        raise ValueError('adaptive_smearing needs alat (group velocities on the dense grid)')
    if transport and (alat is None or cell_volume is None):
        raise ValueError('transport=True needs alat (bohr) and cell_volume (bohr^3).')
    electrons = precompute_dense_electrons(
        HRs,
        at,
        nk_dense,
        np.atleast_1d(sigmas_ry),
        nelec,
        tuple(ng),
        ispin=ispin,
        fs_window=fs_window,
        orbital_positions=orbital_positions,
        velocities=transport or adaptive_smearing is not None,
        alat=alat,
        diagonal_velocities=transport or adaptive_smearing is not None,
        comm=comm,
        spin_degeneracy=spin_degeneracy,
        adaptive_smearing=adaptive_smearing,
    )
    if adaptive_smearing is not None and rank == 0:
        # widths of the states that matter: within fs_window local widths of E_F
        for isig, sig in enumerate(np.atleast_1d(sigmas_ry)):
            widths = electrons['widths'][isig]
            near = np.abs(electrons['E'] - electrons['ef_sig'][isig]) < fs_window * widths
            w = widths[near] * RY_TO_EV
            print(
                '  adaptive Fermi-surface smearing a = %.2f (floor %.1f meV): widths of the states '
                'near E_F %.1f / %.1f / %.1f meV (min / median / max), N(E_F) = %.4f states/eV'
                % (adaptive_smearing, 1e3 * sig * RY_TO_EV, 1e3 * w.min(), 1e3 * np.median(w),
                   1e3 * w.max(), electrons['dos'][isig] / RY_TO_EV),
                flush=True,
            )  # fmt: skip

    # --- (b) dense q-grid loop (distributed over MPI ranks) ---------------- #
    # Fold to the irreducible wedge when symmetries are supplied (identical
    # result, far fewer q); otherwise sample the full grid with unit weights.
    if fs_coupling and sym_rots is None:
        raise ValueError('fs_coupling requires sym_rots (irreducible Fermi-surface states).')
    if sym_rots is not None:
        rots = sym_rots
        if tau_cryst is not None and species is not None:
            rots = _crystal_point_group(sym_rots, tau_cryst, species)
        qmesh, qweights = irreducible_qmesh(nq_dense, rots, at, bg)
        if rank == 0:
            print(
                '  dense-q symmetry: %d irreducible / %d full q  (%.1fx fewer)'
                % (qmesh.shape[0], nq_dense**3, nq_dense**3 / qmesh.shape[0]),
                flush=True,
            )
    else:
        ax = [np.arange(nq_dense) / nq_dense for _ in range(3)]
        qmesh = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
        qweights = np.ones(qmesh.shape[0])
    nqd = qmesh.shape[0]
    qstart, qstop = load_balancing(size, rank, nqd)
    lam_qv = np.zeros((nqd, nmodes))
    lam_tr_qv = np.zeros((nqd, nmodes)) if transport else None
    om_qv = np.zeros((nqd, nmodes))
    # Phonons first: the Fermi-surface coupling needs the highest frequency
    # (over all ranks) for its frequency grid before the coupling loop.
    phonons = {iq: _phonon_modes_at_q(qmesh[iq], phonon_at_q) for iq in range(qstart, qstop)}
    pair_coupling = None
    if fs_coupling:
        omega_max_thz = max((float(np.abs(f).max()) for f, _ in phonons.values()), default=0.0)
        if size > 1:
            omega_max_thz = comm.allreduce(omega_max_thz, op=MPI.MAX)
        pair_coupling = _fermi_surface_accumulator(
            electrons, rots, at, bg, nk_dense, fsthick_ev, fs_window, omega_max_thz,
            n_freq_fs, isig,
        )  # fmt: skip
        if rank == 0:
            print(
                '  Fermi-surface coupling: %d irreducible states within %.3f eV of E_F'
                % (pair_coupling.nstates, pair_coupling.fsthick_ev),
                flush=True,
            )
            if nq_dense != nk_dense:
                print(
                    '  WARNING: nq_dense (%d) != nk_dense (%d): k+q reaches only a sublattice'
                    ' of the k-grid, so the anisotropic equations split into independent'
                    ' sublattice problems.  Use nq_dense = nk_dense.' % (nq_dense, nk_dense),
                    flush=True,
                )
    for iq in range(qstart, qstop):
        q_cryst = qmesh[iq]
        freq_thz, z = phonons.pop(iq)  # (nmode,), (nmode, ncart)
        zmass = z / np.sqrt(mass_flat_ry)[None, :]
        gR = g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p)
        is_gamma = np.linalg.norm(q_cryst - np.round(q_cryst)) < 1.0e-6
        q_weight = qweights[iq] / qweights.sum()
        pair_weights = None
        if pair_coupling is not None:
            # Same mode exclusions as lambda_qv below.
            pair_weights = np.where(freq_thz < min_freq_thz, 0.0, q_weight)
            if is_gamma:
                pair_weights[:] = 0.0
        lr_q = None
        if long_range is not None:
            lr_q = long_range.vertex_pao(q_cryst, A.shape[1], orbital_positions)
        res = lambda_q_dense_ws_fast(
            gR, electrons, q_cryst, zmass, freq_thz,
            pair_coupling=pair_coupling, pair_weights=pair_weights, pair_q_weight=q_weight,
            transport=transport, long_range=lr_q,
        )  # fmt: skip
        lam = res['lambda_qnu'][isig].copy()
        lam[freq_thz < min_freq_thz] = 0.0  # drop spurious soft/imaginary modes
        if is_gamma:
            lam[:] = 0.0  # zero the Gamma acoustic blow-up (QE convention)
        lam_qv[iq] = lam
        om_qv[iq] = np.abs(freq_thz)
        if transport:
            lam_tr = res['lambda_tr_qnu'][isig].copy()
            lam_tr[freq_thz < min_freq_thz] = 0.0  # same exclusions as lambda
            if is_gamma:
                lam_tr[:] = 0.0
            lam_tr_qv[iq] = lam_tr

    # Each rank wrote a disjoint slice; a single SUM reduction rebuilds the grid.
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, lam_qv, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, om_qv, op=MPI.SUM)
        if transport:
            comm.Allreduce(MPI.IN_PLACE, lam_tr_qv, op=MPI.SUM)

    win.Free()  # release the shared g(R_e, R_p) window (all reads are done)

    # star-multiplicity weights (unit weights when no symmetry folding).
    out = eliashberg_from_modes(
        lam_qv, om_qv, q_weights=qweights, mu_star=mu_star, sigma_w_frac=sigma_w_frac
    )
    out['lambda_qv'] = lam_qv
    out['omega_qv_thz'] = om_qv
    if transport:
        out_tr = eliashberg_from_modes(
            lam_tr_qv, om_qv, q_weights=qweights, mu_star=mu_star, sigma_w_frac=sigma_w_frac
        )
        out['a2F_tr'] = out_tr['a2F']
        out['lambda_tr'] = out_tr['lambda']
        out['lambda_tr_qv'] = lam_tr_qv
        plasma2 = fermi_surface_plasma_frequency(electrons, cell_volume, isig, spin_degeneracy)
        out['omega_plasma2_ev2'] = plasma2  # (3, 3) tensor
        out['omega_plasma_ev'] = float(np.sqrt(np.trace(plasma2) / 3.0))  # isotropic
    if pair_coupling is not None:
        pair_coupling.reduce(comm)
        out['fs_coupling'] = pair_coupling.result(bg, at, nq_dense)
    release_dense_electrons(electrons)
    return out


def prepare_dense_vertex(
    A: NDArray[np.complex128],
    kpts_cryst: NDArray[np.float64],
    bg: NDArray[np.float64],
    at: NDArray[np.float64],
    coupling_dir: str,
    qgrid_coarse: Sequence[int],
    q_cryst_coarse: NDArray[np.float64] | None,
    ng: Sequence[int],
    phonon_at_q: Callable | None,
    *,
    source: str = 'epw',
    masses_amu: NDArray[np.float64],
    tau_cryst: NDArray[np.float64] | None = None,
    orbital_positions: NDArray[np.float64] | None = None,
    comm=None,
    allow_missing_long_range: bool = False,
    save_dir: str | None = None,
    pseudo_dir: str | None = None,
    prefix: str | None = None,
    basis: str = 'auto',
    unfold: str = 'full',
    basis_shells=None,
    unfold_tol: float = 1e-6,
    cache_dir: str | None = None,
    cache_tag: str | None = None,
    bands=None,
    long_range=None,
) -> dict:
    """Coarse couplings -> shared ``g(R_e, R_p)`` and the phonon interpolator.

    Common setup of every dense-q electron-phonon property
    (:func:`eliashberg_dense_q`, the phonon-assisted absorption of
    :mod:`~PAOFLOW.elphon.phonon_assisted_absorption`): read the coarse
    couplings, guard against missing long-range terms, build the phonon
    interpolator and the doubly real-space vertex, and the Wigner-Seitz images of
    its phonon cells.

    Parameters
    ----------
    A : ndarray, shape ``(nbnd, nawf, nk)``, complex
        PAO projections on the coarse k-grid.
    kpts_cryst : ndarray, shape ``(nk, 3)``
        Coarse k-points (crystal coordinates).
    bg, at : ndarray, shape ``(3, 3)``
        Reciprocal (rows, 2 pi/alat) and direct (rows, alat) lattice vectors.
    coupling_dir : str
        EPW directory (``source='epw'``), ``ph.x`` outputs (``source='dfpt'``: the
        ``pp.py`` ``save/`` directory or the run directory holding ``_ph0/``) or QE
        coupling directory.
    qgrid_coarse : sequence of int
        Coarse q-grid.
    q_cryst_coarse : ndarray, shape ``(nq_coarse, 3)`` or None
        Full coarse q-grid; ``None`` takes EPW's q-points, or the star-unfolded grid of
        the ``ph.x`` run.
    ng : sequence of int
        Coarse k-grid (real-space electron cells of the vertex).
    phonon_at_q : callable or None
        ``q_cryst -> (freq_thz, z)``; ``None`` builds it from EPW's force constants or
        from the ``ph.x`` dyn files.
    source : {'epw', 'dfpt', 'ahc', 'elphmat'}, optional
        Coarse coupling input (see :func:`eliashberg_dense_q`).  ``'dfpt'`` computes the
        vertex natively from ``dvscf`` and the nscf wavefunctions
        (:class:`~PAOFLOW.elphon.dfpt_vertex.CoarseVertexSetup`), without EPW.
    masses_amu : ndarray, shape ``(nat,)``
        Atomic masses (amu), one per atom.
    tau_cryst : ndarray, shape ``(nat, 3)``, optional
        Atomic positions; required with ``orbital_positions``.
    orbital_positions : ndarray, shape ``(nawf, 3)``, optional
        PAO orbital centres for the pair-resolved Wigner-Seitz sums.
    comm : mpi4py communicator, optional
        ``MPI.COMM_WORLD`` by default.
    allow_missing_long_range : bool, optional
        Run a polar material without the long-range dipole term instead of raising.
    save_dir : str, optional
        ``source='dfpt'``: the nscf save directory of the projections ``A`` (wavefunctions).
    pseudo_dir : str, optional
        ``source='dfpt'``: directory of the UPF files (default: ``save_dir``).
    prefix : str, optional
        ``source='dfpt'``: QE prefix (inferred from the ``dvscf`` file names by default).
    basis : {'auto', 'pao', 'band'}, optional
        ``source='dfpt'``: apply the perturbation to the projected PAO functions or to the
        bands (:meth:`~PAOFLOW.elphon.dfpt_vertex.CoarseVertexSetup.pao_vertex_at_q`).
    unfold : {'full', 'irreducible'}, optional
        ``source='dfpt'``: compute the vertex at every coarse q (``'full'``), or only at the
        irreducible q of ``ph.x`` and rotate it to the stars in the PAO basis
        (:func:`~PAOFLOW.elphon.dfpt_vertex.unfolded_pao_vertices`, ``|star|`` times cheaper;
        exact when the nscf band set does not cut a degenerate multiplet, which a self-check
        verifies on one member).
    basis_shells : DataController or tuple, optional
        ``unfold='irreducible'``: the PAO basis (``data_controller`` after ``projections``, or
        ``(shells_per_atom, atom_labels)`` of :func:`~PAOFLOW.elphon.dfpt_vertex.orbital_shells`).
    unfold_tol : float, optional
        Tolerance of the unfolding self-check.
    cache_dir : str, optional
        Directory of the ``g(R_e, R_p)`` disk cache (:mod:`PAOFLOW.elphon.coupling_cache`);
        the array is read from there when an entry with the same key exists, and written
        after the build otherwise.
    long_range : LongRangeModel, optional
        Polar crystals: subtract the analytic dipole/quadrupole coupling on the coarse grid
        and restore it at every dense q (:class:`~PAOFLOW.elphon.long_range.LongRangeModel`);
        the phonons must be built with the same model (``phonon_at_q=None`` or
        :func:`phonon_interp_from_dyn` with ``long_range``).
    cache_tag : str, optional
        Identity of the PAO basis for the cache key (:func:`~PAOFLOW.elphon.coupling_cache.basis_tag`).
    bands : sequence of int, optional
        ``source='dfpt'``: restrict the band projector of the vertex to these nscf bands
        (EPW's ``exclude_bands``; default all bands, the consistent choice for the PAO
        Hamiltonian).

    Returns
    -------
    dict
        ``g_ReRp`` ``(nawf, nawf, ncart, n1e, n2e, n3e, nq1, nq2, nq3)`` (one copy per
        node in MPI shared memory), ``window`` (the ``MPI.Win`` to ``Free`` once
        all reads are done), ``phonon_ws`` ``(Nint_p, W_p, Midx_p)``,
        ``phonon_at_q``, and the ``epsi`` ``(3, 3)`` and ``zstar`` ``(nat, 3, 3)``
        stored with EPW couplings (``None`` for the other sources).
    """
    masses_amu = np.asarray(masses_amu, dtype=float)
    qgrid_coarse = tuple(int(n) for n in qgrid_coarse)
    nbnd, nk = int(A.shape[0]), int(A.shape[2])
    nmodes = 3 * int(masses_amu.size)
    from mpi4py import MPI

    if comm is None:
        comm = MPI.COMM_WORLD
    # g(R_e, R_p) is large (~GBs) and read-only, and every rank needs all of it.
    # Allocate ONE copy per node in MPI shared memory (built by the node-local
    # rank 0 only) instead of an independent copy per rank, which otherwise
    # multiplies the memory by the number of ranks on the node.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()

    epw = None
    dfpt = None
    small = {'epsi': None, 'zstar': None}
    if source == 'dfpt':
        from .dfpt_vertex import CoarseVertexSetup

        if save_dir is None:
            raise ValueError("source='dfpt' needs save_dir, the nscf save of the projections")
        # Every rank reads the small inputs (XML, UPF tables, patterns); the vertex of
        # each coarse q is computed by one node rank and written to its q-cell.
        dfpt = CoarseVertexSetup(
            save_dir, coupling_dir, qgrid_coarse, prefix=prefix, pseudo_dir=pseudo_dir
        )
        small['epsi'], small['zstar'] = dfpt.long_range_tensors()
        check_long_range_terms(
            small['zstar'], small['epsi'], allow_missing_long_range, long_range=long_range
        )
        if q_cryst_coarse is None:
            q_cryst_coarse = dfpt.coarse_q()
        if phonon_at_q is None:
            atom_positions = None
            if orbital_positions is not None:
                if tau_cryst is None:
                    raise ValueError('orbital_positions requires tau_cryst (atom-pair phonon WS).')
                atom_positions = tau_cryst
            phonon_at_q = phonon_interp_from_dyn(
                list(dfpt.files.dyn), qgrid_coarse, bg, at, atom_positions, long_range=long_range
            )
    if source == 'epw':
        # EPW provides the full coarse q-grid (unfolded from the irreducible ph.x
        # q) and its force constants; both default from the .epb files.  Only
        # node rank 0, which builds g(R_e, R_p), reads the couplings (GBs for
        # dense coarse grids); the other ranks receive the small arrays.
        if node_rank == 0:
            epw = load_epw_coupling(coupling_dir, nbnd, nk, masses_amu.size, bg)
        small = node_comm.bcast(
            {key: epw[key] for key in ('zstar', 'epsi', 'q_cryst', 'dynq')}
            if node_rank == 0
            else None,
            root=0,
        )
        check_long_range_terms(
            small['zstar'], small['epsi'], allow_missing_long_range, long_range=long_range
        )
        if q_cryst_coarse is None:
            q_cryst_coarse = small['q_cryst']
        if phonon_at_q is None:
            atom_positions = None
            if orbital_positions is not None:
                if tau_cryst is None:
                    raise ValueError('orbital_positions requires tau_cryst (atom-pair phonon WS).')
                atom_positions = tau_cryst
            phonon_at_q = phonon_interp_from_epw(
                small['dynq'], small['q_cryst'], masses_amu, qgrid_coarse, at, atom_positions,
                long_range=long_range,
            )  # fmt: skip
    if long_range is not None and getattr(phonon_at_q, 'long_range', None) is not long_range:
        raise ValueError(
            'long_range is set but phonon_at_q was built without it: pass the same model to '
            'phonon_interp_from_dyn / phonon_interp_from_epw (LO-TO splitting), or let the '
            'driver build the phonons (phonon_at_q=None).'
        )
    q_cryst_coarse = np.asarray(q_cryst_coarse, dtype=float)
    nawf = int(A.shape[1])
    qg = qgrid_coarse
    shape = (nawf, nawf, nmodes, ng[0], ng[1], ng[2], qg[0], qg[1], qg[2])
    itemsize = np.dtype(np.complex128).itemsize
    nelem = int(np.prod(shape))
    win = MPI.Win.Allocate_shared(
        nelem * itemsize if node_rank == 0 else 0, itemsize, comm=node_comm
    )
    buf, _ = win.Shared_query(0)
    g_ReRp = np.ndarray(buffer=buf, dtype=np.complex128, shape=shape)

    # --- cache of the finished g(R_e, R_p) (optional) ----------------------- #
    cache_key = None
    cached = False
    if cache_dir is not None:
        from .coupling_cache import dense_vertex_cache_key, load_dense_vertex

        tag = cache_tag if long_range is None else '%s|lr=%s' % (cache_tag, long_range.signature())
        cache_key = dense_vertex_cache_key(
            coupling_dir, source, qgrid_coarse, ng, masses_amu, A.shape, tag
        )
        if node_rank == 0:
            cached = load_dense_vertex(cache_dir, cache_key, g_ReRp)
        cached = node_comm.bcast(cached, root=0)
        if cached and comm.Get_rank() == 0:
            print(
                '  g(R_e, R_p) read from the cache %s (key %s)' % (cache_dir, cache_key), flush=True
            )
    node_comm.Barrier()

    # --- (a) coarse half-vertices g_q(R_e) -> g(R_e, R_p), on node rank 0 --- #
    # Each g_q(R_e) goes straight to its q-cell of the shared buffer, which is
    # then Fourier transformed q -> R_p in place one orbital row at a time
    # (the arithmetic of build_g_ReRp without its full-size temporaries).
    if cached:
        pass
    elif source == 'dfpt':
        # The q-cells are disjoint: the node ranks share the coarse q and then the
        # orbital rows of the q -> R_p transform (identical to the serial loop).
        nq_coarse = qg[0] * qg[1] * qg[2]
        labels = np.round(q_cryst_coarse * np.asarray(qg)).astype(int) % np.asarray(qg)
        if q_cryst_coarse.shape[0] != nq_coarse or len({tuple(lab) for lab in labels}) != nq_coarse:
            raise ValueError(
                'the coarse q-points (%d) must tile the full %s grid'
                % (q_cryst_coarse.shape[0], 'x'.join(map(str, qg)))
            )
        member_of = {
            tuple(np.round(m.q_cryst * np.asarray(qg)).astype(int) % np.asarray(qg)): m
            for m in dfpt.members
        }
        kidx = np.round(np.asarray(kpts_cryst, dtype=float) * np.asarray(ng)).astype(
            int
        ) % np.asarray(ng)
        node_size = node_comm.Get_size()
        if node_rank == 0:
            g_ReRp[...] = 0.0
        node_comm.Barrier()
        if unfold == 'irreducible':
            from .dfpt_vertex import (
                orbital_shells,
                pao_vertex_rotation_operators,
                unfolded_pao_vertices,
            )
            from .elph_bloch import pao_vertex_to_cells

            if basis_shells is None:
                raise ValueError(
                    "unfold='irreducible' needs basis_shells (the data_controller of the projections)"
                )
            shells, atom_labels = (
                basis_shells if isinstance(basis_shells, tuple) else orbital_shells(basis_shells)
            )
            ops = pao_vertex_rotation_operators(dfpt, shells, atom_labels, tuple(ng))
            mine = list(range(node_rank, len(dfpt.q_irr), node_size))
            for member, gk in unfolded_pao_vertices(
                dfpt, A, kpts_cryst, tuple(ng), ops, mine, basis, check_member=True, tol=unfold_tol
            ):
                lab = np.round(member.q_cryst * np.asarray(qg)).astype(int) % np.asarray(qg)
                if long_range is not None:
                    gk = gk - long_range.vertex_pao(member.q_cryst, nawf, orbital_positions)[None]
                g_ReRp[..., lab[0], lab[1], lab[2]] = pao_vertex_to_cells(gk, kidx, tuple(ng))
        elif unfold == 'full':
            from .elph_bloch import pao_vertex_to_cells

            for iq in range(node_rank, nq_coarse, node_size):
                member = member_of[tuple(labels[iq])]
                gk = dfpt.pao_vertex_k(member, A, tuple(ng), basis, bands)
                if long_range is not None:
                    gk = gk - long_range.vertex_pao(member.q_cryst, nawf, orbital_positions)[None]
                gR = pao_vertex_to_cells(gk, kidx, tuple(ng))
                g_ReRp[..., labels[iq, 0], labels[iq, 1], labels[iq, 2]] = gR
        else:
            raise ValueError("unfold must be 'full' or 'irreducible', got %r" % (unfold,))
        dfpt.clear_cache()
        node_comm.Barrier()
        for row in range(node_rank, nawf, node_size):
            g_ReRp[row] = np.fft.fftn(g_ReRp[row], axes=(-3, -2, -1)) / nq_coarse
    elif node_rank == 0:
        nq_coarse = qg[0] * qg[1] * qg[2]
        labels = np.round(q_cryst_coarse * np.asarray(qg)).astype(int) % np.asarray(qg)
        if q_cryst_coarse.shape[0] != nq_coarse or len({tuple(lab) for lab in labels}) != nq_coarse:
            raise ValueError(
                'the coarse q-points (%d) must tile the full %s grid; unfold the irreducible '
                'set first.' % (q_cryst_coarse.shape[0], 'x'.join(map(str, qg)))
            )
        g_ReRp[...] = 0.0
        for iq, q_cryst in enumerate(q_cryst_coarse):
            lr_q = None
            if long_range is not None:
                lr_q = long_range.vertex_pao(q_cryst, nawf, orbital_positions)
            if source == 'epw':
                gR = vertex_from_epw(
                    epw['epmatq'][..., iq], A, kpts_cryst, q_cryst, ng, epw['ibndkept'], lr_q
                )
            elif source == 'ahc':
                gR = vertex_from_qe_ahc(
                    coupling_dir, iq + 1, A, kpts_cryst, q_cryst, ng, nbnd, nmodes, nk, lr_q
                )
            else:
                path = '%s/elphmat.%d.dat' % (coupling_dir, iq + 1)
                gR, _q = vertex_from_qe_elphmat(path, A, kpts_cryst, bg, ng, long_range=lr_q)
            g_ReRp[..., labels[iq, 0], labels[iq, 1], labels[iq, 2]] = gR
        if epw is not None:
            del epw['epmatq']  # only the shared g(R_e, R_p) is needed from here on
        for row in range(nawf):
            g_ReRp[row] = np.fft.fftn(g_ReRp[row], axes=(-3, -2, -1)) / nq_coarse
    node_comm.Barrier()  # ensure the shared buffer is filled before any rank reads
    if cache_dir is not None and not cached:
        from .coupling_cache import store_dense_vertex

        if node_rank == 0 and comm.Get_rank() == 0:
            path = store_dense_vertex(
                cache_dir, cache_key, g_ReRp,
                {'coupling_dir': os.path.abspath(coupling_dir), 'source': source, 'qgrid': list(qgrid_coarse),
                 'ng': [int(n) for n in ng], 'masses_amu': [float(m) for m in masses_amu], 'tag': cache_tag},
            )  # fmt: skip
            print('  g(R_e, R_p) written to the cache %s' % path, flush=True)
        comm.Barrier()

    if orbital_positions is None:
        Nint_p, W_p, Midx_p = _ws_lattice(qgrid_coarse, at)
    else:
        if tau_cryst is None:
            raise ValueError('orbital_positions requires tau_cryst (vertex phonon WS).')
        Nint_p, W_p, Midx_p = vertex_phonon_ws_weights(
            qgrid_coarse, at, orbital_positions, tau_cryst
        )

    return {
        'g_ReRp': g_ReRp,
        'window': win,
        'phonon_ws': (Nint_p, W_p, Midx_p),
        'phonon_at_q': phonon_at_q,
        'epsi': small['epsi'],
        'zstar': small['zstar'],
    }


def _fermi_surface_accumulator(
    electrons: dict,
    rots: Sequence[NDArray],
    at: NDArray[np.float64],
    bg: NDArray[np.float64],
    nk_dense: int,
    fsthick_ev: float | None,
    fs_window: float,
    omega_max_thz: float,
    n_freq: int,
    isig: int,
) -> FermiSurfacePairCoupling:
    """Fermi-surface pair-coupling accumulator on the irreducible dense k-points.

    Parameters
    ----------
    electrons : dict
        Dense electron cache (:func:`precompute_dense_electrons`).
    rots : sequence of ndarray
        Crystal point group (integer crystal-axis rotations).
    at, bg : ndarray, shape ``(3, 3)``
        Direct and reciprocal lattice vectors (rows).
    nk_dense : int
        Dense k-grid size.
    fsthick_ev : float or None
        Fermi window (eV); ``None`` uses ``fs_window`` smearings.
    fs_window : float
        Fermi-surface shell of the isotropic coupling, in smearings.
    omega_max_thz : float
        Highest phonon frequency (THz).
    n_freq : int
        Number of phonon-frequency points.
    isig : int
        Smearing index.

    Returns
    -------
    FermiSurfacePairCoupling
        The empty accumulator.
    """
    _, representatives, star_sizes, star_index = irreducible_mesh(nk_dense, rots, at, bg)
    if fsthick_ev is None:
        fsthick_ev = fs_window * float(electrons['sigmas'][isig]) * RY_TO_EV
    return FermiSurfacePairCoupling(
        electrons, star_index, representatives, star_sizes, fsthick_ev,
        omega_max_thz / RY_TO_THZ * RY_TO_EV, n_freq, isig,
    )  # fmt: skip
