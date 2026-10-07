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
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .do_pao_eph import (
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
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
)
from .eph_kq import eliashberg_from_modes
from .fermi_surface_coupling import FermiSurfacePairCoupling

# Sign of the cell vector in the atom-pair separation of the force constants
# (see :func:`~PAOFLOW.elphon.elph_bloch._ws_lattice_pairs`).
PHONON_WS_SIGN = 1


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
    # Pair-resolved weights (nws_p, nawf, nawf, ncart): one phonon cell at a time.
    phase = np.exp(2j * np.pi * (np.asarray(q_cryst) @ Nint_p.T))
    vertex = np.zeros(g_ReRp.shape[:6], dtype=complex)
    for cell, (m1, m2, m3) in enumerate(Midx_p):
        vertex += g_ReRp[..., m1, m2, m3] * (phase[cell] * W_p[cell])[..., None, None, None]
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
def phonon_interp_from_dyn(dyn_paths, qgrid, bg, at):
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
    return _phonon_interp_from_force_constants(Cgrid, masses_ry, qgrid, at)


def phonon_interp_from_epw(
    dynq: NDArray[np.complex128],
    q_cryst: NDArray[np.float64],
    masses_amu: NDArray[np.float64],
    qgrid: tuple[int, int, int],
    at: NDArray[np.float64],
    atom_positions: NDArray[np.float64] | None = None,
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
    return _phonon_interp_from_force_constants(Cgrid, masses_ry, qgrid, at, atom_positions)


def _phonon_interp_from_force_constants(
    Cgrid: NDArray[np.complex128],
    masses_ry: NDArray[np.float64],
    qgrid: tuple[int, int, int],
    at: NDArray[np.float64],
    atom_positions: NDArray[np.float64] | None = None,
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

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz (nmode,), z (nmode, ncart))``.
    """
    nq = qgrid[0] * qgrid[1] * qgrid[2]
    # q -> R_p (same 1/N, sign convention as build_g_ReRp); grid axes stay leading.
    Cr = np.fft.fftn(Cgrid, axes=(0, 1, 2)) / nq  # (nq1, nq2, nq3, nmode, nmode)

    # Simple acoustic sum rule: force sum_R C_{a,b}(R) = 0 per Cartesian pair /
    # atom, so the acoustic modes go to zero at Gamma and interpolated branches
    # stay real (matdyn asr='simple').  For our monatomic/diagonal-mass grid the
    # correction lands on the R=0 self block.
    nat = masses_ry.size
    Csum = Cr.sum(axis=(0, 1, 2))  # (nmode, nmode) == C(Gamma)
    for a in range(3):
        for b in range(3):
            for na in range(nat):
                tot = sum(Csum[3 * na + a, 3 * nb + b] for nb in range(nat))
                Cr[0, 0, 0, 3 * na + a, 3 * na + b] -= tot

    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses_ry, 3))  # (nmode,)
    mass_weight = np.outer(inv_sqrt_m, inv_sqrt_m)  # D = C / sqrt(Ma Mb)
    if atom_positions is None:
        Nint_p, W_p, Midx_p = _ws_lattice(qgrid, at)
        cells = Cr[Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]]  # (nws_p, nmode, nmode)
    else:
        mode_positions = np.repeat(np.asarray(atom_positions, dtype=float), 3, axis=0)
        Nint_p, W_pairs, Midx_p = _ws_lattice_pairs(qgrid, at, mode_positions, sign=PHONON_WS_SIGN)
        cells = Cr[Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]] * W_pairs
        W_p = np.ones(Nint_p.shape[0])

    def phonon_at_q(q_cryst):
        phase = W_p * np.exp(2j * np.pi * (np.asarray(q_cryst, dtype=float) @ Nint_p.T))
        C = np.tensordot(phase, cells, axes=([0], [0]))  # (nmode, nmode)
        D = 0.5 * (C + C.conj().T) * mass_weight  # mass-weighted -> eig = omega^2 (Ry^2)
        w2, ev = np.linalg.eigh(D)
        freq_thz = np.sign(w2) * np.sqrt(np.abs(w2)) * RY_TO_THZ
        return freq_thz, ev.T  # (nmode,), (nmode, ncart)

    return phonon_at_q


def _read_dyn_matrices(path):
    """Parse the full-precision force-constant matrices from a QE ``*.dyn`` file.

    Reads every ``Dynamical Matrix in cartesian axes`` block (one per star-q) as
    the ``(3*nat, 3*nat)`` force-constant matrix ``C`` (atom-major layout
    ``3*na + alpha``) plus the atomic masses (QE Rydberg units).  ``eig(C)/M`` is
    ``omega^2``; mass-weight with ``1/sqrt(Ma Mb)`` to get the dynamical matrix.

    Returns
    -------
    dict
        ``{'q_cart': list of (3,), 'C': list of (3*nat, 3*nat), 'masses_ry': (nat,)}``.
    """
    lines = open(path).read().splitlines()
    hdr = lines[2].split()
    ntyp, nat = int(hdr[0]), int(hdr[1])
    type_mass = {}
    i = 3
    for _ in range(ntyp):
        parts = lines[i].split("'")
        idx = int(parts[0].split()[0])
        type_mass[idx] = float(parts[2].split()[0])
        i += 1
    masses_ry = np.empty(nat)
    for _ in range(nat):
        tok = lines[i].split()
        masses_ry[int(tok[0]) - 1] = type_mass[int(tok[1])]
        i += 1

    nmode = 3 * nat
    re_q = re.compile(r'q\s*=\s*\(\s*([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)')
    q_cart, mats = [], []
    k = 0
    while k < len(lines):
        if 'Dynamical' in lines[k] and 'cartesian' in lines[k]:
            j = k + 1
            while j < len(lines) and 'q =' not in lines[j]:
                j += 1
            mq = re_q.search(lines[j])
            q = np.array([_dyn_float(mq.group(t)) for t in (1, 2, 3)])
            C = np.zeros((nmode, nmode), dtype=complex)
            j += 1
            while j < len(lines):
                s = lines[j].split()
                if len(s) == 2 and all(t.isdigit() for t in s):
                    na, nb = int(s[0]) - 1, int(s[1]) - 1
                    for a in range(3):
                        v = [_dyn_float(x) for x in lines[j + 1 + a].split()]
                        for b in range(3):
                            C[3 * na + a, 3 * nb + b] = v[2 * b] + 1j * v[2 * b + 1]
                    j += 4
                elif 'Dynamical' in lines[j] or 'Diagonalizing' in lines[j]:
                    break
                else:
                    j += 1
            q_cart.append(q)
            mats.append(C)
            k = j
            continue
        k += 1
    return {'q_cart': q_cart, 'C': mats, 'masses_ry': masses_ry}


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
):
    """SKETCH: Eliashberg properties with BOTH k and q interpolated.

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
    source : {'elphmat', 'ahc', 'epw'}, optional
        Coarse coupling input (see
        :func:`~PAOFLOW.elphon.do_pao_eph.eliashberg_from_qe_coupling`).  Only
        ``'epw'`` gives vertices whose band phases match ``A`` for every q,
        which the ``q -> R_p`` interpolation requires.
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

    Notes
    -----
    Deferred: symmetry unfolding, q-phase validation and the polar Frohlich term.
    """
    if masses_amu is None:
        raise ValueError('masses_amu is required to mass-weight the phonon eigenvectors')
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY  # (ncart,)
    qgrid_coarse = tuple(int(n) for n in qgrid_coarse)
    nbnd, nk = int(A.shape[0]), int(A.shape[2])
    nmodes = int(mass_flat_ry.size)
    epw = None
    if source == 'epw':
        # EPW provides the full coarse q-grid (unfolded from the irreducible ph.x
        # q) and its force constants; both default from the .epb files.
        epw = load_epw_coupling(coupling_dir, nbnd, nk, masses_amu.size, bg)
        if q_cryst_coarse is None:
            q_cryst_coarse = epw['q_cryst']
        if phonon_at_q is None:
            atom_positions = None
            if orbital_positions is not None:
                if tau_cryst is None:
                    raise ValueError('orbital_positions requires tau_cryst (atom-pair phonon WS).')
                atom_positions = tau_cryst
            phonon_at_q = phonon_interp_from_epw(
                epw['dynq'], epw['q_cryst'], masses_amu, qgrid_coarse, at, atom_positions
            )
    q_cryst_coarse = np.asarray(q_cryst_coarse, dtype=float)
    if nq_dense is None:
        nq_dense = nk_dense  # keep the dense q-grid commensurate with k for k+q

    # --- MPI setup (node-local sharing of the large read-only vertex) ------ #
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    # g(R_e, R_p) is large (~GBs) and read-only, and every rank needs all of it.
    # Allocate ONE copy per node in MPI shared memory (built by the node-local
    # rank 0 only) instead of an independent copy per rank, which otherwise
    # multiplies the memory by the number of ranks on the node.
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank = node_comm.Get_rank()
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

    # --- (a) coarse half-vertices g_q(R_e) -> g(R_e, R_p), on node rank 0 --- #
    if node_rank == 0:
        g_list = []
        for iq, q_cryst in enumerate(q_cryst_coarse):
            if source == 'epw':
                gR = vertex_from_epw(
                    epw['epmatq'][..., iq], A, kpts_cryst, q_cryst, ng, epw['ibndkept']
                )
            elif source == 'ahc':
                gR = vertex_from_qe_ahc(
                    coupling_dir, iq + 1, A, kpts_cryst, q_cryst, ng, nbnd, nmodes, nk
                )
            else:
                path = '%s/elphmat.%d.dat' % (coupling_dir, iq + 1)
                gR, _q = vertex_from_qe_elphmat(path, A, kpts_cryst, bg, ng)
            g_list.append(gR)
        g_ReRp[...] = build_g_ReRp(np.stack(g_list, axis=0), q_cryst_coarse, qgrid_coarse)
        del g_list
    if epw is not None:
        del epw['epmatq']  # only the shared g(R_e, R_p) is needed from here on
    node_comm.Barrier()  # ensure the shared buffer is filled before any rank reads

    if orbital_positions is None:
        Nint_p, W_p, Midx_p = _ws_lattice(qgrid_coarse, at)
    else:
        if tau_cryst is None:
            raise ValueError('orbital_positions requires tau_cryst (vertex phonon WS).')
        Nint_p, W_p, Midx_p = vertex_phonon_ws_weights(
            qgrid_coarse, at, orbital_positions, tau_cryst
        )

    # --- electron cache (dense k), shared by every dense q ----------------- #
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
    )

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
        res = lambda_q_dense_ws_fast(
            gR, electrons, q_cryst, zmass, freq_thz,
            pair_coupling=pair_coupling, pair_weights=pair_weights, pair_q_weight=q_weight,
        )  # fmt: skip
        lam = res['lambda_qnu'][isig].copy()
        lam[freq_thz < min_freq_thz] = 0.0  # drop spurious soft/imaginary modes
        if is_gamma:
            lam[:] = 0.0  # zero the Gamma acoustic blow-up (QE convention)
        lam_qv[iq] = lam
        om_qv[iq] = np.abs(freq_thz)

    # Each rank wrote a disjoint slice; a single SUM reduction rebuilds the grid.
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, lam_qv, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, om_qv, op=MPI.SUM)

    win.Free()  # release the shared g(R_e, R_p) window (all reads are done)

    # star-multiplicity weights (unit weights when no symmetry folding).
    out = eliashberg_from_modes(
        lam_qv, om_qv, q_weights=qweights, mu_star=mu_star, sigma_w_frac=sigma_w_frac
    )
    out['lambda_qv'] = lam_qv
    out['omega_qv_thz'] = om_qv
    if pair_coupling is not None:
        pair_coupling.reduce(comm)
        out['fs_coupling'] = pair_coupling.result(bg, at, nq_dense)
    return out


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
