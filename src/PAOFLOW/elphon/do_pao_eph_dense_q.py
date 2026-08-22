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

import re

import numpy as np

from .do_pao_eph import vertex_from_qe_ahc, vertex_from_qe_elphmat
from .elph_bloch import (
    AMU_RY,
    RY_TO_THZ,
    _ws_lattice,
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
)
from .eph_kq import eliashberg_from_modes


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

    Returns
    -------
    gR : ndarray ``(nawf, nawf, ncart, n1e, n2e, n3e)``
        The half-vertex for this q -- the exact input shape expected by
        :func:`~PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws_fast`.
    """
    phase = W_p * np.exp(2j * np.pi * (np.asarray(q_cryst) @ Nint_p.T))  # (nws_p,)
    cells = g_ReRp[..., Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]]  # (..., nws_p)
    return np.tensordot(cells, phase, axes=([-1], [0]))  # (nawf, nawf, ncart, n1e,n2e,n3e)


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
        One ``*.dyn`` per FULL coarse-grid q (``len == prod(qgrid)``).
    qgrid : tuple(int, int, int)
        Coarse phonon q-grid.
    bg, at : ndarray ``(3, 3)``
        Reciprocal- and real-lattice vectors (rows).

    Returns
    -------
    callable
        ``phonon_at_q(q_cryst) -> (freq_thz (nmode,), z (nmode, ncart))``.
    """
    qgrid = tuple(int(n) for n in qgrid)
    nq = qgrid[0] * qgrid[1] * qgrid[2]
    if len(dyn_paths) != nq:
        raise ValueError(
            'phonon_interp_from_dyn needs one dyn file per FULL-grid q (%d), got %d.'
            % (nq, len(dyn_paths))
        )

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
    Nint_p, W_p, Midx_p = _ws_lattice(qgrid, at)

    def phonon_at_q(q_cryst):
        phase = W_p * np.exp(2j * np.pi * (np.asarray(q_cryst, dtype=float) @ Nint_p.T))
        cells = Cr[Midx_p[:, 0], Midx_p[:, 1], Midx_p[:, 2]]  # (nws_p, nmode, nmode)
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
    comm=None,
):
    """SKETCH: Eliashberg properties with BOTH k and q interpolated.

    Mirrors :func:`~PAOFLOW.elphon.do_pao_eph.eliashberg_from_qe_coupling`, but
    replaces the coarse-q loop by (a) one build of ``g(R_e, R_p)`` from the full
    coarse-q couplings and (b) a loop over a dense ``nq_dense^3`` q-grid, WS-
    interpolating both the vertex (:func:`g_Re_at_q`) and the phonons
    (``phonon_at_q``) to each dense q.

    Parameters
    ----------
    q_cryst_coarse : ndarray ``(nq_coarse, 3)``
        FULL coarse q-grid in crystal coordinates (``TODO: unfold_star`` if only
        the irreducible set is available).
    dyn_paths_full : sequence of str
        One ``*.dyn`` per full coarse q (used only for ``source='ahc'`` to supply
        the q-point of each dump).
    phonon_at_q : callable
        ``q_cryst -> (freq_thz, z)``; see :func:`_phonon_modes_at_q`.
    nq_dense : int, optional
        Dense q-grid size (defaults to ``nk_dense`` so k+q stays commensurate).
    min_freq_thz : float, optional
        Soft-mode guard: modes with ``freq_thz < min_freq_thz`` are dropped from
        ``lambda`` (default 0.0 -> only imaginary ``omega^2 < 0`` modes).  Coarse
        DFPT-grid Fourier interpolation can overshoot the acoustic branch into
        spurious soft/imaginary modes whose ``1/omega^2`` blows up ``lambda``;
        genuine near-Gamma acoustic modes contribute negligibly (their coupling
        vanishes), so raising this to a few kelvin (~0.05-0.1 THz) is safe.
    comm : mpi4py communicator, optional
        Distributes the dense-q loop across ranks (``MPI.COMM_WORLD`` by default);
        run ``mpirun -np N python ...`` for an up-to-``nq_dense^3``-fold speedup.
        The coarse ``g(R_e,R_p)`` build and the dense-electron cache are computed
        redundantly on every rank; only the per-q interpolation is parallelised.

    Notes
    -----
    Deferred: symmetry unfolding, q-phase validation and the polar Frohlich term.
    """
    if masses_amu is None:
        raise ValueError('masses_amu is required to mass-weight the phonon eigenvectors')
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY  # (ncart,)
    qgrid_coarse = tuple(int(n) for n in qgrid_coarse)
    q_cryst_coarse = np.asarray(q_cryst_coarse, dtype=float)
    nbnd, nk = int(A.shape[0]), int(A.shape[2])
    nmodes = int(mass_flat_ry.size)
    if nq_dense is None:
        nq_dense = nk_dense  # keep the dense q-grid commensurate with k for k+q

    # --- (a) coarse half-vertices g_q(R_e) on the FULL q-grid --------------- #
    g_list = []
    for iq, q_cryst in enumerate(q_cryst_coarse):
        if source == 'ahc':
            gR = vertex_from_qe_ahc(
                coupling_dir, iq + 1, A, kpts_cryst, q_cryst, ng, nbnd, nmodes, nk
            )
        else:
            path = '%s/elphmat.%d.dat' % (coupling_dir, iq + 1)
            gR, _q = vertex_from_qe_elphmat(path, A, kpts_cryst, bg, ng)
        g_list.append(gR)
    g_qRe = np.stack(g_list, axis=0)  # (nq_coarse, nawf, nawf, ncart, n1e,n2e,n3e)

    # --- build g(R_e, R_p) and the phonon WS lattice ----------------------- #
    g_ReRp = build_g_ReRp(g_qRe, q_cryst_coarse, qgrid_coarse)
    Nint_p, W_p, Midx_p = _ws_lattice(qgrid_coarse, at)

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
    )

    # --- (b) dense q-grid loop (distributed over MPI ranks) ---------------- #
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()

    ax = [np.arange(nq_dense) / nq_dense for _ in range(3)]
    qmesh = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    nqd = qmesh.shape[0]
    qstart, qstop = load_balancing(size, rank, nqd)
    lam_qv = np.zeros((nqd, nmodes))
    om_qv = np.zeros((nqd, nmodes))
    for iq in range(qstart, qstop):
        q_cryst = qmesh[iq]
        freq_thz, z = _phonon_modes_at_q(q_cryst, phonon_at_q)  # (nmode,), (nmode, ncart)
        zmass = z / np.sqrt(mass_flat_ry)[None, :]
        gR = g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p)
        res = lambda_q_dense_ws_fast(gR, electrons, q_cryst, zmass, freq_thz)
        lam = res['lambda_qnu'][isig].copy()
        lam[freq_thz < min_freq_thz] = 0.0  # drop spurious soft/imaginary modes
        if np.linalg.norm(q_cryst - np.round(q_cryst)) < 1.0e-6:
            lam[:] = 0.0  # zero the Gamma acoustic blow-up (QE convention)
        lam_qv[iq] = lam
        om_qv[iq] = np.abs(freq_thz)

    # Each rank wrote a disjoint slice; a single SUM reduction rebuilds the grid.
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, lam_qv, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, om_qv, op=MPI.SUM)

    # uniform weights: the dense q-grid is the full (unreduced) BZ sampling.
    out = eliashberg_from_modes(
        lam_qv, om_qv, q_weights=np.ones(nqd), mu_star=mu_star, sigma_w_frac=sigma_w_frac
    )
    out['lambda_qv'] = lam_qv
    out['omega_qv_thz'] = om_qv
    return out
