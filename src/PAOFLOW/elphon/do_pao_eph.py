"""Pseudo-atomic-orbital (Agapito-Bernardi) electron-phonon coupling driver.

This is the *interpolation* route of :mod:`PAOFLOW.elphon`: instead of
reconstructing the DFPT perturbation from ``dvscf`` (bare local + nonlocal +
induced), it reads Quantum ESPRESSO's **full** coarse-grid coupling
``el_ph_mat`` -- which already contains every contribution (including NLCC and
ultrasoft augmentation) that ``ph.x`` computes -- rotates it into the PAOFLOW
pseudo-atomic-orbital (PAO) gauge, and Wigner-Seitz interpolates the electrons
and the vertex to a dense grid to evaluate the isotropic Eliashberg properties.

Reference: L. A. Agapito and M. Bernardi, "Ab initio electron-phonon
interactions using atomic orbital wave functions", `Phys. Rev. B 97, 235146
(2018) <https://doi.org/10.1103/PhysRevB.97.235146>`_.

The heavy lifting reuses the existing, validated primitives:
:func:`~PAOFLOW.elphon.elph_bloch.vertex_pao_R` (PAO gauge + electron Fourier
transform), :func:`~PAOFLOW.elphon.elph_bloch.lambda_q_dense_ws` (Wigner-Seitz
dense interpolation + Fermi-surface double delta) and
:func:`~PAOFLOW.elphon.eph_kq.eliashberg_from_modes` (alpha^2F / lambda / Tc).
"""

import os

import numpy as np

from .elph_bloch import (
    AMU_RY,
    kq_index_map,
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
    vertex_pao_R,
)
from .eph_kq import eliashberg_from_modes
from .qe_elph_io import (
    el_ph_mat_to_cartesian,
    read_qe_ahc_gkk,
    read_qe_dyn,
    read_qe_el_ph_mat,
)


def _k_permutation(kpts_cryst, xk_cart, bg, ng):
    """Map each dumped k-point onto the PAOFLOW coarse-grid k index.

    Both lists sample the same uniform ``ng`` grid; they are matched by their
    integer grid labels ``round(k_cryst * ng) % ng`` (robust to Brillouin-zone
    folding and ordering differences).

    Returns
    -------
    ndarray ``(nksq,)`` int
        ``perm[i]`` is the PAOFLOW k index of dumped k-point ``i``.
    """
    ng = np.asarray(ng, int)
    lab = np.round(kpts_cryst * ng).astype(int) % ng
    table = {tuple(row): i for i, row in enumerate(lab)}
    xk_cryst = np.linalg.solve(bg.T, xk_cart.T).T
    labq = np.round(xk_cryst * ng).astype(int) % ng
    perm = np.empty(labq.shape[0], dtype=int)
    for i, row in enumerate(labq):
        key = tuple(row)
        if key not in table:
            raise ValueError('dumped k-point %s not on the %s grid' % (xk_cart[i], tuple(ng)))
        perm[i] = table[key]
    return perm


def vertex_from_qe_elphmat(elphmat_path, A, kpts_cryst, bg, ng):
    """PAO-gauge real-space vertex ``g(R_e)`` from one QE ``el_ph_mat`` dump.

    Parameters
    ----------
    elphmat_path : str
        Path to ``elphmat.<iq>.dat`` (patched-QE dump).
    A : ndarray ``(nbnd, nawf, nk)``
        PAO projection matrices ``A_{ni}(k)`` (``arry['U'][..., ispin]``), grabbed
        after ``projectability`` and before ``pao_hamiltonian``.
    kpts_cryst : ndarray ``(nk, 3)``
        PAOFLOW coarse-grid k-points (crystal coordinates).
    bg : ndarray ``(3, 3)``
        Reciprocal-lattice vectors (rows), for the ``xq``/``xk`` -> crystal map.
    ng : tuple(int, int, int)
        Coarse k-grid dimensions (== the ``pw.x`` SCF grid).

    Returns
    -------
    gR : ndarray ``(nawf, nawf, ncart, n1, n2, n3)``
    q_cryst : ndarray ``(3,)``
        The q-point of this dump in crystal coordinates.
    """
    qe = read_qe_el_ph_mat(elphmat_path)
    el_cart = el_ph_mat_to_cartesian(qe['el_ph_mat'], qe['u'])  # (m, n, ksq, c)
    nk = kpts_cryst.shape[0]
    nbnd, ncart = qe['nbnd'], 3 * qe['nat']
    perm = _k_permutation(kpts_cryst, qe['xk'], bg, ng)
    d = np.zeros((nk, nbnd, nbnd, ncart), dtype=complex)
    d[perm] = np.transpose(el_cart, (2, 0, 1, 3))  # (ksq, m, n, c) -> place on grid
    q_cryst = np.linalg.solve(bg.T, qe['xq'])
    ikq, _ = kq_index_map(kpts_cryst, q_cryst, ng)
    kidx = np.round(kpts_cryst * np.asarray(ng)).astype(int) % np.asarray(ng)
    gR = vertex_pao_R(d, A, ikq, kidx, ng)
    return gR, q_cryst


def vertex_from_qe_ahc(ahc_dir, iq, A, kpts_cryst, q_cryst, ng, nbnd, nmodes, nk):
    """PAO-gauge real-space vertex ``g(R_e)`` from one unpatched QE AHC dump.

    Like :func:`vertex_from_qe_elphmat`, but reads the Cartesian ``ahc_gkk``
    (:func:`~PAOFLOW.elphon.qe_elph_io.read_qe_ahc_gkk`): no pattern rotation and
    no k-permutation -- the AHC records are already in the nscf k-order, the same
    order as ``A`` and ``kpts_cryst`` for the same ``lead.save``.

    Parameters
    ----------
    ahc_dir : str
        Directory with the ``ahc_gkk_iq<iq>.bin`` files.
    iq : int
        1-based q index (matches ``<prefix>.dyn<iq>``).
    A : ndarray ``(nbnd, nawf, nk)``
        PAO projections.
    kpts_cryst : ndarray ``(nk, 3)``
        Coarse-grid k-points (crystal), nscf order.
    q_cryst : ndarray ``(3,)``
        q-point (crystal coordinates), from the ``.dyn`` file.
    ng : tuple(int, int, int)
        Coarse k-grid dimensions.
    nbnd, nmodes, nk : int
        AHC array dimensions.

    Returns
    -------
    ndarray ``(nawf, nawf, ncart, n1, n2, n3)``
    """
    ahc = read_qe_ahc_gkk(ahc_dir, iq, nbnd, nmodes, nk)
    d = np.transpose(ahc['el_cart'], (2, 0, 1, 3))  # (k, m, n, c), identity order
    ikq, _ = kq_index_map(kpts_cryst, q_cryst, ng)
    kidx = np.round(kpts_cryst * np.asarray(ng)).astype(int) % np.asarray(ng)
    return vertex_pao_R(d, A, ikq, kidx, ng)


def eliashberg_from_qe_coupling(
    A,
    HRs,
    kpts_cryst,
    bg,
    at,
    coupling_dir,
    q_weights,
    ng,
    dyn_paths,
    elphmat_fmt='elphmat.%d.dat',
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
    comm=None,
):
    """Isotropic Eliashberg properties from QE's coarse ``el_ph_mat`` (PAO route).

    For every irreducible q the coupling dump is rotated into the PAO gauge,
    Wigner-Seitz interpolated to a dense grid and reduced to ``lambda_{q nu}``;
    the modes are combined with their star weights and fed to the validated
    property engine.  The ``q = Gamma`` acoustic modes (``omega -> 0``) are
    zeroed, matching QE.

    Parameters
    ----------
    A : ndarray ``(nbnd, nawf, nk)``
        PAO projections (grab ``arry['U'][..., ispin]`` before ``pao_hamiltonian``).
    HRs : ndarray ``(nawf, nawf, m1, m2, m3, nspin)``
        PAO Hamiltonian (``E_F`` at 0), from ``pao_hamiltonian``.
    kpts_cryst : ndarray ``(nk, 3)``
        Coarse-grid k-points (crystal coordinates), e.g. ``read_nscf`` output.
    bg, at : ndarray ``(3, 3)``
        Reciprocal- and real-lattice vectors (rows).
    coupling_dir : str
        Directory holding the coupling files (``elphmat.<iq>.dat`` for
        ``source='elphmat'``, or ``ahc_gkk_iq<iq>.bin`` for ``source='ahc'``).
    q_weights : array_like ``(nq,)``
        Star sizes of the irreducible q-points (need not be normalised).  For a
        full un-reduced q-grid (e.g. the 27-point AHC output) use all ones.
    ng : tuple(int, int, int)
        Coarse coupling k-grid (== SCF grid).
    dyn_paths : sequence of str
        One QE ``*.dyn`` file per q (for frequencies/eigenvectors; also supplies
        the q-point for ``source='ahc'``).
    elphmat_fmt : str, optional
        Filename template for the patched dumps (``%d`` is the 1-based q index).
    source : {'elphmat', 'ahc'}, optional
        Coupling input: the patched ``el_ph_mat`` dump (any pseudo) or the
        unpatched QE AHC ``ahc_gkk`` binaries (norm-conserving only).
    masses_amu : array_like ``(natom,)``, optional
        Atomic masses (amu); required to mass-weight the phonon eigenvectors.
    nk_dense : int, optional
        Dense interpolation grid size.
    sigmas_ry : array_like, optional
        Fermi-surface smearing(s) (Ry); ``isig`` selects which one is reported.
    nelec : float, optional
        Valence electrons for the dense-grid Fermi-level recompute.
    mu_star : float, optional
        Coulomb pseudopotential for ``Tc``.
    ispin, isig : int, optional
    sigma_w_frac : float, optional
        Gaussian width (fraction of max frequency) for the ``alpha^2F`` histogram.
    fs_window : float, optional
        Fermi-surface window (in smearings) used to prune the dense-grid double
        delta: only k with a band within ``fs_window`` * ``sigma`` of E_F are kept.
        The default (8) is exact to ``exp(-fs_window^2)``; lower it (e.g. 4-5) to
        speed up metals with wide bands at a controlled tolerance.
    comm : mpi4py communicator, optional
        MPI communicator used to distribute the per-q loop (each q is an
        independent unit of work reusing the shared dense-electron cache).
        Defaults to ``MPI.COMM_WORLD``; run ``mpirun -np N python ...`` for an
        (up to) ``N``-fold speedup, capped at ``nq`` ranks.  The dense electron
        diagonalisation is computed redundantly on every rank (the cache is
        needed by all q and is too large to broadcast cheaply); only the
        per-q vertex interpolation is parallelised.

    Returns
    -------
    dict
        The :func:`~PAOFLOW.elphon.eph_kq.eliashberg_from_modes` result, with the
        extra keys ``'lambda_qv'``, ``'omega_qv_thz'`` (both ``(nq, nmode)``) and
        ``'dos_ef'`` (``(nq,)``, states/spin/Ry per q).
    """
    q_weights = np.asarray(q_weights, dtype=float)
    sigmas_ry = np.atleast_1d(np.asarray(sigmas_ry, dtype=float))
    nq = q_weights.size
    if masses_amu is None:
        raise ValueError('masses_amu is required to mass-weight the phonon eigenvectors')
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY  # (3*natom,)
    nbnd, nk = int(A.shape[0]), int(A.shape[2])
    nmodes = int(mass_flat_ry.size)

    # Diagonalise the dense electron spectrum ONCE; every q reuses it (E(k+q) /
    # V(k+q) are index shifts), so the per-q cost is only the vertex interpolation.
    electrons = precompute_dense_electrons(
        HRs, at, nk_dense, sigmas_ry, nelec, tuple(ng), ispin=ispin, fs_window=fs_window
    )

    # Distribute the embarrassingly parallel per-q loop across MPI ranks: each q
    # writes a disjoint slice of the (zero-initialised) output arrays, so a single
    # in-place SUM reduction reconstructs the full grid on every rank.
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    qstart, qstop = load_balancing(size, rank, nq)

    lam_qv = np.zeros((nq, nmodes), dtype=float)
    om_qv = np.zeros((nq, nmodes), dtype=float)
    dos_ef = np.zeros(nq, dtype=float)
    for iq in range(qstart, qstop):
        dyn = read_qe_dyn(dyn_paths[iq])
        if source == 'ahc':
            q_cryst = np.linalg.solve(bg.T, np.asarray(dyn['q'], dtype=float))
            gR = vertex_from_qe_ahc(
                coupling_dir, iq + 1, A, kpts_cryst, q_cryst, ng, nbnd, nmodes, nk
            )
        else:
            path = os.path.join(coupling_dir, elphmat_fmt % (iq + 1))
            gR, q_cryst = vertex_from_qe_elphmat(path, A, kpts_cryst, bg, ng)
        z = dyn['eigenvectors'].reshape(dyn['freq_thz'].size, -1)  # (nmode, 3*natom)
        zmass = z / np.sqrt(mass_flat_ry)[None, :]
        res = lambda_q_dense_ws_fast(gR, electrons, q_cryst, zmass, dyn['freq_thz'])
        lam = res['lambda_qnu'][isig].copy()
        if np.linalg.norm(q_cryst - np.round(q_cryst)) < 1.0e-6:
            lam[:] = 0.0  # zero the Gamma acoustic blow-up (QE convention)
        lam_qv[iq] = lam
        om_qv[iq] = np.abs(dyn['freq_thz'])
        dos_ef[iq] = res['dos_ef'][isig]

    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, lam_qv, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, om_qv, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, dos_ef, op=MPI.SUM)

    out = eliashberg_from_modes(
        lam_qv,
        om_qv,
        q_weights=q_weights,
        mu_star=mu_star,
        sigma_w_frac=sigma_w_frac,
    )
    out['lambda_qv'] = lam_qv
    out['omega_qv_thz'] = om_qv
    out['dos_ef'] = np.asarray(dos_ef)
    return out
