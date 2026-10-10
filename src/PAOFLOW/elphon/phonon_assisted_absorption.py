"""Phonon-assisted (indirect) optical absorption from the PAO electron-phonon route.

Second-order (photon + phonon) absorption of an indirect-gap semiconductor,
following EPW's ``lindabs`` (``indabs.f90``; Noffsinger et al., Phys. Rev. Lett.
108, 167402 (2012)), with every ingredient interpolated in the PAO basis:

* the electron-phonon vertex ``g(R_e, R_p)`` built from EPW's coarse Bloch
  couplings (:func:`~PAOFLOW.elphon.do_pao_eph_dense_q.prepare_dense_vertex`);
* the electrons and the velocity matrix elements on the dense k-grid
  (:func:`~PAOFLOW.elphon.elph_bloch.precompute_dense_electrons`,
  :func:`~PAOFLOW.elphon.elph_bloch.band_velocities`), in the same gauge as the
  band-basis coupling (:func:`~PAOFLOW.elphon.elph_bloch.band_vertex`), so the two
  second-order paths interfere correctly;
* the phonons on the dense q-grid (EPW force constants).

The direct (vertical) absorption of EPW's ``dirabs`` is computed alongside, on the
same velocities.  All quantities are in Rydberg atomic units internally; the
public driver takes and returns electron-volts.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

from .do_pao_eph_dense_q import (
    _crystal_point_group,
    _phonon_modes_at_q,
    g_Re_at_q,
    irreducible_qmesh,
    prepare_dense_vertex,
)
from .elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    band_velocities,
    band_vertex,
    precompute_dense_electrons,
    vertex_ws_coefficients,
)

# Denominator broadenings of the second-order amplitudes (EPW ``indabs.f90``), eV.
EPW_ETAS_EV = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
KELVIN_TO_EV = 8.617333262e-5
EV_TO_CM1 = 8065.543937
HBAR_C_EV_CM = 1.973269804e-5
DELTA_CUTOFF = 6.0  # energy-conservation window in smearings (EPW)


def fermi_occupation(energy: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """Fermi-Dirac occupation ``1 / (1 + exp(e / T))`` (EPW ``wgauss(-e/T, -99)``).

    Parameters
    ----------
    energy : ndarray
        Energies relative to the Fermi level (same units as ``temperature``).
    temperature : float
        ``k_B T``; ``0`` gives a step function.

    Returns
    -------
    ndarray
        Occupations, same shape as ``energy``.
    """
    if temperature <= 0.0:
        return (energy < 0.0).astype(float) + 0.5 * (energy == 0.0)
    return expit(-np.asarray(energy) / temperature)


def bose_occupation(energy: NDArray[np.float64], temperature: float) -> NDArray[np.float64]:
    """Bose-Einstein occupation ``1 / (exp(w / T) - 1)`` for positive ``energy``.

    Parameters
    ----------
    energy : ndarray
        Phonon energies (same units as ``temperature``), positive.
    temperature : float
        ``k_B T``; ``0`` gives zero occupation.

    Returns
    -------
    ndarray
        Occupations, same shape as ``energy``.
    """
    if temperature <= 0.0:
        return np.zeros_like(np.asarray(energy, dtype=float))
    return 1.0 / np.expm1(np.asarray(energy) / temperature)


def gaussian_delta(x: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Normalised Gaussian ``exp(-(x/sigma)^2) / (sigma sqrt(pi))`` (EPW ``w0gauss``)."""
    return np.exp(-((x / sigma) ** 2)) / (sigma * np.sqrt(np.pi))


def lorentzian_delta(x: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """Normalised Lorentzian ``(sigma / pi) / (sigma^2 + x^2)``."""
    return sigma / (sigma**2 + x**2) / np.pi


def indirect_absorption_kernel(
    energy_k: NDArray[np.float64],
    energy_kq: NDArray[np.float64],
    velocity_k: NDArray[np.complex128],
    velocity_kq: NDArray[np.complex128],
    coupling: NDArray[np.complex128],
    phonon_energy: NDArray[np.float64],
    temperatures: Sequence[float],
    photon_energy: NDArray[np.float64],
    etas: Sequence[float],
    degauss: float,
    fsthick: float,
    eps_acoustic: float,
    weight: float = 1.0,
    spin_degeneracy: float = 2.0,
    occupation_tol: float = 1.0e-12,
    pair_block: int = 2048,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Phonon-assisted ``Im eps(omega)`` from one block of ``(k, k+q)`` pairs at one q.

    Parameters
    ----------
    energy_k, energy_kq : ndarray, shape ``(nk, nband)``
        Band energies at ``k`` and ``k+q`` relative to the Fermi level (Ry).
    velocity_k, velocity_kq : ndarray, shape ``(nk, 3, nband, nband)``, complex
        Velocity matrix elements ``<m|v_alpha|n>`` at ``k`` and ``k+q`` (Ry bohr),
        in the gauge of ``coupling``.
    coupling : ndarray, shape ``(nk, nmode, nband, nband)``, complex
        ``g_{mn,nu} = <m, k+q|dV_{q nu}|n, k>`` without :math:`1/\\sqrt{2\\omega}`
        (Rydberg units; :func:`~PAOFLOW.elphon.elph_bloch.band_vertex`).
    phonon_energy : ndarray, shape ``(nmode,)``
        Phonon energies at q (Ry); modes below ``eps_acoustic`` are skipped.
    temperatures : sequence of float
        ``k_B T`` values (Ry).
    photon_energy : ndarray, shape ``(nomega,)``
        Photon energies (Ry), positive.
    etas : sequence of float
        Imaginary broadenings of the intermediate-state denominators (Ry).
    degauss : float
        Smearing of the energy-conserving delta (Ry).
    fsthick : float
        Only states with ``|e| < fsthick`` enter as initial or final states (Ry).
    eps_acoustic : float
        Phonon-energy threshold (Ry).
    weight : float, optional
        Multiplies the block sum, e.g. ``w_q / (N_k Omega)``.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling (EPW's ``16 pi^2`` prefactor), 1 with it.
    occupation_tol : float, optional
        Pairs whose occupation factors ``f_i (1 - f_j) + (1 - f_i) f_j`` stay below
        this at every temperature are skipped (``0`` keeps all, as EPW).
    pair_block : int, optional
        Number of ``(k, i, j)`` pairs processed at once (memory control).

    Returns
    -------
    eps2_gaussian, eps2_lorentzian : ndarray, shape ``(ntemp, neta, 3, nomega)``
        Diagonal components of ``Im eps`` with a Gaussian and a Lorentzian delta.

    Notes
    -----
    For an initial state :math:`i` at :math:`\\mathbf{k}`, a final state
    :math:`j` at :math:`\\mathbf{k}+\\mathbf{q}` and a phonon :math:`\\nu`, the two
    time orderings (photon first, phonon first) give the amplitudes

    .. math::

        S^{a/e}_{1,\\alpha} = \\sum_m
            \\frac{g_{jm,\\nu}\\, v^{\\alpha}_{mi}(\\mathbf{k})}
                 {\\varepsilon_{m\\mathbf{k}} - \\varepsilon_{j\\mathbf{k}+\\mathbf{q}}
                  \\pm \\omega_{\\mathbf{q}\\nu} + i\\eta},
        \\qquad
        S^{a/e}_{2,\\alpha} = \\sum_m
            \\frac{v^{\\alpha}_{jm}(\\mathbf{k}+\\mathbf{q})\\, g_{mi,\\nu}}
                 {\\varepsilon_{m\\mathbf{k}+\\mathbf{q}} - \\varepsilon_{i\\mathbf{k}}
                  \\mp \\omega_{\\mathbf{q}\\nu} + i\\eta},

    for phonon absorption (:math:`a`, upper signs) and emission (:math:`e`), and

    .. math::

        \\mathrm{Im}\\,\\varepsilon_{\\alpha\\alpha}(\\omega) \\mathrel{+}=
            w\\, \\frac{8\\pi^2 g_s}{\\omega^2} \\sum_{ij\\nu} P^{a/e}\\,
            \\frac{|S^{a/e}_{1,\\alpha} + S^{a/e}_{2,\\alpha}|^2}{2\\omega_{\\mathbf{q}\\nu}}\\,
            \\delta(\\varepsilon_{j\\mathbf{k}+\\mathbf{q}} - \\varepsilon_{i\\mathbf{k}}
                   - \\omega \\mp \\omega_{\\mathbf{q}\\nu}),

    with :math:`P^a = n f_i (1 - f_j) - (n + 1)(1 - f_i) f_j` and
    :math:`P^e = (n + 1) f_i (1 - f_j) - n (1 - f_i) f_j` (:math:`n` Bose,
    :math:`f` Fermi occupations).  :math:`8\\pi^2 g_s = 8\\pi^2 e^2` in Rydberg
    units for :math:`g_s = 2`, EPW's ``cfac``.  As in EPW, a ``(pair, mode,
    omega)`` term is kept when either delta lies within six smearings, and the
    intermediate states :math:`m` run over the bands of the window.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    etas = np.atleast_1d(np.asarray(etas, dtype=float))
    photon_energy = np.atleast_1d(np.asarray(photon_energy, dtype=float))
    ntemp, neta, nomega = temperatures.size, etas.size, photon_energy.size
    eps2_gaussian = np.zeros((ntemp, neta, 3, nomega))
    eps2_lorentzian = np.zeros((ntemp, neta, 3, nomega))

    active_modes = np.nonzero(phonon_energy > eps_acoustic)[0]
    if active_modes.size == 0:
        return eps2_gaussian, eps2_lorentzian
    omega_ph = np.asarray(phonon_energy, dtype=float)[active_modes]
    coupling = coupling[:, active_modes]
    omega_ph_max = float(np.max(phonon_energy))
    cutoff = DELTA_CUTOFF * degauss

    occ_k = np.stack([fermi_occupation(energy_k, t) for t in temperatures])  # (nT, nk, nb)
    occ_kq = np.stack([fermi_occupation(energy_kq, t) for t in temperatures])
    bose = np.stack([bose_occupation(omega_ph, t) for t in temperatures])  # (nT, nmode)

    transition = energy_kq[:, None, :] - energy_k[:, :, None]  # (nk, i, j)
    candidate = (np.abs(energy_k)[:, :, None] < fsthick) & (np.abs(energy_kq)[:, None, :] < fsthick)
    candidate &= transition < photon_energy.max() + omega_ph_max + cutoff
    candidate &= transition > photon_energy.min() - omega_ph_max - cutoff
    if occupation_tol > 0.0:
        relevance = (
            occ_k[:, :, :, None] * (1.0 - occ_kq[:, :, None, :])
            + (1.0 - occ_k[:, :, :, None]) * occ_kq[:, :, None, :]
        )
        candidate &= relevance.max(axis=0) > occupation_tol
    k_idx, i_idx, j_idx = np.nonzero(candidate)

    eta_imag = 1j * etas[None, None, :, None]
    for start in range(0, k_idx.size, pair_block):
        kp = k_idx[start : start + pair_block]
        ip = i_idx[start : start + pair_block]
        jp = j_idx[start : start + pair_block]
        # path 1: photon at k (i -> m), then phonon (m, k -> j, k+q)
        g_jm = coupling[kp, :, jp, :]  # (P, nmode, m)
        v_mi = velocity_k[kp, :, :, ip]  # (P, 3, m)
        gap_1 = energy_k[kp, :] - energy_kq[kp, jp][:, None]  # (P, m)
        # path 2: phonon first (i, k -> m, k+q), then photon at k+q (m -> j)
        g_mi = coupling[kp, :, :, ip]  # (P, nmode, m)
        v_jm = velocity_kq[kp, :, jp, :]  # (P, 3, m)
        gap_2 = energy_kq[kp, :] - energy_k[kp, ip][:, None]

        numerator_1 = g_jm[:, :, None, :] * v_mi[:, None, :, :]  # (P, nmode, 3, m)
        numerator_2 = g_mi[:, :, None, :] * v_jm[:, None, :, :]
        w_ph = omega_ph[None, :, None, None]

        def amplitude(numerator, gap, sign):
            denominator = gap[:, None, None, :] + sign * w_ph + eta_imag  # (P, nmode, neta, m)
            return np.einsum('pvam,pvem->pvea', numerator, 1.0 / denominator, optimize=True)

        s_abs = amplitude(numerator_1, gap_1, +1.0) + amplitude(numerator_2, gap_2, -1.0)
        s_emi = amplitude(numerator_1, gap_1, -1.0) + amplitude(numerator_2, gap_2, +1.0)
        s_abs2 = np.abs(s_abs) ** 2  # (P, nmode, neta, 3)
        s_emi2 = np.abs(s_emi) ** 2

        delta_e = transition[kp, ip, jp]  # (P,)
        x_abs = delta_e[:, None, None] - photon_energy[None, None, :] - omega_ph[None, :, None]
        x_emi = delta_e[:, None, None] - photon_energy[None, None, :] + omega_ph[None, :, None]
        keep = (np.abs(x_abs) <= cutoff) | (np.abs(x_emi) <= cutoff)  # (P, nmode, nomega)
        if not keep.any():
            continue
        gauss_abs = gaussian_delta(x_abs, degauss) * keep
        gauss_emi = gaussian_delta(x_emi, degauss) * keep
        lorentz_abs = lorentzian_delta(x_abs, degauss) * keep
        lorentz_emi = lorentzian_delta(x_emi, degauss) * keep

        f_i = occ_k[:, kp, ip]  # (nT, P)
        f_j = occ_kq[:, kp, jp]
        forward = (f_i * (1.0 - f_j))[:, :, None]  # (nT, P, 1)
        backward = ((1.0 - f_i) * f_j)[:, :, None]
        n_ph = bose[:, None, :]  # (nT, 1, nmode)
        factor_abs = (n_ph * forward - (n_ph + 1.0) * backward) / (2.0 * omega_ph)
        factor_emi = ((n_ph + 1.0) * forward - n_ph * backward) / (2.0 * omega_ph)

        for eps2, d_abs, d_emi in (
            (eps2_gaussian, gauss_abs, gauss_emi),
            (eps2_lorentzian, lorentz_abs, lorentz_emi),
        ):
            eps2 += np.einsum('tpv,pvw,pvea->teaw', factor_abs, d_abs, s_abs2, optimize=True)
            eps2 += np.einsum('tpv,pvw,pvea->teaw', factor_emi, d_emi, s_emi2, optimize=True)

    prefactor = weight * 8.0 * np.pi**2 * spin_degeneracy / photon_energy**2
    return eps2_gaussian * prefactor, eps2_lorentzian * prefactor


def direct_absorption_kernel(
    energy_k: NDArray[np.float64],
    velocity_k: NDArray[np.complex128],
    temperatures: Sequence[float],
    photon_energy: NDArray[np.float64],
    degauss: float,
    fsthick: float,
    weight: float = 1.0,
    spin_degeneracy: float = 2.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Direct (vertical) ``Im eps(omega)`` from one block of k-points (EPW ``dirabs``).

    Parameters
    ----------
    energy_k : ndarray, shape ``(nk, nband)``
        Band energies relative to the Fermi level (Ry).
    velocity_k : ndarray, shape ``(nk, 3, nband, nband)``, complex
        Velocity matrix elements (Ry bohr).
    temperatures : sequence of float
        ``k_B T`` values (Ry).
    photon_energy : ndarray, shape ``(nomega,)``
        Photon energies (Ry).
    degauss : float
        Smearing of the energy-conserving delta (Ry).
    fsthick : float
        Energy window around the Fermi level (Ry).
    weight : float, optional
        Multiplies the block sum, e.g. ``1 / (N_k Omega)``.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.

    Returns
    -------
    eps2_gaussian, eps2_lorentzian : ndarray, shape ``(ntemp, 3, nomega)``

    Notes
    -----
    .. math::

        \\mathrm{Im}\\,\\varepsilon_{\\alpha\\alpha}(\\omega) \\mathrel{+}=
            w\\, \\frac{8\\pi^2 g_s}{\\omega^2} \\sum_{ij} (f_i - f_j)\\,
            |v^{\\alpha}_{ji}|^2\\, \\delta(\\varepsilon_j - \\varepsilon_i - \\omega),

    keeping a term when the delta lies within six smearings, as EPW.
    """
    temperatures = np.atleast_1d(np.asarray(temperatures, dtype=float))
    photon_energy = np.atleast_1d(np.asarray(photon_energy, dtype=float))
    cutoff = DELTA_CUTOFF * degauss
    transition = energy_k[:, None, :] - energy_k[:, :, None]  # (nk, i, j)
    candidate = (np.abs(energy_k)[:, :, None] < fsthick) & (np.abs(energy_k)[:, None, :] < fsthick)
    candidate &= (transition > photon_energy.min() - cutoff) & (
        transition < photon_energy.max() + cutoff
    )
    k_idx, i_idx, j_idx = np.nonzero(candidate)
    delta_e = transition[k_idx, i_idx, j_idx]
    v2 = np.abs(velocity_k[k_idx, :, j_idx, i_idx]) ** 2  # (P, 3)
    x = delta_e[:, None] - photon_energy[None, :]
    keep = np.abs(x) <= cutoff
    occ = np.stack([fermi_occupation(energy_k, t) for t in temperatures])  # (nT, nk, nb)
    pfac = occ[:, k_idx, i_idx] - occ[:, k_idx, j_idx]  # (nT, P)
    gauss = np.einsum('tp,pa,pw->taw', pfac, v2, gaussian_delta(x, degauss) * keep)
    lorentz = np.einsum('tp,pa,pw->taw', pfac, v2, lorentzian_delta(x, degauss) * keep)
    prefactor = weight * 8.0 * np.pi**2 * spin_degeneracy / photon_energy**2
    return gauss * prefactor, lorentz * prefactor


def absorption_coefficient(
    photon_energy_ev: NDArray[np.float64],
    eps2: NDArray[np.float64],
    refractive_index: float | NDArray[np.float64] = 3.4,
) -> NDArray[np.float64]:
    """Absorption coefficient ``alpha = omega Im eps / (n_r c)`` in cm^-1.

    Parameters
    ----------
    photon_energy_ev : ndarray, shape ``(nomega,)``
        Photon energies (eV).
    eps2 : ndarray, shape ``(..., nomega)``
        Imaginary part of the dielectric function.
    refractive_index : float or ndarray, shape ``(nomega,)``, optional
        Real refractive index (default 3.4, the constant of the EPW tutorial for Si).

    Returns
    -------
    ndarray, shape ``(..., nomega)``
        Absorption coefficient (cm^-1).
    """
    return np.asarray(photon_energy_ev) * eps2 / (HBAR_C_EV_CM * np.asarray(refractive_index))


def nonlocal_velocity_operator(
    data_controller,
    bg: NDArray[np.float64],
    alat: float,
    sign: int = +1,
    q_max: float = 15.0,
    n_q: int = 300,
    pao_tol: float = 1.0e-3,
) -> Callable[[NDArray[np.float64]], NDArray[np.complex128]]:
    """Non-local pseudopotential velocity correction at arbitrary k (PAO basis).

    Parameters
    ----------
    data_controller : DataController
        PAOFLOW data after ``projections`` (PAO basis, species and UPF files).
    bg : ndarray, shape ``(3, 3)``
        Reciprocal lattice vectors (rows, 2 pi/alat).
    alat : float
        Lattice parameter (bohr).
    sign : {1, -1}, optional
        Injection sign of :func:`~PAOFLOW.hamiltonian.nonlocal_velocity.inject_into_dHksp`
        (``+1``, the scalar-relativistic calibration against ``epsilon.x``).
    q_max, n_q, pao_tol : float, int, float, optional
        Radial-table parameters of :meth:`PAOFLOW.PAOFLOW.nonlocal_velocity_correction`.

    Returns
    -------
    callable
        ``K_cryst (nk, 3) -> (nk, 3, nawf, nawf)``: the correction to the velocity
        operator of :func:`~PAOFLOW.elphon.elph_bloch.band_velocities` (Ry bohr).

    Raises
    ------
    NotImplementedError
        For fully-relativistic runs (the scalar operator only).
    ValueError
        From the loaders, for ultrasoft or PAW pseudopotentials (norm-conserving only).

    Notes
    -----
    PAOFLOW's ``dHksp`` is minus the velocity operator and the correction enters
    it as ``dHksp += sign * Delta p``, so the velocity gains ``-sign * Delta p``.
    k-points are folded to ``[-0.5, 0.5)`` as on PAOFLOW's FFT grid.
    """
    from ..hamiltonian.nonlocal_velocity import (
        build_nl_real_space_tables,
        build_nonlocal_velocity_kspace,
        enumerate_nl_pairs,
        load_beta_projectors,
        load_pao_orbitals,
    )

    arrays, attributes = data_controller.data_dicts()
    if bool(attributes.get('dftSO', False)):
        raise NotImplementedError(
            'nonlocal_velocity_operator supports scalar-relativistic runs only.'
        )
    if sign not in (-1, 1):
        raise ValueError(f'sign must be -1 or +1, got {sign!r}')
    beta_catalog = load_beta_projectors(data_controller)
    pao_catalog = load_pao_orbitals(data_controller)
    a_cart = np.asarray(arrays['a_vectors']) * float(alat)
    pairs = enumerate_nl_pairs(beta_catalog, pao_catalog, a_cart, pao_tol=pao_tol)
    tables = build_nl_real_space_tables(beta_catalog, pao_catalog, pairs, q_max=q_max, n_q=n_q)
    bg = np.asarray(bg, dtype=float)

    def correction(k_cryst: NDArray[np.float64]) -> NDArray[np.complex128]:
        folded = k_cryst - np.floor(k_cryst + 0.5)
        k_cart = (folded @ bg) * (2.0 * np.pi / float(alat))
        delta_p = build_nonlocal_velocity_kspace(
            beta_catalog, pao_catalog, tables, k_cart, units='rydberg'
        )
        return -sign * delta_p

    return correction


def _kq_flat_index(nk_dense: int, shift: NDArray[np.int_]) -> NDArray[np.int_]:
    """Flat dense-grid index of ``k + q`` for every k (``q = shift / nk_dense``)."""
    grid = np.stack(np.meshgrid(*[np.arange(nk_dense)] * 3, indexing='ij'), axis=-1).reshape(-1, 3)
    shifted = (grid + np.asarray(shift)[None, :]) % nk_dense
    return np.ravel_multi_index(shifted.T, (nk_dense,) * 3)


def _shared_velocities(
    electrons: dict,
    bands: NDArray[np.int_],
    nl_correction: Callable | None,
    comm,
    kblock: int,
):
    """Band velocities on the whole dense grid, one copy per node (MPI shared memory)."""
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    node_rank, node_size = node_comm.Get_rank(), node_comm.Get_size()
    nkd, nband = electrons['K'].shape[0], bands.size
    shape = (nkd, 3, nband, nband)
    itemsize = np.dtype(np.complex128).itemsize
    nbytes = int(np.prod(shape)) * itemsize if node_rank == 0 else 0
    window = MPI.Win.Allocate_shared(nbytes, itemsize, comm=node_comm)
    buffer, _ = window.Shared_query(0)
    velocities = np.ndarray(buffer=buffer, dtype=np.complex128, shape=shape)
    start, stop = load_balancing(node_size, node_rank, nkd)
    for s0 in range(start, stop, kblock):
        block = np.arange(s0, min(s0 + kblock, stop))
        velocities[block] = band_velocities(electrons, block, bands, nl_correction)
    node_comm.Barrier()
    return velocities, window


def phonon_assisted_absorption_dense_q(
    A: NDArray[np.complex128],
    HRs: NDArray[np.complex128],
    kpts_cryst: NDArray[np.float64],
    bg: NDArray[np.float64],
    at: NDArray[np.float64],
    alat: float,
    cell_volume: float,
    coupling_dir: str,
    qgrid_coarse: Sequence[int],
    ng: Sequence[int],
    *,
    masses_amu: NDArray[np.float64],
    nelec: float,
    nk_dense: int = 12,
    nq_dense: int = 6,
    omega_ev: tuple[float, float, float] = (0.05, 3.0, 0.05),
    temps_k: Sequence[float] = (300.0,),
    degauss_ev: float = 0.05,
    fsthick_ev: float = 4.0,
    fermi_energy_ev: float | None = None,
    etas_ev: Sequence[float] = EPW_ETAS_EV,
    eps_acoustic_cm: float = 0.1,
    refractive_index: float | NDArray[np.float64] = 3.4,
    nonlocal_velocity=None,
    spin_degeneracy: float = 2.0,
    sym_rots: NDArray[np.int_] | None = None,
    tau_cryst: NDArray[np.float64] | None = None,
    species: Sequence[str] | None = None,
    orbital_positions: NDArray[np.float64] | None = None,
    ispin: int = 0,
    kblock: int = 512,
    comm=None,
    allow_missing_long_range: bool = False,
) -> dict:
    """Phonon-assisted and direct optical absorption with dense k and q interpolation.

    Parameters
    ----------
    A : ndarray, shape ``(nbnd, nawf, nk)``, complex
        PAO projections on the coarse (EPW nscf) k-grid.
    HRs : ndarray, shape ``(nawf, nawf, n1, n2, n3, nspin)``, complex
        PAO Hamiltonian (eV).
    kpts_cryst : ndarray, shape ``(nk, 3)``
        Coarse k-points (crystal coordinates).
    bg, at : ndarray, shape ``(3, 3)``
        Reciprocal (rows, 2 pi/alat) and direct (rows, alat) lattice vectors.
    alat : float
        Lattice parameter (bohr).
    cell_volume : float
        Unit-cell volume (bohr^3).
    coupling_dir : str
        EPW directory with the ``.epb`` files.
    qgrid_coarse : sequence of int
        Coarse (ph.x) q-grid.
    ng : sequence of int
        Coarse k-grid.
    masses_amu : ndarray, shape ``(nat,)``
        Atomic masses (amu), one per atom.
    nelec : float
        Valence electrons (locates the gap for the mid-gap Fermi level).
    nk_dense, nq_dense : int, optional
        Dense k- and q-grids per axis; ``nk_dense`` must be a multiple of
        ``nq_dense`` (EPW tutorial: 12 and 6).
    omega_ev : tuple(float, float, float), optional
        Photon energies ``(min, max, step)`` (eV), EPW ``omegamin/omegamax/omegastep``.
    temps_k : sequence of float, optional
        Temperatures (K).
    degauss_ev : float, optional
        Smearing of the energy-conserving delta (eV), EPW ``degaussw``.
    fsthick_ev : float, optional
        Window around the Fermi level for initial, final and intermediate states
        (eV), EPW ``fsthick``.
    fermi_energy_ev : float, optional
        Fermi level on the energy scale of ``HRs`` (eV); ``None`` puts it mid-gap.
    etas_ev : sequence of float, optional
        Intermediate-state broadenings (eV); EPW's nine values by default.
    eps_acoustic_cm : float, optional
        Phonons below this energy (cm^-1) are skipped (EPW ``eps_acoustic``).
    refractive_index : float or ndarray, optional
        Real refractive index for the absorption coefficient (default 3.4).
    nonlocal_velocity : DataController, optional
        When given, adds the norm-conserving non-local pseudopotential velocity
        correction (:func:`nonlocal_velocity_operator`) built from it.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.
    sym_rots, tau_cryst, species : optional
        Point-group rotations (``read_nscf`` ``'s_cryst'``), atomic positions and
        labels; fold the dense q-grid to its irreducible wedge.  Only the
        polarisation average is then exact, and it is returned in all three
        components of ``eps2_indirect``; pass ``sym_rots=None`` for the
        resolved diagonal components.
    orbital_positions : ndarray, shape ``(nawf, 3)``, optional
        PAO orbital centres (pair-resolved Wigner-Seitz sums and the intersite
        position term of the velocity); ``tau_cryst`` is then required.
    ispin : int, optional
        Spin channel of ``HRs``.
    kblock : int, optional
        Dense k-points per task.
    comm : mpi4py communicator, optional
        ``MPI.COMM_WORLD`` by default; the ``(q, k-block)`` tasks are distributed
        over its ranks.
    allow_missing_long_range : bool, optional
        Run a polar material without the long-range dipole term instead of raising.

    Returns
    -------
    dict
        ``omega_ev`` ``(nomega,)``; ``eps2_indirect`` and ``eps2_indirect_lorentz``
        ``(ntemp, neta, 3, nomega)``; ``eps2_direct`` and ``eps2_direct_lorentz``
        ``(ntemp, 3, nomega)``; ``alpha_cm`` ``(ntemp, neta, nomega)`` (direct +
        indirect, polarisation averaged, Gaussian delta); ``fermi_energy_ev``,
        ``indirect_gap_ev``, ``direct_gap_ev``, ``etas_ev``, ``temps_k``,
        ``nk_dense``, ``nq_dense``, ``bands`` (window band indices).

    Raises
    ------
    ValueError
        If the grids are incommensurate or no gap is found for the mid-gap
        Fermi level.

    Notes
    -----
    The workflow mirrors EPW's ``lindabs``:

    1. build ``g(R_e, R_p)`` and the phonon interpolator from the coarse EPW data;
    2. diagonalise the PAO Hamiltonian on the dense k-grid and evaluate the band
       velocities there (one copy per node);
    3. for every irreducible dense q (star-weighted) and every block of dense k,
       interpolate the coupling, take ``k+q`` on the grid, and accumulate
       :func:`indirect_absorption_kernel`;
    4. add the direct term, :func:`direct_absorption_kernel`, once.

    The irreducible-q reduction leaves the polarisation-averaged spectrum
    unchanged, but not the individual components (each star is represented by a
    single orientation of q), so with ``sym_rots`` the indirect components are
    replaced by their average.
    """
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if nk_dense % nq_dense != 0:
        raise ValueError(f'nk_dense={nk_dense} must be a multiple of nq_dense={nq_dense}.')
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY

    vertex = prepare_dense_vertex(
        A, kpts_cryst, bg, at, coupling_dir, qgrid_coarse, None, ng, None,
        source='epw', masses_amu=masses_amu, tau_cryst=tau_cryst,
        orbital_positions=orbital_positions, comm=comm,
        allow_missing_long_range=allow_missing_long_range,
    )  # fmt: skip
    g_ReRp, phonon_at_q = vertex['g_ReRp'], vertex['phonon_at_q']
    Nint_p, W_p, Midx_p = vertex['phonon_ws']

    degauss = degauss_ev / RY_TO_EV
    electrons = precompute_dense_electrons(
        HRs, at, nk_dense, [degauss], None, tuple(ng), ispin=ispin,
        orbital_positions=orbital_positions, velocities=True, alat=alat,
    )  # fmt: skip
    energies = electrons['E']  # (nkd, nawf), Ry, scale of HRs
    nkd = energies.shape[0]

    nocc = int(round(nelec / spin_degeneracy))
    vbm, cbm = float(energies[:, nocc - 1].max()), float(energies[:, nocc].min())
    direct_gap = float(np.min(energies[:, nocc] - energies[:, nocc - 1]))
    if fermi_energy_ev is None:
        if cbm <= vbm:
            raise ValueError('No gap above band %d: pass fermi_energy_ev explicitly.' % nocc)
        fermi = 0.5 * (vbm + cbm)
    else:
        fermi = fermi_energy_ev / RY_TO_EV
    fsthick = fsthick_ev / RY_TO_EV
    in_window = np.nonzero(np.any(np.abs(energies - fermi) < fsthick, axis=0))[0]
    bands = np.arange(in_window.min(), in_window.max() + 1)
    energy_rel = energies[:, bands] - fermi

    temps_ry = np.asarray(temps_k, dtype=float) * KELVIN_TO_EV / RY_TO_EV
    etas = np.asarray(etas_ev, dtype=float) / RY_TO_EV
    omin, omax, ostep = omega_ev
    photon_ev = omin + ostep * np.arange(int(np.floor((omax - omin) / ostep + 1.0e-6)) + 1)
    photon = photon_ev / RY_TO_EV
    eps_acoustic = eps_acoustic_cm / EV_TO_CM1 / RY_TO_EV

    nl_correction = None
    if nonlocal_velocity is not None:
        nl_correction = nonlocal_velocity_operator(nonlocal_velocity, bg, alat)
    velocities, velocity_window = _shared_velocities(electrons, bands, nl_correction, comm, kblock)

    if sym_rots is not None:
        rots = sym_rots
        if tau_cryst is not None and species is not None:
            rots = _crystal_point_group(sym_rots, tau_cryst, species)
        qmesh, qweights = irreducible_qmesh(nq_dense, rots, at, bg)
    else:
        axes = [np.arange(nq_dense) / nq_dense for _ in range(3)]
        qmesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        qweights = np.ones(qmesh.shape[0])
    qweights = qweights / qweights.sum()
    if rank == 0:
        print(
            '  phonon-assisted absorption: %d bands in the window, %d q (of %d^3), %d^3 k'
            % (bands.size, qmesh.shape[0], nq_dense, nk_dense),
            flush=True,
        )

    ntemp, neta, nomega = temps_ry.size, etas.size, photon.size
    eps_ind = np.zeros((ntemp, neta, 3, nomega))
    eps_ind_l = np.zeros((ntemp, neta, 3, nomega))
    eps_dir = np.zeros((ntemp, 3, nomega))
    eps_dir_l = np.zeros((ntemp, 3, nomega))

    kblocks = [np.arange(s0, min(s0 + kblock, nkd)) for s0 in range(0, nkd, kblock)]
    tasks = [(iq, ib) for iq in range(qmesh.shape[0]) for ib in range(len(kblocks))]
    tstart, tstop = load_balancing(size, rank, len(tasks))
    V = electrons['V']
    phkg = electrons['phkg']
    current_q = None
    for iq, ib in tasks[tstart:tstop]:
        if iq != current_q:
            current_q = iq
            q_cryst = qmesh[iq]
            freq_thz, z = _phonon_modes_at_q(q_cryst, phonon_at_q)
            zmass = z / np.sqrt(mass_flat_ry)[None, :]
            omega_ry = freq_thz / RY_TO_THZ  # imaginary modes are negative -> skipped
            gn_flat = vertex_ws_coefficients(
                g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p), electrons
            )
            shift = np.round(np.asarray(q_cryst) * nk_dense).astype(int)
            ikq = _kq_flat_index(nk_dense, shift)
            weight = qweights[iq] / (nkd * cell_volume)
        kk = kblocks[ib]
        kq = ikq[kk]
        coupling = band_vertex(gn_flat, phkg[kk], V[kq][:, :, bands], V[kk][:, :, bands], zmass)
        g_part, l_part = indirect_absorption_kernel(
            energy_rel[kk], energy_rel[kq], velocities[kk], velocities[kq], coupling,
            omega_ry, temps_ry, photon, etas, degauss, fsthick, eps_acoustic,
            weight=weight, spin_degeneracy=spin_degeneracy,
        )  # fmt: skip
        eps_ind += g_part
        eps_ind_l += l_part

    dstart, dstop = load_balancing(size, rank, len(kblocks))
    for kk in kblocks[dstart:dstop]:
        g_part, l_part = direct_absorption_kernel(
            energy_rel[kk], velocities[kk], temps_ry, photon, degauss, fsthick,
            weight=1.0 / (nkd * cell_volume), spin_degeneracy=spin_degeneracy,
        )  # fmt: skip
        eps_dir += g_part
        eps_dir_l += l_part

    if size > 1:
        for array in (eps_ind, eps_ind_l, eps_dir, eps_dir_l):
            comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    comm.Barrier()
    velocity_window.Free()
    vertex['window'].Free()

    if sym_rots is not None:  # only the polarisation average survives the q-folding
        eps_ind[...] = eps_ind.mean(axis=2, keepdims=True)
        eps_ind_l[...] = eps_ind_l.mean(axis=2, keepdims=True)
    eps_total = eps_ind.mean(axis=2) + eps_dir.mean(axis=1)[:, None, :]
    return {
        'omega_ev': photon_ev,
        'eps2_indirect': eps_ind,
        'eps2_indirect_lorentz': eps_ind_l,
        'eps2_direct': eps_dir,
        'eps2_direct_lorentz': eps_dir_l,
        'alpha_cm': absorption_coefficient(photon_ev, eps_total, refractive_index),
        'refractive_index': np.asarray(refractive_index, dtype=float),
        'fermi_energy_ev': fermi * RY_TO_EV,
        'indirect_gap_ev': (cbm - vbm) * RY_TO_EV,
        'direct_gap_ev': direct_gap * RY_TO_EV,
        'etas_ev': np.asarray(etas_ev, dtype=float),
        'temps_k': np.asarray(temps_k, dtype=float),
        'nk_dense': nk_dense,
        'nq_dense': nq_dense,
        'bands': bands,
    }


def write_absorption_outputs(results: dict, outdir: str) -> list[str]:
    """Write the absorption spectra in EPW's file layout plus an ``.npz`` archive.

    Parameters
    ----------
    results : dict
        Output of :func:`phonon_assisted_absorption_dense_q`.
    outdir : str
        Output directory (created if missing).

    Returns
    -------
    list of str
        Paths written: per temperature ``epsilon2_indabs_<T>K.dat`` and
        ``epsilon2_indabs_lorenz<T>K.dat`` (photon energy, polarisation-averaged
        ``Im eps`` for each eta), ``epsilon2_dirabs_<T>K.dat`` (photon energy,
        ``Im eps`` along x, y, z and their average, Gaussian and Lorentzian) and
        ``alpha_<T>K.dat`` (photon energy, absorption coefficient in cm^-1 for
        each eta), plus ``absorption.npz``.
    """
    os.makedirs(outdir, exist_ok=True)
    omega = results['omega_ev']
    etas = results['etas_ev']
    eta_header = ' '.join('%.3f' % e for e in etas)
    written = []
    for it, temp in enumerate(results['temps_k']):
        tag = '%.1f' % temp
        for name, key in (
            ('epsilon2_indabs_%sK.dat', 'eps2_indirect'),
            ('epsilon2_indabs_lorenz%sK.dat', 'eps2_indirect_lorentz'),
        ):
            path = os.path.join(outdir, name % tag)
            data = np.column_stack([omega, results[key][it].mean(axis=1).T])
            np.savetxt(
                path, data, fmt='%15.6f' + ' %22.14E' * etas.size,
                header='Photon energy (eV), Im eps (phonon-assisted) for eta (eV) = '
                + eta_header,
            )  # fmt: skip
            written.append(path)
        path = os.path.join(outdir, 'epsilon2_dirabs_%sK.dat' % tag)
        direct = results['eps2_direct'][it]
        direct_l = results['eps2_direct_lorentz'][it]
        data = np.column_stack(
            [omega, direct.T, direct.mean(axis=0), direct_l.T, direct_l.mean(axis=0)]
        )
        np.savetxt(
            path, data, fmt='%15.6f' + ' %22.14E' * 8,
            header='Photon energy (eV), Im eps (direct) x y z avg [gaussian], '
            'x y z avg [lorentzian]',
        )  # fmt: skip
        written.append(path)
        path = os.path.join(outdir, 'alpha_%sK.dat' % tag)
        data = np.column_stack([omega, results['alpha_cm'][it].T])
        np.savetxt(
            path, data, fmt='%15.6f' + ' %22.14E' * etas.size,
            header='Photon energy (eV), absorption coefficient (cm^-1, direct + indirect) '
            'for eta (eV) = ' + eta_header,
        )  # fmt: skip
        written.append(path)
    path = os.path.join(outdir, 'absorption.npz')
    np.savez(path, **{k: np.asarray(v) for k, v in results.items()})
    written.append(path)
    return written
