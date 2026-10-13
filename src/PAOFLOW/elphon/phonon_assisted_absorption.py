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
    fermi_levels: Sequence[float] | None = None,
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
    fermi_levels : sequence of float, optional
        Fermi level of each temperature (Ry) on the energy scale of
        ``energy_k``, which only sets the occupations (default 0 for all; the
        ``fsthick`` window stays centred on 0).

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

    mu = _fermi_levels(fermi_levels, temperatures.size)
    occ_k = np.stack([fermi_occupation(energy_k - m, t) for m, t in zip(mu, temperatures)])
    occ_kq = np.stack([fermi_occupation(energy_kq - m, t) for m, t in zip(mu, temperatures)])
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
    fermi_levels: Sequence[float] | None = None,
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
    fermi_levels : sequence of float, optional
        Fermi level of each temperature (Ry), as in
        :func:`indirect_absorption_kernel`.

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
    mu = _fermi_levels(fermi_levels, temperatures.size)
    occ = np.stack([fermi_occupation(energy_k - m, t) for m, t in zip(mu, temperatures)])
    pfac = occ[:, k_idx, i_idx] - occ[:, k_idx, j_idx]  # (nT, P)
    gauss = np.einsum('tp,pa,pw->taw', pfac, v2, gaussian_delta(x, degauss) * keep)
    lorentz = np.einsum('tp,pa,pw->taw', pfac, v2, lorentzian_delta(x, degauss) * keep)
    prefactor = weight * 8.0 * np.pi**2 * spin_degeneracy / photon_energy**2
    return gauss * prefactor, lorentz * prefactor


def _fermi_levels(fermi_levels: Sequence[float] | None, ntemp: int) -> NDArray[np.float64]:
    """Per-temperature Fermi levels, zero by default."""
    if fermi_levels is None:
        return np.zeros(ntemp)
    mu = np.atleast_1d(np.asarray(fermi_levels, dtype=float))
    if mu.size != ntemp:
        raise ValueError('need one Fermi level per temperature (%d), got %d' % (ntemp, mu.size))
    return mu


def intrinsic_fermi_levels(
    energies: NDArray[np.float64],
    temperatures: Sequence[float],
    nelec: float,
    spin_degeneracy: float = 2.0,
    niter: int = 200,
) -> NDArray[np.float64]:
    """Charge-neutral (intrinsic) Fermi level at each temperature.

    Parameters
    ----------
    energies : ndarray, shape ``(nk, nband)``
        Band energies of **all** bands on a uniform k-grid (any energy unit).
    temperatures : sequence of float
        ``k_B T`` values, same unit as ``energies`` (positive).
    nelec : float
        Valence electrons per cell.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.
    niter : int, optional
        Bisection steps.

    Returns
    -------
    ndarray, shape ``(ntemp,)``
        Fermi levels :math:`\\mu(T)` on the scale of ``energies``.

    Notes
    -----
    Solves the charge neutrality

    .. math::

        \\frac{g_s}{N_k} \\sum_{n\\mathbf{k}} f\\left(\\frac{\\varepsilon_{n\\mathbf{k}}
            - \\mu}{k_B T}\\right) = N_{el}

    by bisection, so the thermally excited electrons and holes balance; in a
    semiconductor :math:`\\mu(T)` moves from mid-gap toward the band with the
    smaller density of states.
    """
    energies = np.asarray(energies, dtype=float)
    nk = energies.shape[0]
    levels = []
    for temperature in np.atleast_1d(np.asarray(temperatures, dtype=float)):
        low, high = float(energies.min()) - 1.0, float(energies.max()) + 1.0
        for _ in range(niter):
            mid = 0.5 * (low + high)
            electrons = spin_degeneracy / nk * fermi_occupation(energies - mid, temperature).sum()
            if electrons > nelec:
                high = mid
            else:
                low = mid
        levels.append(0.5 * (low + high))
    return np.array(levels)


SILICON_VARSHNI = (1.170, 4.73e-4, 636.0)  # E_g(0) (eV), alpha (eV/K), beta (K)


def varshni_gap(
    e0_ev: float, alpha_ev_per_k: float = 4.73e-4, beta_k: float = 636.0
) -> Callable[[float], float]:
    """Temperature-dependent band gap of the Varshni form.

    Parameters
    ----------
    e0_ev : float
        Gap at 0 K (eV), e.g. the experimental value or an eACBN0 gap.
    alpha_ev_per_k, beta_k : float, optional
        Varshni parameters (defaults: silicon, Varshni, Physica 34, 149 (1967)).

    Returns
    -------
    callable
        ``T (K) -> E_g(T)`` in eV, the ``target_gap_ev`` of
        :func:`phonon_assisted_absorption_dense_q`.

    Notes
    -----
    .. math::

        E_g(T) = E_0 - \\frac{\\alpha T^2}{T + \\beta}.
    """

    def gap(temperature: float) -> float:
        return float(e0_ev - alpha_ev_per_k * temperature**2 / (temperature + beta_k))

    return gap


def scissor_per_temperature(
    temps_k: Sequence[float],
    dft_gap_ev: float,
    scissor_ev: float | None = None,
    target_gap_ev: float | Callable[[float], float] | None = None,
) -> NDArray[np.float64]:
    """Rigid conduction-band shift at each temperature.

    Parameters
    ----------
    temps_k : sequence of float
        Temperatures (K).
    dft_gap_ev : float
        Indirect gap of the (DFT) PAO spectrum on the dense grid (eV).
    scissor_ev : float, optional
        Explicit shift (eV), the same at every temperature.
    target_gap_ev : float or callable, optional
        Gap to reproduce (eV), fixed or ``T (K) -> E_g(T)`` (:func:`varshni_gap`).

    Returns
    -------
    ndarray, shape ``(ntemp,)``
        Scissor :math:`\\Delta(T) = E_g^{target}(T) - E_g^{DFT}` (eV); zeros when
        neither ``scissor_ev`` nor ``target_gap_ev`` is given.

    Raises
    ------
    ValueError
        If both ``scissor_ev`` and ``target_gap_ev`` are given.
    """
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    if scissor_ev is not None and target_gap_ev is not None:
        raise ValueError('give either scissor_ev or target_gap_ev, not both.')
    if scissor_ev is not None:
        return np.full(temps.size, float(scissor_ev))
    if target_gap_ev is None:
        return np.zeros(temps.size)
    if callable(target_gap_ev):
        targets = np.array([float(target_gap_ev(t)) for t in temps])
    else:
        targets = np.full(temps.size, float(target_gap_ev))
    return targets - float(dft_gap_ev)


def scissor_velocities(
    velocity: NDArray[np.complex128],
    energies: NDArray[np.float64],
    scissor: float,
    conduction: NDArray[np.bool_],
    mode: str = 'renormalized',
) -> NDArray[np.complex128]:
    """Velocity matrix elements of the scissor-corrected Hamiltonian.

    Parameters
    ----------
    velocity : ndarray, shape ``(nk, 3, nband, nband)``, complex
        Kohn-Sham (PAO) band velocities.
    energies : ndarray, shape ``(nk, nband)``
        Kohn-Sham band energies (same unit as ``scissor``).
    scissor : float
        Rigid shift of the conduction bands.
    conduction : ndarray of bool, shape ``(nband,)``
        Conduction-band mask.
    mode : {'renormalized', 'ks'}, optional
        ``'renormalized'`` (default) applies the Levine-Allan factor below;
        ``'ks'`` keeps the Kohn-Sham velocities (EPW's ``scissor``).

    Returns
    -------
    ndarray, shape ``(nk, 3, nband, nband)``, complex
        Velocities to use with the shifted energies.

    Raises
    ------
    ValueError
        For an unknown ``mode``.

    Notes
    -----
    A scissor operator :math:`H_\\Delta = H + \\Delta\\, P_c` (:math:`P_c` the
    conduction projector) leaves the position matrix elements unchanged, so the
    velocity :math:`i[H_\\Delta, \\mathbf r]` of a valence-conduction pair scales
    with its transition energy (Levine and Allan, Phys. Rev. Lett. 63, 1719
    (1989); Del Sole and Girlanda, Phys. Rev. B 48, 11789 (1993)):

    .. math::

        v^{\\Delta}_{mn} = v_{mn}\\,
            \\frac{\\varepsilon_m - \\varepsilon_n \\pm \\Delta}{\\varepsilon_m - \\varepsilon_n},

    while intraband and within-manifold elements (e.g. the free-carrier
    velocities) are unchanged.  The direct absorption then shifts rigidly.
    """
    if mode not in ('renormalized', 'ks'):
        raise ValueError("scissor velocity mode must be 'renormalized' or 'ks', got %r" % mode)
    if mode == 'ks' or scissor == 0.0:
        return velocity
    conduction = np.asarray(conduction, dtype=bool)
    cross = conduction[:, None] != conduction[None, :]
    shift = scissor * (conduction[:, None].astype(float) - conduction[None, :].astype(float))
    gap = energies[:, :, None] - energies[:, None, :]  # (nk, m, n)
    factor = np.where(cross[None], (gap + shift[None]) / np.where(cross[None], gap, 1.0), 1.0)
    return velocity * factor[:, None, :, :]


def scissor_shift_dielectric(
    omega_ev: NDArray[np.float64],
    eps1: NDArray[np.float64],
    eps2: NDArray[np.float64],
    scissor_ev: float,
    mode: str = 'renormalized',
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Apply a scissor to an independent-particle dielectric function.

    Parameters
    ----------
    omega_ev : ndarray, shape ``(nomega,)``
        Uniform photon-energy grid (eV).
    eps1, eps2 : ndarray, shape ``(nomega,)``
        Real (vacuum 1 included) and imaginary parts.
    scissor_ev : float
        Rigid conduction-band shift (eV).
    mode : {'renormalized', 'ks'}, optional
        Velocity convention of :func:`scissor_velocities`.

    Returns
    -------
    eps1, eps2 : ndarray, shape ``(nomega,)``
        Scissor-corrected real and imaginary parts.

    Notes
    -----
    With renormalized velocities the interband spectrum shifts rigidly,
    :math:`\\varepsilon_2^\\Delta(\\omega) = \\varepsilon_2(\\omega - \\Delta)`;
    with Kohn-Sham velocities it is also scaled by
    :math:`(\\omega - \\Delta)^2/\\omega^2`.  The real part changes by the
    Kramers-Kronig transform of the change of :math:`\\varepsilon_2` (so the
    contribution of the spectrum beyond the grid is kept); weight shifted past
    the end of the grid is dropped.
    """
    from ..response.do_epsilon import kramers_kronig_eps1

    if mode not in ('renormalized', 'ks'):
        raise ValueError("scissor velocity mode must be 'renormalized' or 'ks', got %r" % mode)
    omega = np.asarray(omega_ev, dtype=float)
    if scissor_ev == 0.0:
        return np.asarray(eps1, dtype=float), np.asarray(eps2, dtype=float)
    step = omega[1] - omega[0]
    grid = np.arange(0.0, omega.max() + 0.5 * step, step)
    original = np.interp(grid, omega, eps2, left=0.0, right=0.0)
    shifted = np.interp(grid - scissor_ev, grid, original, left=0.0, right=0.0)
    if mode == 'ks':
        with np.errstate(divide='ignore', invalid='ignore'):
            shifted = np.where(grid > 0.0, shifted * ((grid - scissor_ev) / grid) ** 2, 0.0)
    delta_eps1 = kramers_kronig_eps1(grid, shifted) - kramers_kronig_eps1(grid, original)
    return (
        np.asarray(eps1, dtype=float) + np.interp(omega, grid, delta_eps1),
        np.interp(omega, grid, shifted),
    )


def pao_band_gaps(
    HRs: NDArray[np.complex128],
    at: NDArray[np.float64],
    nk_dense: int,
    nelec: float,
    orbital_positions: NDArray[np.float64] | None = None,
    spin_degeneracy: float = 2.0,
    ispin: int = 0,
) -> dict:
    """Indirect and direct gaps of a PAO Hamiltonian on a dense k-grid.

    Parameters
    ----------
    HRs : ndarray, shape ``(nawf, nawf, n1, n2, n3, nspin)``, complex
        PAO Hamiltonian (eV), e.g. of an eACBN0 (DFT+U+V, ortho-atomic) run.
    at : ndarray, shape ``(3, 3)``
        Direct lattice vectors (rows, alat units).
    nk_dense : int
        Dense grid per axis (Wigner-Seitz interpolation, as the absorption).
    nelec : float
        Valence electrons.
    orbital_positions : ndarray, shape ``(nawf, 3)``, optional
        PAO orbital centres (pair-resolved Wigner-Seitz sums).
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.
    ispin : int, optional
        Spin channel.

    Returns
    -------
    dict
        ``indirect_gap_ev``, ``direct_gap_ev``, ``vbm_ev``, ``cbm_ev`` (on the
        energy scale of ``HRs``).  The ``target_gap_ev`` of a scissor-corrected
        :func:`phonon_assisted_absorption_dense_q` is ``indirect_gap_ev`` of the
        reference Hamiltonian.
    """
    electrons = precompute_dense_electrons(
        HRs, at, nk_dense, [0.01], None, tuple(HRs.shape[2:5]), ispin=ispin,
        orbital_positions=orbital_positions,
    )  # fmt: skip
    energies = electrons['E'] * RY_TO_EV
    nocc = int(round(nelec / spin_degeneracy))
    vbm, cbm = float(energies[:, nocc - 1].max()), float(energies[:, nocc].min())
    return {
        'indirect_gap_ev': cbm - vbm,
        'direct_gap_ev': float(np.min(energies[:, nocc] - energies[:, nocc - 1])),
        'vbm_ev': vbm,
        'cbm_ev': cbm,
    }


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
    fermi_energy_ev: float | str | None = None,
    etas_ev: Sequence[float] = EPW_ETAS_EV,
    eps_acoustic_cm: float = 0.1,
    refractive_index: float | NDArray[np.float64] = 3.4,
    nonlocal_velocity=None,
    kk_omega_max_ev: float | None = None,
    kk_degauss_ev: float = 0.1,
    scissor_ev: float | None = None,
    target_gap_ev: float | Callable[[float], float] | None = None,
    scissor_velocity: str = 'renormalized',
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
    fermi_energy_ev : float or 'intrinsic', optional
        Fermi level on the energy scale of ``HRs`` (eV); ``None`` puts it mid-gap
        (EPW ``efermi_read``) and ``'intrinsic'`` uses the charge-neutral level of
        each temperature (:func:`intrinsic_fermi_levels`), which sets the
        density of thermally excited carriers (free-carrier absorption).  The
        ``fsthick`` window stays centred mid-gap either way.
    etas_ev : sequence of float, optional
        Intermediate-state broadenings (eV); EPW's nine values by default.
    eps_acoustic_cm : float, optional
        Phonons below this energy (cm^-1) are skipped (EPW ``eps_acoustic``).
    refractive_index : float or ndarray, optional
        Real refractive index for the absorption coefficient (default 3.4).
    nonlocal_velocity : DataController, optional
        When given, adds the norm-conserving non-local pseudopotential velocity
        correction (:func:`nonlocal_velocity_operator`) built from it.
    kk_omega_max_ev : float, optional
        Also compute the direct ``Im eps`` over **all** PAO bands on
        ``[0, kk_omega_max_ev]`` (same step), the input of the Kramers-Kronig
        real part used by :func:`thermal_emissivity`.  About 10 eV: the PAO
        spectrum above that includes transitions into poorly projected (shifted)
        bands that violate the f-sum rule (check ``f_sum_ratio`` of
        :func:`thermal_emissivity`).
    kk_degauss_ev : float, optional
        Smearing (eV) of that wide-range direct term (default 0.1).  The real
        part is smooth, but the Kramers-Kronig transform of an under-sampled,
        weakly smeared spectrum oscillates (even below zero) and spoils the
        reflectivity; it also smooths the direct term the emissivity uses above
        the gap.
    scissor_ev : float, optional
        Rigid shift (eV) of the conduction bands (scissor operator).
    target_gap_ev : float or callable, optional
        Indirect gap (eV) to reproduce instead of the DFT one: a number (e.g. the
        experimental gap, or the dense-grid gap of an eACBN0 PAO Hamiltonian from
        :func:`pao_band_gaps`) or ``T (K) -> E_g(T)`` (:func:`varshni_gap`).  The
        scissor is :math:`\Delta(T) = E_g^{target}(T) - E_g^{DFT}`; the coupling
        and the wavefunctions stay those of the DFT (DFPT) calculation.
    scissor_velocity : {'renormalized', 'ks'}, optional
        Velocities of the scissor-corrected Hamiltonian
        (:func:`scissor_velocities`): Levine-Allan renormalized (default) or the
        Kohn-Sham ones (EPW's ``scissor``).
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
        ``nk_dense``, ``nq_dense``, ``bands`` (window band indices),
        ``fermi_levels_ev`` ``(ntemp,)``, ``scissor_ev``, ``indirect_gap_ev_t``,
        ``direct_gap_ev_t`` ``(ntemp,)`` (scissor-corrected; the scalar
        ``indirect_gap_ev``/``direct_gap_ev`` are those of the first temperature
        and ``*_gap_dft_ev`` the uncorrected ones), ``scissor_velocity``; with
        ``kk_omega_max_ev`` also
        ``omega_kk_ev`` and ``eps2_direct_wide`` ``(ntemp, 3, nomega_kk)``.

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
    intrinsic = isinstance(fermi_energy_ev, str)
    if intrinsic and fermi_energy_ev != 'intrinsic':
        raise ValueError("fermi_energy_ev must be a number, None or 'intrinsic'.")
    midgap = fermi_energy_ev is None or intrinsic
    if cbm <= vbm and midgap:
        raise ValueError(
            'No gap above band %d: pass fermi_energy_ev explicitly (metals: the mid-gap '
            "and 'intrinsic' Fermi levels, and the thermal emissivity, need a gap)." % nocc
        )

    temps_k = np.atleast_1d(np.asarray(temps_k, dtype=float))
    scissor_t = (
        scissor_per_temperature(temps_k, (cbm - vbm) * RY_TO_EV, scissor_ev, target_gap_ev)
        / RY_TO_EV
    )  # Ry, (ntemp,)
    if np.any(scissor_t != 0.0) and cbm <= vbm:
        raise ValueError('A scissor needs a gap above band %d.' % nocc)
    if scissor_velocity not in ('renormalized', 'ks'):
        raise ValueError("scissor_velocity must be 'renormalized' or 'ks'.")
    conduction_all = np.arange(energies.shape[1]) >= nocc
    # Temperatures sharing a scissor are processed together; the coupling is
    # computed once per (q, k-block) and only the second-order kernel is
    # repeated for each distinct scissor.
    scissor_values, group_of_t = np.unique(np.round(scissor_t, 12), return_inverse=True)
    groups = [
        (float(delta), np.nonzero(group_of_t == ig)[0]) for ig, delta in enumerate(scissor_values)
    ]

    fermi_dft = 0.5 * (vbm + cbm) if midgap else fermi_energy_ev / RY_TO_EV
    fsthick = fsthick_ev / RY_TO_EV
    # The scissor moves every state away from the mid-gap reference, so the
    # window of the unshifted spectrum (widened for a negative scissor) holds
    # the windows of all temperatures; fsthick is applied per state in the kernels.
    widen = max(0.0, -float(scissor_t.min())) / 2.0 if midgap else abs(scissor_t).max()
    in_window = np.nonzero(np.any(np.abs(energies - fermi_dft) < fsthick + widen, axis=0))[0]
    bands = np.arange(in_window.min(), in_window.max() + 1)
    conduction = conduction_all[bands]

    temps_ry = temps_k * KELVIN_TO_EV / RY_TO_EV
    etas = np.asarray(etas_ev, dtype=float) / RY_TO_EV
    omin, omax, ostep = omega_ev
    photon_ev = omin + ostep * np.arange(int(np.floor((omax - omin) / ostep + 1.0e-6)) + 1)
    photon = photon_ev / RY_TO_EV
    eps_acoustic = eps_acoustic_cm / EV_TO_CM1 / RY_TO_EV

    # Per scissor group: reference energy, window energies, Fermi levels.
    reference = np.zeros(temps_k.size)
    fermi_levels = np.zeros(temps_k.size)  # relative to the reference
    group_energy = []
    for delta, tidx in groups:
        fermi_g = fermi_dft + 0.5 * delta if midgap else fermi_dft
        reference[tidx] = fermi_g
        shifted_all = energies + delta * conduction_all - fermi_g
        if intrinsic:
            fermi_levels[tidx] = intrinsic_fermi_levels(
                shifted_all, temps_ry[tidx], nelec, spin_degeneracy
            )
        group_energy.append(shifted_all[:, bands])

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
        for (delta, tidx), energy_g in zip(groups, group_energy):
            v_k = scissor_velocities(
                velocities[kk], energies[kk][:, bands], delta, conduction, scissor_velocity
            )
            v_kq = scissor_velocities(
                velocities[kq], energies[kq][:, bands], delta, conduction, scissor_velocity
            )
            g_part, l_part = indirect_absorption_kernel(
                energy_g[kk], energy_g[kq], v_k, v_kq, coupling,
                omega_ry, temps_ry[tidx], photon, etas, degauss, fsthick, eps_acoustic,
                weight=weight, spin_degeneracy=spin_degeneracy, fermi_levels=fermi_levels[tidx],
            )  # fmt: skip
            eps_ind[tidx] += g_part
            eps_ind_l[tidx] += l_part

    dstart, dstop = load_balancing(size, rank, len(kblocks))
    for kk in kblocks[dstart:dstop]:
        for (delta, tidx), energy_g in zip(groups, group_energy):
            v_k = scissor_velocities(
                velocities[kk], energies[kk][:, bands], delta, conduction, scissor_velocity
            )
            g_part, l_part = direct_absorption_kernel(
                energy_g[kk], v_k, temps_ry[tidx], photon, degauss, fsthick,
                weight=1.0 / (nkd * cell_volume), spin_degeneracy=spin_degeneracy,
                fermi_levels=fermi_levels[tidx],
            )  # fmt: skip
            eps_dir[tidx] += g_part
            eps_dir_l[tidx] += l_part

    # Direct term over all bands on [0, kk_omega_max_ev] for the Kramers-Kronig ε1.
    wide = None
    if kk_omega_max_ev is not None:
        omega_kk_ev = ostep * np.arange(int(np.floor(kk_omega_max_ev / ostep + 1.0e-6)) + 1)
        wide = np.zeros((ntemp, 3, omega_kk_ev.size))
        for kk in kblocks[dstart:dstop]:
            v_all = band_velocities(electrons, kk, slice(None), nl_correction)
            for delta, tidx in groups:
                v_g = scissor_velocities(
                    v_all, energies[kk], delta, conduction_all, scissor_velocity
                )
                energy_all = energies[kk] + delta * conduction_all - reference[tidx[0]]
                g_part, _ = direct_absorption_kernel(
                    energy_all, v_g, temps_ry[tidx], omega_kk_ev[1:] / RY_TO_EV,
                    kk_degauss_ev / RY_TO_EV, np.inf,
                    weight=1.0 / (nkd * cell_volume), spin_degeneracy=spin_degeneracy,
                    fermi_levels=fermi_levels[tidx],
                )  # fmt: skip
                wide[tidx, :, 1:] += g_part

    if size > 1:
        arrays = [eps_ind, eps_ind_l, eps_dir, eps_dir_l] + ([wide] if wide is not None else [])
        for array in arrays:
            comm.Allreduce(MPI.IN_PLACE, array, op=MPI.SUM)
    comm.Barrier()
    velocity_window.Free()
    vertex['window'].Free()

    if sym_rots is not None:  # only the polarisation average survives the q-folding
        eps_ind[...] = eps_ind.mean(axis=2, keepdims=True)
        eps_ind_l[...] = eps_ind_l.mean(axis=2, keepdims=True)
    eps_total = eps_ind.mean(axis=2) + eps_dir.mean(axis=1)[:, None, :]
    results = {
        'omega_ev': photon_ev,
        'eps2_indirect': eps_ind,
        'eps2_indirect_lorentz': eps_ind_l,
        'eps2_direct': eps_dir,
        'eps2_direct_lorentz': eps_dir_l,
        'alpha_cm': absorption_coefficient(photon_ev, eps_total, refractive_index),
        'refractive_index': np.asarray(refractive_index, dtype=float),
        'fermi_energy_ev': reference[0] * RY_TO_EV,
        'indirect_gap_ev': (cbm - vbm + scissor_t[0]) * RY_TO_EV,
        'direct_gap_ev': (direct_gap + scissor_t[0]) * RY_TO_EV,
        'indirect_gap_dft_ev': (cbm - vbm) * RY_TO_EV,
        'direct_gap_dft_ev': direct_gap * RY_TO_EV,
        'indirect_gap_ev_t': (cbm - vbm + scissor_t) * RY_TO_EV,
        'direct_gap_ev_t': (direct_gap + scissor_t) * RY_TO_EV,
        'scissor_ev': scissor_t * RY_TO_EV,
        'scissor_velocity': scissor_velocity,
        'etas_ev': np.asarray(etas_ev, dtype=float),
        'temps_k': temps_k,
        'nk_dense': nk_dense,
        'nq_dense': nq_dense,
        'bands': bands,
        'fermi_levels_ev': (reference + fermi_levels) * RY_TO_EV,
        'spin_degeneracy': spin_degeneracy,
        'nelec': float(nelec),
        'cell_volume_bohr3': float(cell_volume),
    }
    if wide is not None:
        results['omega_kk_ev'] = omega_kk_ev
        results['eps2_direct_wide'] = wide
    return results


def thermal_emissivity(
    results: dict,
    thickness_m: float,
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
    ntheta: int = 90,
    eta_ev: float = 0.05,
    taper_ev: float = 0.1,
    direct_dielectric: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    | None = None,
) -> dict:
    """Spectral and total hemispherical emissivity of a slab at each temperature.

    Parameters
    ----------
    results : dict
        Output of :func:`phonon_assisted_absorption_dense_q`, run with
        ``fermi_energy_ev='intrinsic'`` and either ``kk_omega_max_ev`` (keys
        ``omega_kk_ev``, ``eps2_direct_wide``) or together with
        ``direct_dielectric``.
    thickness_m : float
        Slab thickness (m); ``inf`` gives the opaque half-space.
    angles_deg : sequence of float, optional
        Emission angles (degrees) of the directional spectra.
    ntheta : int, optional
        Polar-angle samples of the hemispherical integral.
    eta_ev : float, optional
        Intermediate-state broadening of the phonon-assisted term (closest
        computed value).
    taper_ev : float, optional
        Width (eV) of the cosine taper that switches the phonon-assisted term off
        just below the direct gap.
    direct_dielectric : tuple of ndarray, optional
        ``(omega_ev, eps1, eps2)`` of the direct (independent-particle)
        dielectric function from a separate, better-converged calculation:
        PAOFLOW's ``dielectric_tensor`` on the extended basis with the non-local
        velocity (:func:`read_direct_dielectric`).  ``eps1``/``eps2`` have shape
        ``(nomega_d,)`` or ``(ncomp, nomega_d)`` (averaged); ``eps1`` includes
        the vacuum 1 and the grid (uniform) must cover ``results['omega_ev']``.
        When given, the wide-range direct term of ``results`` is not used.  A
        scissor of ``results`` (``scissor_ev``, ``scissor_velocity``) is applied
        to it too (:func:`scissor_shift_dielectric`).

    Returns
    -------
    dict
        ``omega_ev`` ``(nomega,)``, ``temps_k`` ``(ntemp,)``, ``eps1``, ``eps2``,
        ``hemispherical_slab``, ``hemispherical_opaque`` ``(ntemp, nomega)``,
        ``directional_slab`` ``(ntemp, nangle, nomega)``, ``total_slab``,
        ``total_opaque``, ``planck_coverage`` ``(ntemp,)``, ``thickness_m``,
        ``angles_deg``, ``eta_ev``, ``direct_source`` (``'external'`` or
        ``'kramers-kronig'``) and, when ``results`` carries ``nelec`` and
        ``cell_volume_bohr3``, ``f_sum_ratio`` ``(ntemp,)``: the f-sum
        :math:`\\int \\omega\\, \\varepsilon_2\\, d\\omega` of the direct
        spectrum divided by :math:`\\pi \\omega_p^2 / 2` of all valence
        electrons (close to 1 for a complete, physical spectrum).

    Raises
    ------
    NotImplementedError
        For a metal (no gap): the intraband (Drude) term is not included.
    KeyError
        If neither a wide-range direct term nor ``direct_dielectric`` is given.
    ValueError
        If ``direct_dielectric`` does not cover the photon-energy grid.

    Notes
    -----
    For every temperature :math:`T` the polarisation-averaged dielectric
    function is assembled from a direct part and the phonon-assisted
    :math:`\\varepsilon_2^{ind}(\\omega, T)`, which is tapered off by
    :math:`w(\\omega)` just below the direct gap, where its second-order
    amplitude diverges.

    * Without ``direct_dielectric``:
      :math:`\\varepsilon_2 = \\varepsilon_2^{dir} + w\\,\\varepsilon_2^{ind}`, with
      the direct term over all PAO bands, and :math:`\\varepsilon_1` by
      Kramers-Kronig (:func:`~PAOFLOW.response.do_epsilon.kramers_kronig_eps1`).
    * With ``direct_dielectric``
      (:math:`\\varepsilon^{ext} = \\varepsilon_1^{ext} + i\\varepsilon_2^{ext}`):

      .. math::

          \\varepsilon_2 &= w\\,(\\varepsilon_2^{dir} + \\varepsilon_2^{ind})
                          + (1 - w)\\,\\varepsilon_2^{ext}, \\\\
          \\varepsilon_1 &= \\varepsilon_1^{ext}
                          + \\mathrm{KK}[w\\,\\varepsilon_2^{ind}] - 1,

      i.e. the external spectrum above the direct gap and for the real part,
      and this run's direct term (Gaussian, window bands) plus the
      phonon-assisted term below it.  The broadening tails of the external
      spectrum (Drude-Lorentz, ``delta`` of ``dielectric_tensor``) would
      otherwise add a sub-gap absorption orders of magnitude larger than the
      phonon-assisted and free-carrier one; :math:`\\varepsilon_1` is insensitive
      to that broadening.

    The slab absorptance
    (:func:`~PAOFLOW.response.do_epsilon.slab_directional_absorptance`) is the
    emissivity by Kirchhoff's law; its hemispherical average and the
    Planck-weighted total at the **same** temperature,

    .. math::

        \\varepsilon(T) = \\frac{\\int \\varepsilon(\\omega, T)\\, B(\\omega, T)\\, d\\omega}
                               {\\int B(\\omega, T)\\, d\\omega},

    are taken over the photon-energy grid; ``planck_coverage`` is the fraction
    of the blackbody spectrum that the grid covers.
    """
    from ..response.do_epsilon import kramers_kronig_eps1
    from ..response.emissivity import emissivity_from_dielectric

    if float(results.get('indirect_gap_ev', 1.0)) <= 0.0:
        raise NotImplementedError(
            'thermal_emissivity treats semiconductors and insulators only: for a metal the '
            'intraband (Drude) term sets the infrared reflectivity; use '
            'PAOFLOW.response.emissivity.metal_thermal_emissivity.'
        )
    if direct_dielectric is None and 'eps2_direct_wide' not in results:
        raise KeyError(
            'thermal_emissivity needs a run with kk_omega_max_ev set, or direct_dielectric.'
        )
    omega = np.asarray(results['omega_ev'], dtype=float)
    temps = np.atleast_1d(np.asarray(results['temps_k'], dtype=float))
    etas = np.asarray(results['etas_ev'], dtype=float)
    ieta = int(np.argmin(np.abs(etas - eta_ev)))
    ntemp = temps.size
    # scissor-corrected direct gap and scissor of each temperature
    direct_gaps = np.asarray(
        results.get('direct_gap_ev_t', np.full(ntemp, float(results['direct_gap_ev']))), dtype=float
    )
    scissors = np.asarray(results.get('scissor_ev', np.zeros(ntemp)), dtype=float)
    scissor_mode = str(results.get('scissor_velocity', 'renormalized'))

    def taper(energies: NDArray[np.float64], direct_gap: float) -> NDArray[np.float64]:
        # 1 below E_g^dir - taper_ev, 0 above E_g^dir
        return np.sin(0.5 * np.pi * np.clip((direct_gap - energies) / taper_ev, 0.0, 1.0)) ** 2

    nomega = omega.size
    eps1_t = np.zeros((ntemp, nomega))
    eps2_t = np.zeros((ntemp, nomega))
    f_sum_ratio = None
    plasma2_ev2 = None
    if 'nelec' in results and 'cell_volume_bohr3' in results:
        # omega_p^2 = 16 pi n Ry^2 in Rydberg atomic units (e^2 = 2, m = 1/2)
        density = float(results['nelec']) / float(results['cell_volume_bohr3'])
        plasma2_ev2 = 16.0 * np.pi * density * RY_TO_EV**2
        f_sum_ratio = np.zeros(ntemp)

    if direct_dielectric is not None:
        ext_omega, ext_eps1, ext_eps2 = (np.asarray(a, dtype=float) for a in direct_dielectric)
        if ext_eps1.ndim > 1:
            ext_eps1, ext_eps2 = ext_eps1.mean(axis=0), ext_eps2.mean(axis=0)
        step = omega[1] - omega[0]
        if ext_omega.min() > omega.min() + 0.5 * step or ext_omega.max() < omega.max() - 0.5 * step:
            raise ValueError(
                'direct_dielectric covers %.3f-%.3f eV, the photon-energy grid %.3f-%.3f eV'
                % (ext_omega.min(), ext_omega.max(), omega.min(), omega.max())
            )
        kk_grid = np.arange(0.0, omega.max() + 0.5 * step, step)  # uniform, from 0
        shifted_ext = {}  # scissor (eV) -> external (eps1, eps2) on the photon grid
        for scissor in np.unique(scissors):
            e1, e2 = scissor_shift_dielectric(ext_omega, ext_eps1, ext_eps2, scissor, scissor_mode)
            shifted_ext[scissor] = (
                np.interp(omega, ext_omega, e1),
                np.interp(omega, ext_omega, e2),
            )

    for it, temp in enumerate(temps):
        indirect = np.asarray(results['eps2_indirect'])[it, ieta].mean(axis=0)
        if direct_dielectric is None:
            omega_kk = np.asarray(results['omega_kk_ev'], dtype=float)
            eps2_kk = np.asarray(results['eps2_direct_wide'])[it].mean(axis=0) + taper(
                omega_kk, direct_gaps[it]
            ) * np.interp(omega_kk, omega, indirect, left=0.0, right=0.0)
            eps1_kk = kramers_kronig_eps1(omega_kk, eps2_kk)
            eps1 = np.interp(omega, omega_kk, eps1_kk)
            eps2 = np.interp(omega, omega_kk, eps2_kk)
            f_sum = np.trapezoid(omega_kk * eps2_kk, omega_kk)
        else:
            direct = np.asarray(results['eps2_direct'])[it].mean(axis=0)
            weight = taper(omega, direct_gaps[it])
            eps1_ext, eps2_ext = shifted_ext[scissors[it]]
            eps2 = weight * (direct + indirect) + (1.0 - weight) * eps2_ext
            phonon_assisted = np.interp(kk_grid, omega, weight * indirect, left=0.0, right=0.0)
            delta_eps1 = kramers_kronig_eps1(kk_grid, phonon_assisted) - 1.0
            eps1 = eps1_ext + np.interp(omega, kk_grid, delta_eps1)
            f_sum = np.trapezoid(ext_omega * ext_eps2, ext_omega)
        if f_sum_ratio is not None:
            f_sum_ratio[it] = f_sum / (0.5 * np.pi * plasma2_ev2)
        eps1_t[it], eps2_t[it] = eps1, eps2
    out = emissivity_from_dielectric(omega, eps1_t, eps2_t, temps, thickness_m, angles_deg, ntheta)
    out['eta_ev'] = float(etas[ieta])
    out['direct_source'] = 'kramers-kronig' if direct_dielectric is None else 'external'
    if f_sum_ratio is not None:
        out['f_sum_ratio'] = f_sum_ratio
    return out


def read_direct_dielectric(
    outdir: str, components: Sequence[str] = ('xx', 'yy', 'zz')
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Read the diagonal dielectric function written by PAOFLOW's ``dielectric_tensor``.

    Parameters
    ----------
    outdir : str
        Output directory of the ``dielectric_tensor`` run (files
        ``epsr_<c>.dat`` and ``epsi_<c>.dat``, photon energy in eV in the first
        column).  Typically a separate run on the extended basis with
        ``gradient_and_momenta(nonlocal_velocity=True)``.
    components : sequence of str, optional
        Diagonal components to read (default ``xx``, ``yy``, ``zz``).

    Returns
    -------
    omega_ev, eps1, eps2 : ndarray
        Photon energies ``(nomega,)`` and the real (vacuum 1 included) and
        imaginary parts ``(ncomp, nomega)``, the ``direct_dielectric`` input of
        :func:`thermal_emissivity`.

    Raises
    ------
    ValueError
        If the components use different energy grids.
    """
    omega = None
    eps1, eps2 = [], []
    for comp in components:
        real = np.loadtxt(os.path.join(outdir, 'epsr_%s.dat' % comp))
        imag = np.loadtxt(os.path.join(outdir, 'epsi_%s.dat' % comp))
        if omega is None:
            omega = real[:, 0]
        if not (np.allclose(real[:, 0], omega) and np.allclose(imag[:, 0], omega)):
            raise ValueError('epsr/epsi files of %s use different energy grids' % comp)
        eps1.append(real[:, 1])
        eps2.append(imag[:, 1])
    return omega, np.array(eps1), np.array(eps2)


def write_emissivity_outputs(emissivity: dict, outdir: str) -> list[str]:
    """Write the thermal-emissivity spectra and totals.

    Parameters
    ----------
    emissivity : dict
        Output of :func:`thermal_emissivity`.
    outdir : str
        Output directory (created if missing).

    Returns
    -------
    list of str
        Paths written: per temperature ``eps_<T>K.dat`` (photon energy, Re and
        Im eps), ``emish_<T>K.dat`` (photon energy, slab and opaque
        hemispherical emissivity) and ``emis_th<deg>_<T>K.dat`` (photon energy,
        directional slab emissivity), plus ``emist.dat`` (temperature, total slab
        and opaque emissivity, Planck coverage) and ``emissivity.npz``.
    """
    os.makedirs(outdir, exist_ok=True)
    omega = emissivity['omega_ev']
    thickness_um = emissivity['thickness_m'] * 1.0e6
    written = []

    def save(name, data, header):
        path = os.path.join(outdir, name)
        np.savetxt(path, data, fmt='%15.6f' + ' %15.8f' * (data.shape[1] - 1), header=header)
        written.append(path)

    for it, temp in enumerate(emissivity['temps_k']):
        tag = '%.1f' % temp
        save(
            'eps_%sK.dat' % tag,
            np.column_stack([omega, emissivity['eps1'][it], emissivity['eps2'][it]]),
            'Photon energy (eV), Re eps (Kramers-Kronig), Im eps (direct + phonon-assisted)',
        )
        save(
            'emish_%sK.dat' % tag,
            np.column_stack(
                [
                    omega,
                    emissivity['hemispherical_slab'][it],
                    emissivity['hemispherical_opaque'][it],
                ]
            ),
            'Photon energy (eV), hemispherical emissivity: slab (d = %g um), opaque half-space'
            % thickness_um,
        )
        for ia, angle in enumerate(emissivity['angles_deg']):
            save(
                'emis_th%d_%sK.dat' % (int(round(angle)), tag),
                np.column_stack([omega, emissivity['directional_slab'][it, ia]]),
                'Photon energy (eV), directional emissivity at %g deg (slab, d = %g um)'
                % (angle, thickness_um),
            )
    save(
        'emist.dat',
        np.column_stack(
            [
                emissivity['temps_k'],
                emissivity['total_slab'],
                emissivity['total_opaque'],
                emissivity['planck_coverage'],
            ]
        ),
        'T (K), total hemispherical emissivity: slab (d = %g um), opaque half-space; '
        'Planck fraction covered by the photon-energy grid' % thickness_um,
    )
    path = os.path.join(outdir, 'emissivity.npz')
    np.savez(path, **{k: np.asarray(v) for k, v in emissivity.items()})
    written.append(path)
    return written


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
