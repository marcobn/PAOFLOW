"""Electron-phonon self-energies for spectral functions and ARPES.

Three levels, all returning ``Sigma(omega, T)`` on an energy grid relative to the Fermi level:

1. :func:`eliashberg_self_energy`: the isotropic (or per-state) Engelsberg-Schrieffer form from
   an Eliashberg function :math:`\\alpha^2F(\\Omega)` (constant density of states),

   .. math::

       \\Sigma(\\omega, T) = \\int_0^\\infty d\\Omega\\,\\alpha^2F(\\Omega)\\Big[-2\\pi i\\,(n_\\Omega + \\tfrac12)
       + \\psi\\Big(\\tfrac12 + i\\frac{\\Omega-\\omega}{2\\pi k_BT}\\Big)
       - \\psi\\Big(\\tfrac12 - i\\frac{\\Omega+\\omega}{2\\pi k_BT}\\Big)\\Big],

   with :math:`\\psi` the digamma function, :math:`\\mathrm{Re}\\,\\Sigma'(0) = -\\lambda` and
   :math:`\\mathrm{Im}\\,\\Sigma(\\omega) \\to -\\pi\\int_0^{|\\omega|}\\alpha^2F` at :math:`T = 0`
   (Grimvall, *The electron-phonon interaction in metals*, 1981);
2. :func:`fan_migdal_kernel`: the state-resolved Fan-Migdal self-energy of a block of states
   from the coupling at one q,

   .. math::

       \\Sigma_{n\\mathbf k}(\\omega) = \\sum_{m\\nu\\mathbf q} w_{\\mathbf q}\\,
       \\frac{|g_{mn\\nu}(\\mathbf k,\\mathbf q)|^2}{2\\omega_{\\mathbf q\\nu}}\\Big[
       \\frac{n_{\\mathbf q\\nu} + f_{m\\mathbf k+\\mathbf q}}{\\omega - \\varepsilon_{m\\mathbf k+\\mathbf q} + \\omega_{\\mathbf q\\nu} + i\\eta}
       + \\frac{n_{\\mathbf q\\nu} + 1 - f_{m\\mathbf k+\\mathbf q}}{\\omega - \\varepsilon_{m\\mathbf k+\\mathbf q} - \\omega_{\\mathbf q\\nu} + i\\eta}\\Big],

   EPW ``selfen_elec`` / ``specfun_el`` (no Debye-Waller term); its on-shell imaginary part is
   the scattering rate of :func:`~PAOFLOW.elphon.carrier_scattering.scattering_rate_kernel`
   with a Lorentzian delta of width :math:`\\eta`;
3. :func:`fan_migdal_self_energy`: the dense-q driver evaluating 2 at arbitrary k-points (an
   ARPES path or mesh, no grid commensurability) with the interpolated vertex of
   :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.prepare_dense_vertex`.

:class:`TabulatedSelfEnergy` adapts the result to the self-energy protocol of
:mod:`PAOFLOW.arpes` (``f(w, energies, kpoints, bands)``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma

from .adaptive_smearing import grid_spacing, state_widths
from .carrier_scattering import fermi_levels_for_doping, statistics_scissor
from .coupling_screening import resolve_coupling_screening, screening_report
from .do_pao_eph_dense_q import _phonon_modes_at_q, g_Re_at_q, prepare_dense_vertex
from .elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    band_vertex,
    bloch_states_at,
    group_velocities_at,
    precompute_dense_electrons,
    release_dense_electrons,
    vertex_phases_at,
    vertex_ws_coefficients,
)
from .pao_electrons import check_orbital_dimensions
from .phonon_assisted_absorption import (
    EV_TO_CM1,
    KELVIN_TO_EV,
    bose_occupation,
    fermi_occupation,
)


def eliashberg_self_energy(
    omega_ev: NDArray[np.float64],
    a2f: NDArray[np.float64],
    w_ev: NDArray[np.float64],
    temperature_k: float,
) -> NDArray[np.complex128]:
    """Engelsberg-Schrieffer self-energy of an Eliashberg function (eV).

    Parameters
    ----------
    omega_ev : ndarray ``(nomega,)``
        Phonon energies of the Eliashberg function (eV, positive).
    a2f : ndarray ``(..., nomega)``
        :math:`\\alpha^2F(\\Omega)`; leading axes (states) are broadcast.
    w_ev : ndarray ``(nw,)``
        Electron energies relative to the Fermi level.
    temperature_k : float
        ``0`` selects the zero-temperature closed form.

    Returns
    -------
    ndarray ``(..., nw)``, complex
        ``Sigma(w)``; ``Im Sigma <= 0``.

    Notes
    -----
    At :math:`T = 0`: :math:`\\mathrm{Re}\\,\\Sigma(\\omega) = \\int d\\Omega\\,\\alpha^2F(\\Omega)
    \\ln|(\\Omega - \\omega)/(\\Omega + \\omega)|` and :math:`\\mathrm{Im}\\,\\Sigma(\\omega) =
    -\\pi\\int_0^{|\\omega|} d\\Omega\\,\\alpha^2F(\\Omega)`.
    """
    omega = np.asarray(omega_ev, dtype=float)
    w = np.asarray(w_ev, dtype=float)
    a2f = np.asarray(a2f, dtype=float)
    if temperature_k <= 0.0:
        numerator = np.abs(omega[None, :] - w[:, None])
        denominator = np.abs(omega[None, :] + w[:, None])
        ratio = np.maximum(numerator, 1e-300) / np.maximum(denominator, 1e-300)
        kernel = np.log(np.clip(ratio, 1e-300, 1e300)).astype(complex)  # (nw, nomega)
        kernel = kernel - 1j * np.pi * (omega[None, :] < np.abs(w)[:, None])
    else:
        kt = KELVIN_TO_EV * float(temperature_k)
        positive = omega > 0.0
        bose = np.zeros_like(omega)
        bose[positive] = 1.0 / np.expm1(np.minimum(omega[positive] / kt, 500.0))
        plus = digamma(0.5 + 1j * (omega[None, :] - w[:, None]) / (2.0 * np.pi * kt))
        minus = digamma(0.5 - 1j * (omega[None, :] + w[:, None]) / (2.0 * np.pi * kt))
        kernel = -2j * np.pi * (bose + 0.5)[None, :] + plus - minus
        kernel[:, ~positive] = 0.0  # a2F(0) = 0: the origin carries no weight (Bose pole)
    # trapezoid over omega: Sigma[..., w] = sum_omega a2f[..., omega] kernel[w, omega] d omega
    weights = np.gradient(omega) if omega.size > 1 else np.ones(1)
    sigma = np.einsum('...o,wo->...w', a2f * weights, kernel)
    sigma = sigma.real + 1j * np.minimum(sigma.imag, 0.0)
    return sigma


def mass_enhancement(
    w_ev: NDArray[np.float64], sigma: NDArray[np.complex128]
) -> tuple[float, float]:
    """``(lambda, Z)`` from the slope of ``Re Sigma`` at the Fermi level: ``lambda = -dRe Sigma/dw``, ``Z = 1/(1 + lambda)``."""
    w = np.asarray(w_ev, dtype=float)
    i0 = int(np.argmin(np.abs(w)))
    lo, hi = max(i0 - 1, 0), min(i0 + 1, w.size - 1)
    slope = (np.real(sigma[..., hi]) - np.real(sigma[..., lo])) / (w[hi] - w[lo])
    lam = float(np.mean(-slope))
    return lam, 1.0 / (1.0 + lam)


def fan_migdal_kernel(
    energy_kq: NDArray[np.float64],
    g2: NDArray[np.float64],
    phonon_energy: NDArray[np.float64],
    temps_ev: Sequence[float],
    occupation_energy_kq: NDArray[np.float64],
    w: NDArray[np.float64],
    eta: float | NDArray[np.float64],
    eps_acoustic: float,
    weight: float = 1.0,
) -> NDArray[np.complex128]:
    """Fan-Migdal self-energy of a block of states from one q (eV).

    Parameters
    ----------
    energy_kq : ndarray ``(nb, nband_kq)``
        Energies of the intermediate states ``m`` at ``k+q`` (eV, same scale as ``w``).
    g2 : ndarray ``(nb, nmode, nband_kq, nband_k)``
        ``|g|^2`` (eV^2) **with** the zero-point factor ``1 / 2 omega``.
    phonon_energy : ndarray ``(nmode,)``
        Phonon energies (eV); modes below ``eps_acoustic`` are skipped.
    temps_ev : sequence of float
        ``k_B T`` values.
    occupation_energy_kq : ndarray ``(ntemp, nb, nband_kq)``
        ``e_m - mu(T)`` of the intermediate states.
    w : ndarray ``(nw,)``
        Energies at which the self-energy is evaluated (same scale as ``energy_kq``).
    eta : float or ndarray ``(nb, nband_kq)``
        Lorentzian width (eV), one value or one per intermediate state (adaptive smearing).
    eps_acoustic : float
    weight : float, optional
        q-point weight.

    Returns
    -------
    ndarray ``(ntemp, nb, nband_k, nw)``, complex
    """
    temps = np.atleast_1d(np.asarray(temps_ev, dtype=float))
    nb, nmode, nkq, nk_bands = g2.shape
    nw = np.asarray(w).size
    out = np.zeros((temps.size, nb, nk_bands, nw), dtype=complex)
    active = np.nonzero(np.asarray(phonon_energy) > eps_acoustic)[0]
    if active.size == 0:
        return out
    eta_arr = np.asarray(eta, dtype=float)
    eta_b = eta_arr[..., None] if eta_arr.ndim == 2 else eta_arr  # (nb, m, 1) or scalar
    gap = np.asarray(w)[None, None, :] - np.asarray(energy_kq)[:, :, None]  # (nb, m, nw)
    for it, kt in enumerate(temps):
        f_m = fermi_occupation(occupation_energy_kq[it], kt)  # (nb, m)
        for nu in active:
            omega = float(phonon_energy[nu])
            n_ph = float(bose_occupation(np.array([omega]), kt)[0])
            absorption = 1.0 / (gap + omega + 1j * eta_b)  # (nb, m, nw)
            emission = 1.0 / (gap - omega + 1j * eta_b)
            kernel = (n_ph + f_m)[..., None] * absorption + (n_ph + 1.0 - f_m)[..., None] * emission
            out[it] += np.einsum('bmn,bmw->bnw', g2[:, nu], kernel, optimize=True)
    return weight * out


def fan_migdal_self_energy(
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
    k_cryst: NDArray[np.float64],
    w_ev: NDArray[np.float64],
    temps_k: Sequence[float] = (300.0,),
    nq_dense: int = 24,
    eta_ev: float = 0.01,
    adaptive_smearing: float | None = None,
    bands: Sequence[int] | None = None,
    window_margin_ev: float = 0.5,
    doping_cm3: float = 0.0,
    scissor_ev=None,
    target_gap_ev=None,
    eps_acoustic_cm: float = 5.0,
    screening_eps: float | None = None,
    screening_eps_dft: float | None = None,
    spin_degeneracy: float = 2.0,
    tau_cryst: NDArray[np.float64] | None = None,
    species: Sequence[str] | None = None,
    orbital_positions: NDArray[np.float64] | None = None,
    ispin: int = 0,
    comm=None,
    allow_missing_long_range: bool = False,
    coupling_source: str = 'epw',
    save_dir: str | None = None,
    pseudo_dir: str | None = None,
    prefix: str | None = None,
    basis: str = 'auto',
    unfold: str = 'full',
    basis_shells=None,
    unfold_tol: float = 1e-6,
    cache_dir: str | None = None,
    cache_tag: str | None = None,
    long_range=None,
) -> dict:
    """Band-resolved Fan-Migdal self-energy at arbitrary k-points from the dense-q vertex.

    Parameters
    ----------
    A, HRs, kpts_cryst, bg, at, alat, cell_volume, coupling_dir, qgrid_coarse, ng
        As :func:`~PAOFLOW.elphon.carrier_scattering.carrier_scattering_dense_q`.
    masses_amu, nelec
        Masses (one per atom) and valence electrons.
    k_cryst : ndarray ``(nk, 3)``
        k-points (crystal coordinates) of the states, e.g. an ARPES path or mesh.
    w_ev : ndarray ``(nw,)``
        Energies relative to the Fermi level of each temperature.
    temps_k : sequence of float, optional
    nq_dense : int, optional
        Dense q-grid (full grid, no symmetry reduction: the k-points are arbitrary).
    eta_ev : float, optional
        Lorentzian width of the denominators (EPW ``degaussw``); the floor of the widths with
        ``adaptive_smearing``.
    adaptive_smearing : float, optional
        Adaptive widths ``a |v_{m k+q}| dq`` per intermediate state (Yates prefactor ``a``,
        0.7 for Gaussians; :mod:`~PAOFLOW.elphon.adaptive_smearing`), floored at ``eta_ev``,
        with the group velocities by finite differences at ``k+q``
        (:func:`~PAOFLOW.elphon.elph_bloch.group_velocities_at`).  Metals need it: a fixed
        width far below the level spacing ``|v_F| dq`` of the q-grid gives a noisy self-energy.
    bands : sequence of int, optional
        Bands ``n`` to resolve; default: every band with a state within
        ``[w_min - margin, w_max + margin]`` of the Fermi level at the given k-points.
    window_margin_ev : float, optional
    doping_cm3, scissor_ev, target_gap_ev, spin_degeneracy
        Carrier statistics of the intermediate states
        (:func:`~PAOFLOW.elphon.carrier_scattering.fermi_levels_for_doping`).
    eps_acoustic_cm, screening_eps, screening_eps_dft, tau_cryst, species, orbital_positions,
    ispin, comm, allow_missing_long_range, coupling_source, save_dir, pseudo_dir, prefix, basis,
    unfold, basis_shells, unfold_tol, cache_dir, cache_tag, long_range
        As :func:`~PAOFLOW.elphon.carrier_scattering.carrier_scattering_dense_q`.

    Returns
    -------
    dict
        ``sigma_ev`` ``(ntemp, nk, nband, nw)`` complex, ``energies_ev`` ``(nk, nband)`` relative
        to the Fermi level of the first temperature (``energies_abs_ev`` on the PAO scale),
        ``bands``, ``w_ev``, ``temps_k``, ``fermi_levels_ev`` ``(ntemp,)``, ``k_cryst``,
        ``eta_ev``, ``adaptive_smearing``, ``nq_dense``, ``metal``, ``coupling_screening``.

    Notes
    -----
    The intermediate states ``k+q`` are diagonalised at every (k, q) with the cached Bloch
    Hamiltonian (:func:`~PAOFLOW.elphon.elph_bloch.bloch_states_at`), so ``k`` need not be
    commensurate with any grid; the sum runs over all bands ``m``.  The work is split over the
    dense q-points between the MPI ranks.
    """
    from mpi4py import MPI

    from ..utils.communication import load_balancing

    if comm is None:
        comm = MPI.COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    masses_amu = np.asarray(masses_amu, dtype=float)
    mass_flat_ry = np.repeat(masses_amu, 3) * AMU_RY
    temps_k = np.atleast_1d(np.asarray(temps_k, dtype=float))
    temps_ev = temps_k * KELVIN_TO_EV
    w = np.asarray(w_ev, dtype=float)
    k_cryst = np.atleast_2d(np.asarray(k_cryst, dtype=float))
    nocc = int(round(nelec / spin_degeneracy))
    check_orbital_dimensions(A, HRs, orbital_positions)
    if screening_eps is None and screening_eps_dft is not None:
        raise ValueError('screening_eps_dft needs screening_eps (the target dielectric constant).')

    vertex = prepare_dense_vertex(
        A, kpts_cryst, bg, at, coupling_dir, qgrid_coarse, None, ng, None,
        source=coupling_source, masses_amu=masses_amu, tau_cryst=tau_cryst,
        orbital_positions=orbital_positions, comm=comm,
        allow_missing_long_range=allow_missing_long_range, long_range=long_range,
        save_dir=save_dir, pseudo_dir=pseudo_dir, prefix=prefix, basis=basis, unfold=unfold,
        basis_shells=basis_shells, unfold_tol=unfold_tol, cache_dir=cache_dir, cache_tag=cache_tag,
    )  # fmt: skip
    g_ReRp, phonon_at_q = vertex['g_ReRp'], vertex['phonon_at_q']
    Nint_p, W_p, Midx_p = vertex['phonon_ws']
    screening = resolve_coupling_screening(
        screening_eps, screening_eps_dft, vertex['epsi'], nelec, cell_volume
    )
    screening_info = {'coupling_screening': 'none'}
    if screening is not None:
        screening_info = screening.summary(bg, alat)
        if rank == 0:
            print(screening_report(screening_info), flush=True)

    electrons = precompute_dense_electrons(
        HRs, at, nq_dense, [eta_ev / RY_TO_EV], None, tuple(ng), ispin=ispin,
        orbital_positions=orbital_positions, comm=comm,
    )  # fmt: skip
    energies_dense = electrons['E'] * RY_TO_EV  # (nqd, nawf), PAO scale
    vbm, cbm = float(energies_dense[:, nocc - 1].max()), float(energies_dense[:, nocc].min())
    metal = cbm <= vbm
    conduction_all = np.arange(energies_dense.shape[1]) >= nocc
    scissor = statistics_scissor(temps_k, cbm - vbm, scissor_ev, target_gap_ev, metal)
    mu = fermi_levels_for_doping(
        energies_dense, temps_k, nelec, doping_cm3, cell_volume, spin_degeneracy,
        conduction_all, scissor,
    )  # fmt: skip

    # states at the requested k-points
    e_k_ry, v_k = bloch_states_at(electrons, k_cryst)
    e_k = e_k_ry * RY_TO_EV
    if bands is None:
        rel = e_k - mu[0]
        inside = (rel >= w.min() - window_margin_ev) & (rel <= w.max() + window_margin_ev)
        bands = np.nonzero(inside.any(axis=0))[0]
    bands = np.asarray(bands, dtype=int)
    if bands.size == 0:
        raise ValueError('no band within the energy window at the given k-points')
    nk, nb = k_cryst.shape[0], bands.size
    V = electrons['V']
    phases_k = vertex_phases_at(electrons, k_cryst)
    v_k_bands = v_k[:, :, bands]

    axes = [np.arange(nq_dense) / nq_dense for _ in range(3)]
    qmesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    nqd = qmesh.shape[0]
    q_weight = 1.0 / nqd
    eps_acoustic = eps_acoustic_cm / EV_TO_CM1
    if rank == 0:
        print(
            '  Fan-Migdal self-energy: %d k-points, %d bands, %d^3 q, eta %.1f meV, %s'
            % (nk, nb, nq_dense, 1e3 * eta_ev, 'metal' if metal else 'gap %.3f eV' % (cbm - vbm)),
            flush=True,
        )
    sigma = np.zeros((temps_k.size, nk, nb, w.size), dtype=complex)
    spacing = grid_spacing(cell_volume, nq_dense) if adaptive_smearing is not None else None
    qstart, qstop = load_balancing(size, rank, nqd)
    for iq in range(qstart, qstop):
        q_cryst = qmesh[iq]
        freq_thz, z = _phonon_modes_at_q(q_cryst, phonon_at_q)
        omega_ry = freq_thz / RY_TO_THZ
        zmass = z / np.sqrt(mass_flat_ry)[None, :]
        if screening is not None:
            zmass = zmass * screening.factor(q_cryst, bg, alat)
        safe = np.where(omega_ry > 1.0e-8, omega_ry, 1.0)
        gn_flat = vertex_ws_coefficients(g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p), electrons)
        lr_q = None
        if long_range is not None:
            lr_q = long_range.vertex_pao(q_cryst, V.shape[1], orbital_positions)
        e_kq_ry, v_kq = bloch_states_at(electrons, k_cryst + q_cryst[None, :])
        coupling = band_vertex(gn_flat, phases_k, v_kq, v_k_bands, zmass, lr_q)  # (nk, nu, m, n)
        g2 = np.abs(coupling) ** 2 / (2.0 * safe)[None, :, None, None] * RY_TO_EV**2
        g2[:, omega_ry <= 1.0e-8] = 0.0
        e_kq = e_kq_ry * RY_TO_EV
        occupation = (e_kq[None] + scissor[:, None, None] * conduction_all[None, None, :]
                      - mu[:, None, None])  # (ntemp, nk, m)  # fmt: skip
        eta = eta_ev
        if adaptive_smearing is not None:
            velocity = group_velocities_at(electrons, k_cryst + q_cryst[None, :], at, alat)
            eta = state_widths(velocity * RY_TO_EV, spacing, adaptive_smearing, eta_ev)  # (nk, m)
        # energies relative to the Fermi level of each temperature on the w scale
        for it in range(temps_k.size):
            sigma[it] += fan_migdal_kernel(
                e_kq - mu[it], g2, omega_ry * RY_TO_EV, [temps_ev[it]], occupation[it : it + 1],
                w, eta, eps_acoustic, weight=q_weight,
            )[0]  # fmt: skip
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, sigma, op=MPI.SUM)
    comm.Barrier()
    vertex['window'].Free()
    release_dense_electrons(electrons)
    return {
        'sigma_ev': sigma,
        'energies_ev': e_k[:, bands] - mu[0],
        'energies_abs_ev': e_k[:, bands],
        'bands': bands,
        'w_ev': w,
        'temps_k': temps_k,
        'fermi_levels_ev': mu,
        'scissor_ev': scissor,
        'k_cryst': k_cryst,
        'eta_ev': float(eta_ev),
        'adaptive_smearing': None if adaptive_smearing is None else float(adaptive_smearing),
        'nq_dense': int(nq_dense),
        'metal': bool(metal),
        **screening_info,
    }


class TabulatedSelfEnergy:
    """Self-energy of :func:`fan_migdal_self_energy` in the protocol of :mod:`PAOFLOW.arpes`.

    Parameters
    ----------
    kpoints : ndarray ``(nk, 3)``
        The momenta the self-energy was computed at, in the units the ARPES driver uses
        (Cartesian, 1/Å) and the same order or any order (rows are matched).
    bands : sequence of int
        Band indices of the second axis of ``sigma``.
    w_grid : ndarray ``(nw,)``
    sigma : ndarray ``(nk, nband, nw)``, complex
        One temperature.
    missing : complex, optional
        Value for bands or k-points not in the table.
    """

    def __init__(self, kpoints, bands, w_grid, sigma, missing: complex = 0.0) -> None:
        self.kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
        self.bands = list(int(b) for b in bands)
        self.w_grid = np.asarray(w_grid, dtype=float)
        self.sigma = np.asarray(sigma, dtype=complex)
        self.missing = complex(missing)

    @classmethod
    def from_result(cls, result: dict, kpoints, temperature_index: int = 0) -> TabulatedSelfEnergy:
        """From the dict of :func:`fan_migdal_self_energy` and the Cartesian momenta it was run at."""
        return cls(kpoints, result['bands'], result['w_ev'], result['sigma_ev'][temperature_index])

    def __call__(self, w, energies=None, kpoints=None, bands=None) -> NDArray[np.complex128]:
        w = np.asarray(w, dtype=float)
        if kpoints is None or bands is None:
            raise ValueError('TabulatedSelfEnergy needs the k-points and band indices of the block')
        kp = np.atleast_2d(np.asarray(kpoints, dtype=float))
        rows = np.array(
            [int(np.argmin(np.linalg.norm(self.kpoints - k[None, :], axis=1))) for k in kp]
        )
        if np.any(np.linalg.norm(self.kpoints[rows] - kp, axis=1) > 1e-6):
            raise ValueError('a k-point of the block is not in the self-energy table')
        out = np.full((kp.shape[0], len(bands), w.size), self.missing, dtype=complex)
        for j, band in enumerate(bands):
            if int(band) in self.bands:
                table = self.sigma[rows, self.bands.index(int(band)), :]  # (nblock, nw_table)
                out[:, j, :] = np.array(
                    [
                        np.interp(w, self.w_grid, t.real) + 1j * np.interp(w, self.w_grid, t.imag)
                        for t in table
                    ]
                )
        return out
