"""Ab initio carrier scattering rates (SERTA / MRTA) and phonon-limited transport.

State-resolved electron-phonon relaxation rates :math:`1/\\tau_{n\\mathbf k}(T)` on
the dense PAO grid, for semiconductors and insulators (states near the band
edges) and for metals (states near the Fermi level):

* :func:`carrier_scattering_dense_q` -- the rates in the self-energy (SERTA) and
  momentum-relaxation (MRTA) relaxation-time approximations, from the EPW+PAO
  interpolation of the coupling (irreducible k-points against the full dense
  q-grid, unfolded by symmetry);
* :func:`rta_transport` -- route (a): the k-resolved conductivity, carrier
  densities, mobilities and Seebeck coefficient on the same dense grid;
* :func:`energy_resolved_rates` -- route (b): the transport-weighted
  :math:`\\tau_{eff}(E, T)` for the Boltzmann module
  (:class:`~PAOFLOW.boltzmann.scattering_channels.EnergyResolvedRate` in
  :meth:`PAOFLOW.PAOFLOW.transport`);
* :func:`fermi_levels_for_doping` -- the chemical potential of a doped (or
  intrinsic) crystal.

Energies in eV, temperatures in K, rates in 1/s, velocities in m/s, densities
in cm^-3 and mobilities in cm^2/(V s) at the interface.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .coupling_screening import resolve_coupling_screening, screening_report
from .do_pao_eph_dense_q import (
    _crystal_point_group,
    _phonon_modes_at_q,
    g_Re_at_q,
    irreducible_mesh,
    prepare_dense_vertex,
)
from .elph_bloch import (
    AMU_RY,
    RY_TO_EV,
    RY_TO_THZ,
    _diagonal_velocities,
    band_vertex,
    precompute_dense_electrons,
    release_dense_electrons,
    vertex_phases,
    vertex_ws_coefficients,
)
from .pao_electrons import check_orbital_dimensions
from .phonon_assisted_absorption import (
    EV_TO_CM1,
    KELVIN_TO_EV,
    bose_occupation,
    fermi_occupation,
    scissor_per_temperature,
)

HBAR_EV_S = 6.582119569e-16
BOHR_M = 0.529177210903e-10
ELEMENTARY_CHARGE = 1.602176634e-19
DELTA_CUTOFF = 6.0
# EPW (transport_legacy.f90): the MRTA weight falls back to 1 for |v|^2 <= 1e-4 Ry a.u.
MRTA_VELOCITY_FLOOR_MS = 1.0e-2 * RY_TO_EV * BOHR_M / HBAR_EV_S


def fermi_levels_for_doping(
    energies_ev: NDArray[np.float64],
    temps_k: Sequence[float],
    nelec: float,
    doping_cm3: float = 0.0,
    cell_volume_bohr3: float | None = None,
    spin_degeneracy: float = 2.0,
    conduction: NDArray[np.bool_] | None = None,
    scissor_ev: Sequence[float] | None = None,
    niter: int = 200,
) -> NDArray[np.float64]:
    """Chemical potential of a doped (or intrinsic) crystal at each temperature.

    Parameters
    ----------
    energies_ev : ndarray, shape ``(nk, nband)``
        Band energies of all bands on a uniform k-grid (eV).
    temps_k : sequence of float
        Temperatures (K).
    nelec : float
        Valence electrons per cell.
    doping_cm3 : float, optional
        Net doping (cm^-3): positive adds electrons (n type), negative removes
        them (p type), 0 gives the intrinsic level.
    cell_volume_bohr3 : float, optional
        Cell volume (bohr^3); required with ``doping_cm3``.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.
    conduction : ndarray of bool, shape ``(nband,)``, optional
        Conduction bands, shifted by ``scissor_ev`` for the statistics.
    scissor_ev : sequence of float, optional
        Scissor of each temperature (eV).
    niter : int, optional
        Bisection steps.

    Returns
    -------
    ndarray, shape ``(ntemp,)``
        :math:`\\mu(T)` (eV), on the scale of ``energies_ev``.

    Notes
    -----
    Solves the charge neutrality

    .. math::

        \\frac{g_s}{N_k}\\sum_{n\\mathbf k} f\\left(\\frac{\\varepsilon_{n\\mathbf k} - \\mu}{k_BT}\\right)
            = N_{el} + N_d\\,\\Omega

    by bisection; for :math:`N_d = 0` it is the intrinsic level.  With
    ``conduction`` given (and integer band filling) the balance is solved as
    :math:`n_e - n_h = N_d`, the holes evaluated as :math:`f(-x)`, so that
    carrier densities far below :math:`10^{-16}` per cell (wide gaps, low
    temperatures) are resolved exactly.
    """
    energies = np.asarray(energies_ev, dtype=float)
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    extra = 0.0
    if doping_cm3:
        if cell_volume_bohr3 is None:
            raise ValueError('doping_cm3 needs cell_volume_bohr3')
        extra = float(doping_cm3) * float(cell_volume_bohr3) * (BOHR_M * 1.0e2) ** 3
    nk = energies.shape[0]
    shifts = np.zeros(temps.size) if scissor_ev is None else np.asarray(scissor_ev, dtype=float)
    # Count carriers, not electrons: with full valence bands sum(f) - N_occ =
    # sum_c f - sum_v (1 - f), and 1 - f = f(-x) is evaluated directly, so carrier
    # densities far below the 1e-16 resolution of the total electron count (wide
    # gaps, low temperatures) stay exact.
    occupied = float(nelec) / spin_degeneracy
    carrier_count = conduction is not None and abs(occupied - round(occupied)) < 1.0e-9
    if carrier_count:
        conduction = np.asarray(conduction, dtype=bool)
        carrier_count = int(np.count_nonzero(~conduction)) == int(round(occupied))
    levels = np.empty(temps.size)
    for it, temp in enumerate(temps):
        shifted = energies if conduction is None else energies + shifts[it] * conduction
        kt = KELVIN_TO_EV * temp
        low, high = float(shifted.min()) - 1.0, float(shifted.max()) + 1.0
        for _ in range(niter):
            mid = 0.5 * (low + high)
            if carrier_count:
                electrons = fermi_occupation(shifted[:, conduction] - mid, kt).sum()
                holes = fermi_occupation(mid - shifted[:, ~conduction], kt).sum()
                excess = spin_degeneracy / nk * (electrons - holes) - extra
            else:
                excess = spin_degeneracy / nk * fermi_occupation(shifted - mid, kt).sum() - (
                    float(nelec) + extra
                )
            if excess > 0.0:
                high = mid
            else:
                low = mid
        levels[it] = 0.5 * (low + high)
    return levels


def scattering_rate_kernel(
    energy_k: NDArray[np.float64],
    energy_kq: NDArray[np.float64],
    velocity_k: NDArray[np.float64],
    velocity_kq: NDArray[np.float64],
    g2: NDArray[np.float64],
    phonon_energy: NDArray[np.float64],
    temps_ev: Sequence[float],
    occupation_energy_kq: NDArray[np.float64],
    smearing: float,
    eps_acoustic: float,
    weight: float = 1.0,
    velocity_floor: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """SERTA and MRTA scattering rates of a block of states from one q.

    Parameters
    ----------
    energy_k, energy_kq : ndarray, shape ``(nb, nband)``
        Band energies at ``k`` and ``k+q`` (eV).
    velocity_k, velocity_kq : ndarray, shape ``(nb, nband, 3)``
        Band velocities (any unit; only their ratios enter).
    g2 : ndarray, shape ``(nb, nmode, nband_kq, nband_k)``
        :math:`|g_{mn\\nu}(\\mathbf k, \\mathbf q)|^2` (eV^2), **with** the
        :math:`1/2\\omega` zero-point factor.
    phonon_energy : ndarray, shape ``(nmode,)``
        Phonon energies at q (eV); modes below ``eps_acoustic`` are skipped.
    temps_ev : sequence of float
        :math:`k_BT` values (eV).
    occupation_energy_kq : ndarray, shape ``(ntemp, nb, nband_kq)``
        :math:`\\varepsilon_{m\\mathbf k+\\mathbf q} - \\mu(T)` used in the Fermi
        occupation of the final states (eV).
    smearing : float
        Gaussian width of the energy-conserving delta (eV), cut at six widths.
    eps_acoustic : float
        Phonon-energy threshold (eV).
    weight : float, optional
        q-point weight (1/N_q for the full grid).
    velocity_floor : float, optional
        States with :math:`|\\mathbf v_{n\\mathbf k}|` at or below it (same unit as
        the velocities) take the SERTA weight in MRTA, as in EPW.

    Returns
    -------
    serta, mrta : ndarray, shape ``(ntemp, nb, nband_k)``
        Rates :math:`1/\\tau_{n\\mathbf k}` (1/s) contributed by this q.

    Notes
    -----
    .. math::

        \\frac{1}{\\tau_{n\\mathbf k}} = \\frac{2\\pi}{\\hbar}\\, w_{\\mathbf q}\\sum_{m\\nu}
            |g_{mn\\nu}|^2\\Big[(n_{\\nu} + f_m)\\,\\delta(\\varepsilon_m - \\varepsilon_n - \\omega_\\nu)
            + (n_\\nu + 1 - f_m)\\,\\delta(\\varepsilon_m - \\varepsilon_n + \\omega_\\nu)\\Big],

    the imaginary part of the Fan-Migdal self-energy (SERTA).  MRTA weights each
    process by :math:`1 - \\mathbf v_{n\\mathbf k}\\cdot\\mathbf v_{m\\mathbf k+\\mathbf q}
    /|\\mathbf v_{n\\mathbf k}|^2` (momentum relaxation).
    """
    temps = np.atleast_1d(np.asarray(temps_ev, dtype=float))
    nb, nband = energy_k.shape
    serta = np.zeros((temps.size, nb, nband))
    mrta = np.zeros((temps.size, nb, nband))
    active = np.nonzero(phonon_energy > eps_acoustic)[0]
    if active.size == 0:
        return serta, mrta
    omega = np.asarray(phonon_energy, dtype=float)[active]
    g2 = g2[:, active]
    cutoff = DELTA_CUTOFF * smearing
    norm = 1.0 / (smearing * np.sqrt(np.pi))
    gap = energy_kq[:, None, :, None] - energy_k[:, None, None, :]  # (nb, 1, m, n)
    x_abs = gap - omega[None, :, None, None]
    x_emi = gap + omega[None, :, None, None]
    d_abs = np.where(np.abs(x_abs) < cutoff, norm * np.exp(-((x_abs / smearing) ** 2)), 0.0)
    d_emi = np.where(np.abs(x_emi) < cutoff, norm * np.exp(-((x_emi / smearing) ** 2)), 0.0)
    if not (d_abs.any() or d_emi.any()):
        return serta, mrta
    v2 = np.einsum('bna,bna->bn', velocity_k, velocity_k)
    projection = np.einsum('bna,bma->bmn', velocity_k, velocity_kq)
    with np.errstate(divide='ignore', invalid='ignore'):
        moving = v2[:, None, :] > velocity_floor**2
        mrta_weight = np.where(moving, 1.0 - projection / v2[:, None, :], 1.0)
    prefactor = 2.0 * np.pi / HBAR_EV_S * weight
    for it, kt in enumerate(temps):
        n_ph = bose_occupation(omega, kt)[None, :, None, None]
        f_m = fermi_occupation(occupation_energy_kq[it], kt)[:, None, :, None]
        kernel = g2 * ((n_ph + f_m) * d_abs + (n_ph + 1.0 - f_m) * d_emi)  # (nb, nu, m, n)
        serta[it] = prefactor * kernel.sum(axis=(1, 2))
        mrta[it] = prefactor * np.einsum('bvmn,bmn->bn', kernel, mrta_weight)
    return serta, mrta


def _window_bands(energies_ry: NDArray, nocc: int, window_ev: float, metal_level_ev: float | None):
    """Bands with a state inside the transport window (band edges or Fermi level)."""
    energies = energies_ry * RY_TO_EV
    if metal_level_ev is not None:
        hit = np.any(np.abs(energies - metal_level_ev) < window_ev, axis=0)
        return np.nonzero(hit)[0]
    vbm, cbm = energies[:, nocc - 1].max(), energies[:, nocc].min()
    valence = [n for n in range(nocc) if energies[:, n].max() > vbm - window_ev]
    conduction = [
        n for n in range(nocc, energies.shape[1]) if energies[:, n].min() < cbm + window_ev
    ]
    return np.array(valence + conduction, dtype=int)


def statistics_scissor(
    temps_k: Sequence[float],
    gap_ev: float,
    scissor_ev: float | Callable[[float], float] | None = None,
    target_gap_ev: float | Callable[[float], float] | None = None,
    metal: bool = False,
) -> NDArray[np.float64]:
    """Scissor of the carrier statistics at each temperature.

    Parameters
    ----------
    temps_k : sequence of float
        Temperatures (K).
    gap_ev : float
        Indirect gap of the Hamiltonian on the dense grid (eV).
    scissor_ev : float or callable, optional
        Shift (eV) or ``T -> Delta(T)``
        (:func:`~PAOFLOW.elphon.phonon_assisted_absorption.varshni_shift`).
    target_gap_ev : float or callable, optional
        Target gap (eV) or ``T -> E_g(T)``
        (:func:`~PAOFLOW.elphon.phonon_assisted_absorption.varshni_gap`).
    metal : bool, optional
        No scissor for metals.

    Returns
    -------
    ndarray, shape ``(ntemp,)``
        The conduction-band shift per temperature (eV); zeros for a metal or when
        nothing is given.
    """
    temps = np.atleast_1d(np.asarray(temps_k, dtype=float))
    if metal or (scissor_ev is None and target_gap_ev is None):
        return np.zeros(temps.size)
    return scissor_per_temperature(temps, gap_ev, scissor_ev, target_gap_ev)


def carrier_scattering_dense_q(
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
    nk_dense: int = 24,
    nq_dense: int = 24,
    temps_k: Sequence[float] = (300.0,),
    doping_cm3: float = 0.0,
    window_ev: float = 0.3,
    smearing_ev: float = 0.01,
    scissor_ev: float | Callable[[float], float] | None = None,
    target_gap_ev: float | Callable[[float], float] | None = None,
    eps_acoustic_cm: float = 5.0,
    screening_eps: float | None = None,
    screening_eps_dft: float | None = None,
    spin_degeneracy: float = 2.0,
    sym_rots: NDArray[np.int_] | None = None,
    tau_cryst: NDArray[np.float64] | None = None,
    species: Sequence[str] | None = None,
    orbital_positions: NDArray[np.float64] | None = None,
    ispin: int = 0,
    kblock: int = 256,
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
) -> dict:
    """State-resolved electron-phonon scattering rates on the dense PAO grid.

    Parameters
    ----------
    A, HRs, kpts_cryst, bg, at, alat, cell_volume, coupling_dir, qgrid_coarse, ng
        As :func:`~PAOFLOW.elphon.phonon_assisted_absorption.phonon_assisted_absorption_dense_q`
        (PAO projections and Hamiltonian, lattice, EPW directory, coarse grids).
    masses_amu : ndarray, shape ``(nat,)``
        Atomic masses (amu), one per atom.
    nelec : float
        Valence electrons per cell.
    nk_dense, nq_dense : int, optional
        Dense k- and q-grids per axis (``nk_dense % nq_dense == 0``).
    temps_k : sequence of float, optional
        Temperatures (K).
    doping_cm3 : float, optional
        Net doping (cm^-3, positive n type, negative p type, 0 intrinsic).
    window_ev : float, optional
        Transport window: states within ``window_ev`` of the band edges
        (semiconductors and insulators) or of the Fermi level (metals).
    smearing_ev : float, optional
        Gaussian width of the energy-conserving delta (eV).
    scissor_ev : float or callable, optional
        Scissor of the carrier statistics: a shift (eV) or ``T -> Delta(T)``
        relative to the Hamiltonian's own gap
        (:func:`~PAOFLOW.elphon.phonon_assisted_absorption.varshni_shift`, for a
        ``HRs`` that already has the 0 K gap).
    target_gap_ev : float or callable, optional
        Scissor target gap (eV, or ``T -> E_g(T)``) for the carrier statistics
        (:func:`~PAOFLOW.elphon.phonon_assisted_absorption.scissor_per_temperature`);
        the rates involve intraband energy differences only and are unchanged.
    eps_acoustic_cm : float, optional
        Phonons below this energy (cm^-1) are skipped.
    screening_eps, screening_eps_dft : float, optional
        Poncé-Resta screening correction of the coupling, as in
        :func:`~PAOFLOW.elphon.phonon_assisted_absorption.phonon_assisted_absorption_dense_q`
        (:mod:`~PAOFLOW.elphon.coupling_screening`): the target dielectric
        constant (e.g. 11.94 for Si, which lowers the Si mobilities by about
        9-12 % according to Poncé *et al.* 2018) and that of the DFPT coupling
        (default: the ``.epb`` tensor).  ``None`` keeps the DFPT screening.
    spin_degeneracy : float, optional
        2 without spin-orbit coupling, 1 with it.  The band velocities are the
        group velocities of the PAO Hamiltonian, which contains the non-local
        pseudopotential (Hellmann-Feynman); the non-local velocity correction
        belongs to interband (optical) dipoles only.
    sym_rots, tau_cryst, species : optional
        Point group (``read_nscf`` ``'s_cryst'``), atomic positions and labels:
        irreducible q-points and star-averaged k (see Notes).
    orbital_positions : ndarray, optional
        PAO orbital centres (pair-resolved Wigner-Seitz sums, velocity operator).
    ispin, kblock, comm, allow_missing_long_range : optional
        Spin channel, k-points per task, MPI communicator, polar guard.
    coupling_source : {'epw', 'dfpt'}, optional
        ``'epw'`` reads EPW's ``.epb`` files from ``coupling_dir``; ``'dfpt'`` computes the
        coarse vertex from the ``ph.x`` outputs in ``coupling_dir`` and the nscf
        wavefunctions of ``save_dir`` (:func:`~PAOFLOW.elphon.do_pao_eph_dense_q.prepare_dense_vertex`).
    save_dir, pseudo_dir, prefix, basis, unfold, basis_shells, unfold_tol, cache_dir, cache_tag
        ``coupling_source='dfpt'`` and cache options of
        :func:`~PAOFLOW.elphon.do_pao_eph_dense_q.prepare_dense_vertex`.

    Returns
    -------
    dict
        ``energies_ev`` ``(nkd, nband)`` (window bands, PAO energy scale),
        ``velocities_ms`` ``(nkd, nband, 3)``, ``in_window`` ``(nkd, nband)``,
        ``carrier`` ``(nband,)`` (+1 electron/conduction, -1 hole/valence, 0
        metal), ``rates_serta`` and ``rates_mrta`` ``(ntemp, nkd, nband)`` (1/s,
        zero outside the window), ``fermi_levels_ev``, ``scissor_ev``,
        ``n_e_cm3``, ``n_h_cm3`` ``(ntemp,)``, ``temps_k``, ``doping_cm3``,
        ``bands``, ``vbm_ev``, ``cbm_ev``, ``metal``, ``cell_volume_bohr3``,
        ``spin_degeneracy``, ``nk_dense``, ``nq_dense``, ``coupling_screening``
        (``'none'`` or ``'ponce-resta'`` with the ``screening_*`` constants).

    Notes
    -----
    With the point group :math:`G` (and time reversal), the rates are summed over
    the **irreducible** dense q-points with their star weights :math:`w_q` for
    every k that carries window states, and then averaged over the star of k:

    .. math::

        \\frac{1}{\\tau_{n\\mathbf k}} = \\sum_{\\mathbf q} F_n(\\mathbf k, \\mathbf q)
            = \\frac{1}{|G|}\\sum_{S\\in G} A_n(S^{-1}\\mathbf k), \\qquad
        A_n(\\mathbf k) = \\sum_{\\mathbf q \\in \\mathrm{IBZ}} w_{\\mathbf q}
            F_n(\\mathbf k, \\mathbf q),

    exact because :math:`F(S\\mathbf k, S\\mathbf q) = F(\\mathbf k, \\mathbf q)`
    for the scalar SERTA and MRTA kernels.  Pairs whose ``k+q`` has no state
    within one phonon energy (plus the delta tail) of the window are skipped, and
    the rates are averaged over degenerate bands.  The chemical potential follows
    from charge neutrality with the doping (:func:`fermi_levels_for_doping`).
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
    temps_k = np.atleast_1d(np.asarray(temps_k, dtype=float))
    temps_ev = temps_k * KELVIN_TO_EV
    nocc = int(round(nelec / spin_degeneracy))
    check_orbital_dimensions(A, HRs, orbital_positions)
    if screening_eps is None and screening_eps_dft is not None:
        raise ValueError('screening_eps_dft needs screening_eps (the target dielectric constant).')

    vertex = prepare_dense_vertex(
        A, kpts_cryst, bg, at, coupling_dir, qgrid_coarse, None, ng, None,
        source=coupling_source, masses_amu=masses_amu, tau_cryst=tau_cryst,
        orbital_positions=orbital_positions, comm=comm,
        allow_missing_long_range=allow_missing_long_range,
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

    def keep(energies_ry):
        e = energies_ry * RY_TO_EV
        metal = e[:, nocc].min() <= e[:, nocc - 1].max()
        level = None
        if metal:
            level = float(fermi_levels_for_doping(e, [temps_k.min()], nelec, 0.0, cell_volume)[0])
        return _window_bands(energies_ry, nocc, window_ev, level)

    electrons = precompute_dense_electrons(
        HRs, at, nk_dense, [smearing_ev / RY_TO_EV], None, tuple(ng), ispin=ispin,
        orbital_positions=orbital_positions, velocities=True, alat=alat, keep_bands=keep,
        comm=comm,
    )  # fmt: skip
    # band velocities (with the non-local term): k-points shared between ranks
    vstart, vstop = load_balancing(size, rank, electrons['K'].shape[0])
    velocity_ry = _diagonal_velocities(electrons, kblock, k_range=(vstart, vstop))
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, velocity_ry, op=MPI.SUM)
    bands = electrons['bands']
    energies_all = electrons['E'] * RY_TO_EV  # (nkd, nawf)
    energies = energies_all[:, bands]
    velocity_ms = velocity_ry * RY_TO_EV * BOHR_M / HBAR_EV_S
    nkd, nband = energies.shape
    vbm, cbm = float(energies_all[:, nocc - 1].max()), float(energies_all[:, nocc].min())
    metal = cbm <= vbm
    conduction_all = np.arange(energies_all.shape[1]) >= nocc
    conduction = conduction_all[bands]

    # carrier statistics (scissor on the conduction bands only)
    scissor = statistics_scissor(temps_k, cbm - vbm, scissor_ev, target_gap_ev, metal)
    mu = fermi_levels_for_doping(
        energies_all, temps_k, nelec, doping_cm3, cell_volume, spin_degeneracy,
        conduction_all, scissor,
    )  # fmt: skip
    omega_cm3 = cell_volume * (BOHR_M * 1.0e2) ** 3
    n_e = np.zeros(temps_k.size)
    n_h = np.zeros(temps_k.size)
    for it in range(temps_k.size):
        occ = fermi_occupation(energies_all + scissor[it] * conduction_all - mu[it], temps_ev[it])
        n_e[it] = spin_degeneracy * occ[:, conduction_all].sum() / (nkd * omega_cm3)
        hole = fermi_occupation(mu[it] - energies_all[:, ~conduction_all], temps_ev[it])  # 1 - f
        n_h[it] = spin_degeneracy * hole.sum() / (nkd * omega_cm3)

    # transport window
    if metal:
        in_window = np.abs(energies - mu.min()) < window_ev
        carrier = np.zeros(nband, dtype=int)
    else:
        in_window = np.where(
            conduction[None, :], energies < cbm + window_ev, energies > vbm - window_ev
        )
        carrier = np.where(conduction, 1, -1)
    has_state = in_window.any(axis=1)
    # energy distance of every state from the window: final states k+q are only
    # reachable within one phonon energy (plus the delta tail) of it
    if metal:
        outside = np.maximum(np.abs(energies - mu.min()) - window_ev, 0.0)
    else:
        outside = np.where(
            conduction[None, :],
            np.maximum(energies - (cbm + window_ev), 0.0),
            np.maximum(vbm - window_ev - energies, 0.0),
        )
    closest = outside.min(axis=1)  # (nkd,)

    # Irreducible q (star weights) against every k with window states, then the
    # star average over k, exact for the scalar rates (see Notes).
    window_k = np.nonzero(has_state)[0]
    if sym_rots is not None:
        rots = sym_rots
        if tau_cryst is not None and species is not None:
            rots = _crystal_point_group(sym_rots, tau_cryst, species)
        qmesh, q_reps, q_weights, _ = irreducible_mesh(nq_dense, rots, at, bg)
        qmesh, q_weights = qmesh[q_reps], q_weights / qmesh.shape[0]
        _, _, _, k_star = irreducible_mesh(nk_dense, rots, at, bg)
    else:
        axes = [np.arange(nq_dense) / nq_dense for _ in range(3)]
        qmesh = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        q_weights = np.full(qmesh.shape[0], 1.0 / qmesh.shape[0])
        k_star = None
    nqd = qmesh.shape[0]
    if rank == 0:
        print(
            '  carrier scattering: %d window bands, %d k with window states (of %d^3), '
            '%d q (of %d^3), %s' % (nband, window_k.size, nk_dense, nqd, nq_dense,
                                    'metal' if metal else 'gap %.3f eV' % (cbm - vbm)),
            flush=True,
        )  # fmt: skip

    row = -np.ones(nkd, dtype=int)  # grid index -> row of the window-k arrays
    row[window_k] = np.arange(window_k.size)
    kblocks = [window_k[s0 : s0 + kblock] for s0 in range(0, window_k.size, kblock)]
    tasks = [(iq, ib) for iq in range(nqd) for ib in range(len(kblocks))]
    tstart, tstop = load_balancing(size, rank, len(tasks))
    serta = np.zeros((temps_k.size, window_k.size, nband))
    mrta = np.zeros((temps_k.size, window_k.size, nband))
    grid_index = np.round(electrons['K'] * nk_dense).astype(int) % nk_dense
    V = electrons['V']
    eps_acoustic = eps_acoustic_cm / EV_TO_CM1
    stat_shift = scissor[:, None, None] * conduction[None, None, :] - mu[:, None, None]
    current_q = None
    for iq, ib in tasks[tstart:tstop]:
        if iq != current_q:
            current_q = iq
            q_cryst = qmesh[iq]
            freq_thz, z = _phonon_modes_at_q(q_cryst, phonon_at_q)
            omega_ry = freq_thz / RY_TO_THZ
            reach = max(float(omega_ry.max()) * RY_TO_EV, 0.0) + DELTA_CUTOFF * smearing_ev
            shift = np.round(q_cryst * nk_dense).astype(int)
            gn_flat = None  # built on the first block with reachable final states
        kk = kblocks[ib]
        kq = np.ravel_multi_index(((grid_index[kk] + shift) % nk_dense).T, (nk_dense,) * 3)
        reachable = closest[kq] < reach
        if not reachable.any():
            continue
        kk, kq = kk[reachable], kq[reachable]
        if gn_flat is None:
            zmass = z / np.sqrt(mass_flat_ry)[None, :]
            if screening is not None:  # g is linear in zmass
                zmass = zmass * screening.factor(q_cryst, bg, alat)
            safe = np.where(omega_ry > 1.0e-8, omega_ry, 1.0)
            gn_flat = vertex_ws_coefficients(
                g_Re_at_q(g_ReRp, q_cryst, Nint_p, W_p, Midx_p), electrons
            )
        coupling = band_vertex(
            gn_flat, vertex_phases(electrons, kk), V[kq], V[kk], zmass
        )  # (nb, nu, m, n)
        g2 = np.abs(coupling) ** 2 / (2.0 * safe)[None, :, None, None] * RY_TO_EV**2
        g2[:, omega_ry <= 1.0e-8] = 0.0
        occupation_kq = energies[kq][None] + stat_shift  # (ntemp, nb, m)
        s_part, m_part = scattering_rate_kernel(
            energies[kk], energies[kq], velocity_ms[kk], velocity_ms[kq], g2,
            omega_ry * RY_TO_EV, temps_ev, occupation_kq, smearing_ev, eps_acoustic,
            weight=q_weights[iq], velocity_floor=MRTA_VELOCITY_FLOOR_MS,
        )  # fmt: skip
        serta[:, row[kk]] += s_part
        mrta[:, row[kk]] += m_part
    if size > 1:
        comm.Allreduce(MPI.IN_PLACE, serta, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, mrta, op=MPI.SUM)
    comm.Barrier()
    vertex['window'].Free()
    release_dense_electrons(electrons)

    e_window = energies[window_k]
    same = (np.abs(e_window[:, :, None] - e_window[:, None, :]) < 1.0e-5).astype(float)
    full = {}
    for name, rates in (('rates_serta', serta), ('rates_mrta', mrta)):
        if k_star is not None:
            orbit = k_star[window_k]
            total = np.zeros((temps_k.size, orbit.max() + 1, nband))
            np.add.at(total, (slice(None), orbit), rates)
            count = np.bincount(orbit, minlength=orbit.max() + 1)
            rates = total[:, orbit] / count[orbit][None, :, None]
        rates = np.einsum('knm,tkm->tkn', same, rates) / same.sum(axis=2)[None]
        unfolded = np.zeros((temps_k.size, nkd, nband))
        unfolded[:, window_k] = rates
        full[name] = np.where(in_window[None], unfolded, 0.0)
    return {
        'energies_ev': energies,
        'velocities_ms': velocity_ms,
        'in_window': in_window,
        'carrier': carrier,
        'rates_serta': full['rates_serta'],
        'rates_mrta': full['rates_mrta'],
        'fermi_levels_ev': mu,
        'scissor_ev': scissor,
        'n_e_cm3': n_e,
        'n_h_cm3': n_h,
        'temps_k': temps_k,
        'doping_cm3': float(doping_cm3),
        'bands': bands,
        'vbm_ev': vbm,
        'cbm_ev': cbm,
        'metal': bool(metal),
        'cell_volume_bohr3': float(cell_volume),
        'spin_degeneracy': float(spin_degeneracy),
        'nk_dense': nk_dense,
        'nq_dense': nq_dense,
        **screening_info,
    }


def rta_transport(results: dict, kind: str = 'mrta') -> dict:
    """k-resolved relaxation-time transport on the dense e-ph grid (route a).

    Parameters
    ----------
    results : dict
        Output of :func:`carrier_scattering_dense_q`.
    kind : {'mrta', 'serta'}, optional
        Which rates to use.

    Returns
    -------
    dict
        ``temps_k``, ``sigma`` ``(ntemp, 3, 3)`` (S/m), ``sigma_avg``, ``sigma_e``,
        ``sigma_h`` (trace/3 of the total and of the conduction and valence bands,
        S/m; a metal has ``sigma_avg`` only), ``n_e_cm3``, ``n_h_cm3``, ``mobility_e``, ``mobility_h``
        (cm^2/(V s), NaN without carriers of that kind), ``seebeck_uv_k``
        (trace form, microvolt/K) and ``kind``.

    Notes
    -----
    .. math::

        \\sigma_{\\alpha\\beta} = \\frac{e^2 g_s}{\\Omega N_k}\\sum_{n\\mathbf k}
            v^\\alpha_{n\\mathbf k} v^\\beta_{n\\mathbf k}\\,\\tau_{n\\mathbf k}
            \\left(-\\frac{\\partial f}{\\partial\\varepsilon}\\right),
        \\qquad \\mu_{e,h} = \\frac{\\sigma_{e,h}}{e\\, n_{e,h}},

    over the window states, conduction (electrons) and valence (holes) bands
    separately; the Seebeck coefficient is
    :math:`S = -\\mathrm{Tr}\\,L_1/(eT\\,\\mathrm{Tr}\\,L_0)` with the energy moment
    :math:`(\\varepsilon - \\mu)` in :math:`L_1`.
    """
    rates = np.asarray(results['rates_%s' % kind])
    temps = np.asarray(results['temps_k'])
    energies = np.asarray(results['energies_ev'])
    velocity = np.asarray(results['velocities_ms'])
    window = np.asarray(results['in_window'])
    carrier = np.asarray(results['carrier'])
    scissor = np.asarray(results['scissor_ev'])
    mu = np.asarray(results['fermi_levels_ev'])
    nkd = energies.shape[0]
    volume_m3 = float(results['cell_volume_bohr3']) * BOHR_M**3
    prefactor = ELEMENTARY_CHARGE**2 * float(results['spin_degeneracy']) / (volume_m3 * nkd)
    conduction, valence = carrier > 0, carrier < 0
    out = {
        'temps_k': temps,
        'sigma': np.zeros((temps.size, 3, 3)),
        'sigma_avg': np.zeros(temps.size),
        'sigma_e': np.zeros(temps.size),
        'sigma_h': np.zeros(temps.size),
        'n_e_cm3': np.asarray(results['n_e_cm3']),
        'n_h_cm3': np.asarray(results['n_h_cm3']),
        'mobility_e': np.full(temps.size, np.nan),
        'mobility_h': np.full(temps.size, np.nan),
        'seebeck_uv_k': np.zeros(temps.size),
        'kind': kind,
    }
    for it, temp in enumerate(temps):
        kt = KELVIN_TO_EV * temp
        x = energies + scissor[it] * conduction[None, :] - mu[it]
        f = fermi_occupation(x, kt)
        # f (1 - f) with 1 - f = f(-x): no cancellation for deep valence states
        minus_df = f * fermi_occupation(-x, kt) / (kt * ELEMENTARY_CHARGE)  # 1/J
        with np.errstate(divide='ignore'):
            tau = np.where(
                window & (rates[it] > 0.0), 1.0 / np.where(rates[it] > 0, rates[it], 1.0), 0.0
            )
        weight = tau * minus_df  # (nkd, nband)
        sigma = prefactor * np.einsum('kna,knb,kn->ab', velocity, velocity, weight)
        out['sigma'][it] = sigma
        v2 = np.einsum('kna,kna->kn', velocity, velocity) / 3.0
        sig_e = prefactor * (v2 * weight)[:, conduction].sum()
        sig_h = prefactor * (v2 * weight)[:, valence].sum()
        out['sigma_avg'][it] = np.trace(sigma) / 3.0
        out['sigma_e'][it], out['sigma_h'][it] = sig_e, sig_h
        if out['n_e_cm3'][it] > 0.0 and conduction.any():
            out['mobility_e'][it] = sig_e / (ELEMENTARY_CHARGE * out['n_e_cm3'][it] * 1.0e6) * 1.0e4
        if out['n_h_cm3'][it] > 0.0 and valence.any():
            out['mobility_h'][it] = sig_h / (ELEMENTARY_CHARGE * out['n_h_cm3'][it] * 1.0e6) * 1.0e4
        l0 = (v2 * weight).sum()
        l1 = (v2 * weight * x).sum()
        out['seebeck_uv_k'][it] = -l1 / (temp * l0) * 1.0e6 if l0 > 0.0 else 0.0
    return out


def energy_resolved_rates(results: dict, kind: str = 'mrta', bin_ev: float = 0.005) -> dict:
    """Transport-weighted relaxation rate versus energy, for the Boltzmann module (route b).

    Parameters
    ----------
    results : dict
        Output of :func:`carrier_scattering_dense_q`.
    kind : {'mrta', 'serta'}, optional
        Which rates to use.
    bin_ev : float, optional
        Energy bin (eV).

    Returns
    -------
    dict
        ``energy_ev`` ``(nbin,)`` on the PAO energy scale (the scale of
        PAOFLOW's ``E_k``), ``temps_k`` and ``rate_per_s`` ``(ntemp, nbin)``,
        the input of
        :class:`~PAOFLOW.boltzmann.scattering_channels.EnergyResolvedRate`.

    Notes
    -----
    .. math::

        \\frac{1}{\\tau_{eff}(E, T)} = \\left[\\frac{\\sum_{n\\mathbf k} v^2_{n\\mathbf k}\\,
            \\tau_{n\\mathbf k}\\,\\delta(E - \\varepsilon_{n\\mathbf k})}
            {\\sum_{n\\mathbf k} v^2_{n\\mathbf k}\\,\\delta(E - \\varepsilon_{n\\mathbf k})}\\right]^{-1}:

    the energy-only relaxation time that reproduces the trace of the k-resolved
    conductivity.  Empty bins (e.g. the gap) are interpolated.
    """
    rates = np.asarray(results['rates_%s' % kind])
    energies = np.asarray(results['energies_ev'])
    velocity = np.asarray(results['velocities_ms'])
    window = np.asarray(results['in_window'])
    temps = np.asarray(results['temps_k'])
    e_states = energies[window]
    v2 = np.einsum('kna,kna->kn', velocity, velocity)[window]
    edges = np.arange(e_states.min(), e_states.max() + bin_ev, bin_ev)
    centres = 0.5 * (edges[1:] + edges[:-1])
    which = np.clip(np.digitize(e_states, edges) - 1, 0, centres.size - 1)
    denom = np.bincount(which, weights=v2, minlength=centres.size)
    filled = denom > 0.0
    rate_per_s = np.zeros((temps.size, centres.size))
    for it in range(temps.size):
        r = rates[it][window]
        tau = np.where(r > 0.0, 1.0 / np.where(r > 0.0, r, 1.0), 0.0)
        num = np.bincount(which, weights=v2 * tau, minlength=centres.size)
        tau_eff = np.interp(centres, centres[filled], num[filled] / denom[filled])
        rate_per_s[it] = np.where(tau_eff > 0.0, 1.0 / np.where(tau_eff > 0.0, tau_eff, 1.0), 0.0)
    return {'energy_ev': centres, 'temps_k': temps, 'rate_per_s': rate_per_s, 'kind': kind}


def write_mobility_outputs(results: dict, outdir: str) -> list[str]:
    """Write mobilities, scattering rates and the Boltzmann-channel input.

    Parameters
    ----------
    results : dict
        Output of :func:`carrier_scattering_dense_q`.
    outdir : str
        Output directory (created if missing).

    Returns
    -------
    list of str
        ``mobility.dat`` (T, mu_F, n_e, n_h, SERTA and MRTA mobilities and
        conductivities), ``rates_<T>K.dat`` (energy, SERTA and MRTA rates of the
        window states), ``scattering_rates.npz`` (energy-resolved SERTA and MRTA
        rates, the input of ``EnergyResolvedRate.from_npz``) and
        ``carrier_scattering.npz`` (all state-resolved arrays).
    """
    os.makedirs(outdir, exist_ok=True)
    written = []
    serta = rta_transport(results, 'serta')
    mrta = rta_transport(results, 'mrta')
    table = np.column_stack([
        results['temps_k'], results['fermi_levels_ev'], results['n_e_cm3'], results['n_h_cm3'],
        serta['mobility_e'], serta['mobility_h'], mrta['mobility_e'], mrta['mobility_h'],
        serta['sigma_avg'], mrta['sigma_avg'],
        mrta['seebeck_uv_k'],
    ])  # fmt: skip
    path = os.path.join(outdir, 'mobility.dat')
    np.savetxt(
        path, table, fmt='%14.6e',
        header='T (K), mu_F (eV), n_e, n_h (cm^-3), mobility e/h SERTA, e/h MRTA (cm^2/Vs), '
        'sigma SERTA, MRTA (S/m), Seebeck MRTA (uV/K)',
    )  # fmt: skip
    written.append(path)
    window = np.asarray(results['in_window'])
    for it, temp in enumerate(results['temps_k']):
        path = os.path.join(outdir, 'rates_%.1fK.dat' % temp)
        data = np.column_stack([
            results['energies_ev'][window], results['rates_serta'][it][window],
            results['rates_mrta'][it][window],
        ])  # fmt: skip
        np.savetxt(path, data[np.argsort(data[:, 0])], fmt='%14.6e',
                   header='E (eV, PAO scale), 1/tau SERTA, 1/tau MRTA (1/s)')  # fmt: skip
        written.append(path)
    channels = {}
    for kind in ('serta', 'mrta'):
        er = energy_resolved_rates(results, kind)
        channels['energy_ev'] = er['energy_ev']
        channels['rate_%s' % kind] = er['rate_per_s']
    path = os.path.join(outdir, 'scattering_rates.npz')
    np.savez(
        path, temps_k=results['temps_k'], fermi_levels_ev=results['fermi_levels_ev'],
        vbm_ev=results['vbm_ev'], cbm_ev=results['cbm_ev'], metal=bool(results.get('metal', False)),
        **channels,
    )  # fmt: skip
    written.append(path)
    path = os.path.join(outdir, 'carrier_scattering.npz')
    np.savez(path, **{k: np.asarray(v) for k, v in results.items()})
    written.append(path)
    return written
