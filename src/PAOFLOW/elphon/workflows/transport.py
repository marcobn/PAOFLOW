"""Transport workflow: phonon-limited carrier mobility (semiconductors) and metal resistivity.

Wraps :func:`~PAOFLOW.elphon.carrier_scattering.carrier_scattering_dense_q` and the
scattering-channel route of :mod:`PAOFLOW.response.drude` /
:mod:`PAOFLOW.response.emissivity` with the inputs of an :class:`~.system.ElphonSystem`.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from ..carrier_scattering import carrier_scattering_dense_q, rta_transport, write_mobility_outputs
from ..coupling_screening import screening_report
from ..phonon_assisted_absorption import (
    read_direct_dielectric,
    varshni_gap,
    varshni_shift,
    write_emissivity_outputs,
)
from .system import Electrons, ElphonSystem, is_root


def statistics_scissor(
    target_gap_ev: float | None = None, varshni: Sequence[float] | None = None
) -> tuple:
    """Scissor of the carrier statistics: ``(scissor_ev, target_gap_ev, mode)``.

    Parameters
    ----------
    target_gap_ev : float, optional
        Target indirect gap (0 K value with ``varshni``).
    varshni : (alpha, beta), optional
        ``E_g(T) = E0 - alpha T^2 / (T + beta)``.

    Returns
    -------
    tuple
        ``'target'``, ``'relative'`` (only the Varshni shift on top of the Hamiltonian's
        gap: eACBN0 electrons, or the PBE gap at 0 K) or ``'none'``.
    """
    if target_gap_ev is not None:
        target = varshni_gap(target_gap_ev, *varshni) if varshni else target_gap_ev
        return None, target, 'target'
    if varshni:
        return varshni_shift(*varshni), None, 'relative'
    return None, None, 'none'


def carrier_mobility(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    nk: int,
    nq: int,
    temps_k: Sequence[float] = (300.0,),
    doping_cm3: float = 0.0,
    window_ev: float = 0.3,
    smearing_ev: float = 0.01,
    adaptive_smearing: float | None = None,
    scissor_ev=None,
    target_gap_ev=None,
    scissor_mode: str = 'none',
    screening_eps: float | None = None,
    screening_eps_dft: float | None = None,
    comm=None,
) -> dict:
    """SERTA/MRTA scattering rates of the window states and the carrier mobilities.

    Parameters
    ----------
    system, electrons
        The system and its electrons (``electrons_save`` for eACBN0 electrons).
    nk, nq : int
        Dense electron and phonon grids per axis (``nk % nq == 0``; mobilities need
        >= 48^3).
    temps_k : sequence of float
        Temperatures (K).
    doping_cm3 : float
        Net doping (> 0 n type, < 0 p type, 0 intrinsic).
    window_ev : float
        States within this window of the band edges (EPW ``fsthick``).
    smearing_ev : float
        Delta smearing (EPW ``degaussw``); the floor with ``adaptive_smearing``.
    adaptive_smearing : float, optional
        Yates prefactor of the per-final-state widths ``a |v| dq``.
    scissor_ev, target_gap_ev, scissor_mode
        Of :func:`statistics_scissor` (the rates are unchanged).
    screening_eps, screening_eps_dft : float, optional
        Poncé screening correction of the coupling.
    comm : MPI communicator, optional

    Returns
    -------
    dict
        The backend output with the provenance of the electrons and ``scissor_mode``.

    Raises
    ------
    ValueError
        If ``nk`` is not a multiple of ``nq``.
    """
    if nk % nq:
        raise ValueError('nk (%d) must be a multiple of nq (%d)' % (nk, nq))
    out = carrier_scattering_dense_q(
        *system.positional_arguments(electrons),
        nelec=system.nelec,
        nk_dense=nk,
        nq_dense=nq,
        temps_k=list(temps_k),
        doping_cm3=doping_cm3,
        window_ev=window_ev,
        smearing_ev=smearing_ev,
        adaptive_smearing=adaptive_smearing,
        min_smearing_ev=smearing_ev,
        scissor_ev=scissor_ev,
        target_gap_ev=target_gap_ev,
        screening_eps=screening_eps,
        screening_eps_dft=screening_eps_dft,
        spin_degeneracy=system.degeneracy(electrons),
        comm=comm,
        **system.coupling_arguments(electrons),
    )
    out.update(electrons.provenance)
    out['scissor_mode'] = scissor_mode
    out['doping_cm3'] = doping_cm3
    return out


def mobility_label(
    nk: int, nq: int, smearing_ev: float, adaptive_smearing: float | None, window_ev: float
) -> str:
    """``'k 48^3, q 48^3, smearing 0.010 eV, window 0.30 eV'``."""
    smearing = (
        '%.3f eV' % smearing_ev
        if adaptive_smearing is None
        else 'adaptive a = %.2f (floor %.1f meV)' % (adaptive_smearing, 1e3 * smearing_ev)
    )
    return 'k %d^3, q %d^3, smearing %s, window %.2f eV' % (nk, nq, smearing, window_ev)


def write_mobility(out: dict, outdir: str, *, label: str = '', comm=None) -> list[str]:
    """Print the mobilities and write ``mobility.dat``, ``rates_<T>K.dat``, the npz files (rank 0).

    Parameters
    ----------
    out : dict
        Of :func:`carrier_mobility`.
    outdir : str
        Output directory (``carrier_scattering.npz`` and ``scattering_rates.npz`` for
        PAOFLOW's Boltzmann transport, ``EnergyResolvedRate.from_npz``).
    label : str, optional
        :func:`mobility_label`.
    comm : MPI communicator, optional

    Returns
    -------
    list of str
        Written files (empty on the other ranks).
    """
    if not is_root(comm):
        return []
    print('PAOFLOW carrier transport%s:' % (' (%s)' % label if label else ''))
    if out['metal']:
        print('  metal: conductivity only')
    else:
        print(
            '  %s gap %.3f eV, doping %.3e cm^-3'
            % (out['electrons_dft'], out['cbm_ev'] - out['vbm_ev'], out['doping_cm3'])
        )
    if out['electrons_save']:
        print(
            '  band energies and eigenvectors from %s; coupling from %s'
            % (out['electrons_save'], out['coupling_save'])
        )
    if out['scissor_mode'] == 'relative':
        print('  carrier statistics: Varshni shift relative to the Hamiltonian gap')
    print(screening_report(out))
    serta, mrta = rta_transport(out, 'serta'), rta_transport(out, 'mrta')
    print(
        '    T (K)  mu_F (eV)   n_e (cm^-3)   n_h (cm^-3)   mobility e / h (cm^2/Vs): '
        'SERTA             MRTA'
    )
    for it, temp in enumerate(out['temps_k']):
        print(
            '  %7.1f  %9.4f  %12.4e  %12.4e  %28.1f / %-8.1f %8.1f / %-8.1f'
            % (
                temp,
                out['fermi_levels_ev'][it],
                out['n_e_cm3'][it],
                out['n_h_cm3'][it],
                serta['mobility_e'][it],
                serta['mobility_h'][it],
                mrta['mobility_e'][it],
                mrta['mobility_h'][it],
            )
        )
    paths = write_mobility_outputs(out, outdir)
    for path in paths:
        print('  wrote %s' % path)
    return paths


def metal_resistivity(
    eliashberg_npz: str,
    outdir: str,
    *,
    temps_k: Sequence[float] = (20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0),
    rho0_uohm_cm: float = 0.0,
    b_ee_per_ev: float = 0.0,
    omega_ev: tuple[float, float, float] = (0.005, 2.0, 0.005),
    thickness_um: float = float('inf'),
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
    interband_dielectric_dir: str | None = None,
    extended_drude: bool = True,
) -> dict:
    """Resistivity rho(T), infrared optics and thermal emissivity of a metal from alpha^2F_tr.

    Parameters
    ----------
    eliashberg_npz : str
        ``eliashberg.npz`` of an Eliashberg run with ``transport=True`` (alpha^2F_tr,
        lambda_tr, the plasma frequency).
    outdir : str
        Output directory (``resistivity.dat``, ``eps_<T>K.dat``, ``emish_<T>K.dat``,
        ``emis_th<deg>_<T>K.dat``, ``emist.dat``, ``emissivity.npz``).
    temps_k : sequence of float
        Temperatures (K).
    rho0_uohm_cm : float
        Residual (impurity) resistivity, micro-ohm cm (0: none).
    b_ee_per_ev : float
        Fermi-liquid coefficient ``b`` of ``hbar/tau = b [(hbar w)^2 + (2 pi kT)^2]`` (0: none).
    omega_ev : (min, max, step)
        Photon energies of the optics.
    thickness_um : float
        Film thickness (``inf``: opaque).
    angles_deg : sequence of float
        Emission angles of the directional spectra.
    interband_dielectric_dir : str, optional
        Output of a ``dielectric_tensor(intraband=False)`` run on the extended basis
        (``epsr/epsi_*.dat``); without it ``eps_inter = 1``.
    extended_drude : bool
        Frequency-dependent rate and mass enhancement (``False``: the DC rate).

    Returns
    -------
    dict
        ``channels``, ``rows`` (T, rho total and per channel in micro-ohm cm),
        ``emissivity``, ``lambda_tr`` and ``plasma_ev``.

    Raises
    ------
    FileNotFoundError
        If ``eliashberg_npz`` has no alpha^2F_tr.
    """
    from ...boltzmann.scattering_channels import (
        FermiLiquidRate,
        ResidualResistivity,
        TransportSpectralChannel,
    )
    from ...response.drude import dc_resistivity
    from ...response.emissivity import metal_thermal_emissivity

    with np.load(eliashberg_npz) as data:
        if 'a2F_tr' not in data:
            raise FileNotFoundError(
                '%s has no alpha^2F_tr: run eliashberg.py --transport first.' % eliashberg_npz
            )
        plasma = float(data['omega_plasma_ev'])
        lambda_tr = float(data['lambda_tr'])
    channels = [TransportSpectralChannel.from_npz(eliashberg_npz)]
    if rho0_uohm_cm > 0.0:
        channels.append(ResidualResistivity(rho0_uohm_cm * 1.0e-8, plasma))
    if b_ee_per_ev > 0.0:
        channels.append(FermiLiquidRate(b_ee_per_ev))
    os.makedirs(outdir, exist_ok=True)
    print(
        'Metal transport from alpha^2F_tr: lambda_tr = %.3f, hbar omega_p = %.3f eV'
        % (lambda_tr, plasma)
    )
    print('    T (K)   rho (uOhm cm)   ' + '   '.join('%s' % c.name for c in channels))
    rows = []
    for temp in temps_k:
        rho = dc_resistivity(plasma, channels, temp)
        parts = [rho['channels'][c.name][1] * 1.0e8 for c in channels]
        rows.append([temp, rho['rho_ohm_m'] * 1.0e8] + parts)
        print(
            '  %7.1f  %14.4f   ' % (temp, rho['rho_ohm_m'] * 1.0e8)
            + '   '.join('%.4f' % p for p in parts)
        )
    np.savetxt(
        os.path.join(outdir, 'resistivity.dat'),
        np.array(rows),
        fmt='%14.6f',
        header='T (K), rho total and per channel (uOhm cm): ' + ', '.join(c.name for c in channels),
    )
    omin, omax, ostep = omega_ev
    omega = np.arange(omin, omax + 0.5 * ostep, ostep)
    interband = (
        read_direct_dielectric(interband_dielectric_dir) if interband_dielectric_dir else None
    )
    emissivity = metal_thermal_emissivity(
        omega,
        list(temps_k),
        plasma,
        channels,
        interband=interband,
        thickness_m=thickness_um * 1.0e-6,
        angles_deg=tuple(angles_deg),
        extended=extended_drude,
    )
    print(
        'Thermal emissivity (%s Drude, interband: %s):'
        % (emissivity['drude'], interband_dielectric_dir or 'none')
    )
    print('    T (K)   total hemispherical   Planck coverage')
    for temp, total, cover in zip(
        emissivity['temps_k'], emissivity['total_opaque'], emissivity['planck_coverage']
    ):
        print('  %7.1f  %20.5f  %16.4f' % (temp, total, cover))
    for path in write_emissivity_outputs(emissivity, outdir):
        print('  wrote %s' % path)
    return {
        'channels': channels,
        'rows': np.array(rows),
        'emissivity': emissivity,
        'lambda_tr': lambda_tr,
        'plasma_ev': plasma,
    }
