"""Optics workflow: phonon-assisted + direct absorption, slab emissivity, direct dielectric function.

Wraps :func:`~PAOFLOW.elphon.phonon_assisted_absorption.phonon_assisted_absorption_dense_q`,
:func:`~PAOFLOW.elphon.phonon_assisted_absorption.thermal_emissivity` and PAOFLOW's
``dielectric_tensor`` route with the inputs of an :class:`~.system.ElphonSystem`.
"""

from __future__ import annotations

import os
import shutil
from typing import Sequence

import numpy as np

from ..coupling_screening import screening_report
from ..phonon_assisted_absorption import (
    external_refractive_index,
    pao_band_gaps,
    phonon_assisted_absorption_dense_q,
    read_direct_dielectric,
    thermal_emissivity,
    varshni_gap,
    varshni_shift,
    write_absorption_outputs,
    write_emissivity_outputs,
)
from .system import Electrons, ElphonSystem, is_root


def reference_gap(system: ElphonSystem, savedir: str, nk: int, electrons: Electrons) -> float:
    """Indirect gap (eV) of the PAO Hamiltonian of another QE run (e.g. eACBN0) on the dense grid.

    Parameters
    ----------
    system : ElphonSystem
    savedir : str
        The reference save (same crystal, same PAO basis).
    nk : int
        Dense grid per axis of the absorption run.
    electrons : Electrons
        For the spin degeneracy.

    Returns
    -------
    float
        The indirect gap.
    """
    run = system.hamiltonian_of(savedir)
    gaps = pao_band_gaps(
        run.HRs,
        run.nscf['at'],
        nk,
        system.nelec,
        run.nscf['orbital_positions'],
        spin_degeneracy=system.degeneracy(electrons),
    )
    return gaps['indirect_gap_ev']


def scissor_settings(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    nk: int,
    target_gap_ev: float | None = None,
    gap_reference_save: str | None = None,
    varshni: Sequence[float] | None = None,
    electrons_save: str | None = None,
    comm=None,
) -> tuple:
    """Scissor of the band energies: ``(scissor_ev, target_gap_ev, mode)``.

    Parameters
    ----------
    system, electrons
        The system and its electrons.
    nk : int
        Dense grid of the run (the reference gap is evaluated on it).
    target_gap_ev : float, optional
        Target indirect gap (0 K value with ``varshni``).
    gap_reference_save : str, optional
        QE save of an eACBN0 run whose PAO gap is the target.
    varshni : (alpha, beta), optional
        ``E_g(T) = E0 - alpha T^2 / (T + beta)``.
    electrons_save : str, optional
        The eACBN0 electrons save of the run (incompatible with ``gap_reference_save``).
    comm : MPI communicator, optional

    Returns
    -------
    tuple
        ``'target'``: the conduction bands are shifted to the target, optionally
        temperature dependent; ``'relative'``: only the Varshni shift on top of the
        Hamiltonian's own gap; ``'none'``.

    Raises
    ------
    ValueError
        If both ``electrons_save`` and ``gap_reference_save`` are given.
    """
    if electrons_save and gap_reference_save:
        raise ValueError(
            'gap_reference_save is a scissor target for PBE electrons; with electrons_save the '
            'Hamiltonian already is the eACBN0 one (target_gap_ev fine-tunes its gap).'
        )
    e0 = target_gap_ev
    if gap_reference_save:
        e0 = reference_gap(system, gap_reference_save, nk, electrons)
        if is_root(comm):
            print('  reference gap of %s: %.3f eV' % (gap_reference_save, e0))
    if e0 is not None:
        return None, (varshni_gap(e0, *varshni) if varshni else e0), 'target'
    if varshni:
        return varshni_shift(*varshni), None, 'relative'
    return None, None, 'none'


def phonon_assisted_absorption(
    system: ElphonSystem,
    electrons: Electrons,
    *,
    nk: int,
    nq: int,
    omega_ev: tuple[float, float, float] = (0.05, 3.0, 0.05),
    temps_k: Sequence[float] = (300.0,),
    degauss_ev: float = 0.05,
    adaptive_smearing: float | None = None,
    fsthick_ev: float = 4.0,
    fermi_energy_ev: float | None = None,
    refractive_index: float | None = None,
    direct_dielectric: tuple | None = None,
    scissor_ev=None,
    target_gap_ev=None,
    scissor_mode: str = 'none',
    scissor_velocity: str = 'renormalized',
    screening_eps: float | None = None,
    screening_eps_dft: float | None = None,
    position_dipoles: bool = True,
    position_dipole_level: str = 'full',
    emissivity: bool = False,
    kk_omega_max_ev: float | None = 10.0,
    comm=None,
) -> dict:
    """Phonon-assisted + direct Im eps(omega) and alpha(omega) on the dense grids.

    Parameters
    ----------
    system, electrons
        The system and its electrons (``electrons_save`` for eACBN0 electrons).
    nk, nq : int
        Dense electron and phonon grids per axis (``nk % nq == 0``).
    omega_ev : (min, max, step)
        Photon energies (EPW ``omegamin``, ``omegamax``, ``omegastep``).
    temps_k : sequence of float
        Temperatures (K).
    degauss_ev : float
        Energy-conserving delta smearing (EPW ``degaussw``); the floor with
        ``adaptive_smearing``.
    adaptive_smearing : float, optional
        Yates prefactor of the per-transition widths.
    fsthick_ev : float
        States within this window of the Fermi level (EPW ``fsthick``).
    fermi_energy_ev : float, optional
        Fermi level on the PAO energy scale (``None``: mid-gap); with ``emissivity`` the
        charge-neutral ``'intrinsic'`` level of every temperature is used.
    refractive_index : float, optional
        Constant ``n_r`` of ``alpha = omega Im eps / (n_r c)``; ``None``: ``n(omega)`` of
        ``direct_dielectric`` when given, else 3.4 (EPW).
    direct_dielectric : tuple, optional
        :func:`~PAOFLOW.elphon.phonon_assisted_absorption.read_direct_dielectric` of an
        extended-basis ``dielectric_tensor`` run.
    scissor_ev, target_gap_ev, scissor_mode
        Of :func:`scissor_settings`.
    scissor_velocity : {'renormalized', 'ks'}
        Velocities of the scissor-corrected Hamiltonian.
    screening_eps, screening_eps_dft : float, optional
        Poncé screening correction of the coupling (target and DFT dielectric constants).
    position_dipoles : bool
        Position dipoles of the PAO orbitals in the velocity (``i[H, d]``).
    position_dipole_level : {'full', 'intra-atomic'}
    emissivity : bool
        Prepare the run for the slab emissivity: intrinsic Fermi level and, without an
        external direct dielectric function, the Kramers-Kronig ``eps_1`` of the direct term.
    kk_omega_max_ev : float, optional
        Kramers-Kronig range of the direct term.
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
    if refractive_index is not None:
        n_r = refractive_index
    elif direct_dielectric is not None:
        n_r = external_refractive_index(direct_dielectric)
    else:
        n_r = 3.4
    out = phonon_assisted_absorption_dense_q(
        *system.positional_arguments(electrons),
        nelec=system.nelec,
        nk_dense=nk,
        nq_dense=nq,
        omega_ev=tuple(omega_ev),
        temps_k=list(temps_k),
        degauss_ev=degauss_ev,
        adaptive_smearing=adaptive_smearing,
        min_degauss_ev=degauss_ev,
        fsthick_ev=fsthick_ev,
        fermi_energy_ev='intrinsic' if emissivity else fermi_energy_ev,
        kk_omega_max_ev=kk_omega_max_ev if emissivity and direct_dielectric is None else None,
        refractive_index=n_r,
        scissor_ev=scissor_ev,
        target_gap_ev=target_gap_ev,
        scissor_velocity=scissor_velocity,
        screening_eps=screening_eps,
        screening_eps_dft=screening_eps_dft,
        position_dipoles=electrons.data_controller if position_dipoles else None,
        position_dipole_level=position_dipole_level,
        spin_degeneracy=system.degeneracy(electrons),
        comm=comm,
        **system.coupling_arguments(electrons),
    )
    out.update(electrons.provenance)
    out['scissor_mode'] = scissor_mode
    return out


def absorption_label(
    nk: int,
    nq: int,
    degauss_ev: float,
    adaptive_smearing: float | None,
    position_dipoles: bool,
    position_dipole_level: str,
) -> str:
    """``'k 12^3, q 6^3, degauss 0.050 eV, position dipoles (full)'``."""
    smearing = (
        '%.3f eV' % degauss_ev
        if adaptive_smearing is None
        else 'adaptive a = %.2f (floor %.1f meV)' % (adaptive_smearing, 1e3 * degauss_ev)
    )
    dipoles = ', position dipoles (%s)' % position_dipole_level if position_dipoles else ''
    return 'k %d^3, q %d^3, degauss %s%s' % (nk, nq, smearing, dipoles)


def write_absorption(out: dict, outdir: str, *, label: str = '', comm=None) -> list[str]:
    """Print the gaps and the scissor and write the spectra (rank 0).

    Parameters
    ----------
    out : dict
        Of :func:`phonon_assisted_absorption`.
    outdir : str
        Output directory: EPW-format ``epsilon2_indabs_<T>K.dat``, ``epsilon2_dirabs_<T>K.dat``,
        ``alpha_<T>K.dat`` and ``absorption.npz``.
    label : str, optional
        :func:`absorption_label`.
    comm : MPI communicator, optional

    Returns
    -------
    list of str
        Written files (empty on the other ranks).
    """
    if not is_root(comm):
        return []
    print('PAOFLOW phonon-assisted absorption%s:' % (' (%s)' % label if label else ''))
    n = out['refractive_index']
    print(
        '  alpha = omega Im eps / (n c) with n %s'
        % (
            '= %.3g' % n
            if np.ndim(n) == 0
            else 'of the direct dielectric run, %.3g-%.3g' % (n.min(), n.max())
        )
    )
    print(
        '  %s gaps: indirect %.3f eV, direct %.3f eV'
        % (out['electrons_dft'], out['indirect_gap_dft_ev'], out['direct_gap_dft_ev'])
    )
    if out['electrons_save']:
        print(
            '  band energies and eigenvectors from %s; coupling from %s'
            % (out['electrons_save'], out['coupling_save'])
        )
    print(screening_report(out))
    if abs(out['scissor_ev']).max() > 0.0:
        relative = out['scissor_mode'] == 'relative'
        print(
            '  scissor (%s velocities%s):'
            % (
                out['scissor_velocity'],
                '; Varshni shift relative to the Hamiltonian gap' if relative else '',
            )
        )
        for temp, delta, gap in zip(out['temps_k'], out['scissor_ev'], out['indirect_gap_ev_t']):
            print('    T = %7.1f K   scissor %+.3f eV   indirect gap %.3f eV' % (temp, delta, gap))
    paths = write_absorption_outputs(out, outdir)
    for path in paths:
        print('  wrote %s' % path)
    return paths


def slab_emissivity(
    out: dict,
    outdir: str,
    *,
    thickness_um: float = 500.0,
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
    direct_dielectric: tuple | None = None,
    direct_dielectric_label: str | None = None,
    direct_scissor: str = 'same',
    comm=None,
) -> dict | None:
    """Thermal emissivity of a free-standing slab from an absorption run; writes ``outdir`` (rank 0).

    Parameters
    ----------
    out : dict
        Of :func:`phonon_assisted_absorption` (with ``emissivity=True``), or
        ``absorption.npz`` loaded as a dict.
    outdir : str
        Output directory (``emish_<T>K.dat``, ``emis_th<deg>_<T>K.dat``, ``eps_<T>K.dat``,
        ``emist.dat``, ``emissivity.npz``).
    thickness_um : float
        Slab thickness (``inf``: opaque half-space).
    angles_deg : sequence of float
        Emission angles of the directional spectra.
    direct_dielectric : tuple, optional
        External direct dielectric function (extended basis); ``None``: Kramers-Kronig of
        the run.
    direct_dielectric_label : str, optional
        Its directory, for the summary.
    direct_scissor : {'same', 'thermal', 'none'}
        Shift of the external direct dielectric function.
    comm : MPI communicator, optional

    Returns
    -------
    dict or None
        The emissivity (``None`` on the other ranks).
    """
    if not is_root(comm):
        return None
    emissivity = thermal_emissivity(
        out,
        thickness_um * 1.0e-6,
        angles_deg=tuple(angles_deg),
        direct_dielectric=direct_dielectric,
        direct_scissor=direct_scissor,
    )
    print(
        'Thermal emissivity (slab, d = %g um; direct eps from %s, f-sum ratio %.2f):'
        % (
            thickness_um,
            direct_dielectric_label or 'Kramers-Kronig of this run',
            emissivity['f_sum_ratio'].mean(),
        )
    )
    if 'direct_onset_ev' in emissivity:
        print(
            '  external direct eps: onset %.2f eV, direct gap of the Hamiltonian %.2f eV; shifted by '
            '%+.3f ... %+.3f eV (%s)'
            % (
                emissivity['direct_onset_ev'],
                emissivity['direct_gap_h_ev'],
                emissivity['direct_scissor_ev'].min(),
                emissivity['direct_scissor_ev'].max(),
                emissivity['direct_scissor'],
            )
        )
    fermi = np.atleast_1d(out['fermi_levels_ev']) if 'fermi_levels_ev' in out else None
    print('    T (K)   E_F (eV)   total slab   total opaque   Planck coverage')
    for it, temp in enumerate(emissivity['temps_k']):
        print(
            '  %7.1f  %9s  %11.4f  %13.4f  %16.4f'
            % (
                temp,
                '%.4f' % fermi[it] if fermi is not None else '',
                emissivity['total_slab'][it],
                emissivity['total_opaque'][it],
                emissivity['planck_coverage'][it],
            )
        )
    for path in write_emissivity_outputs(emissivity, outdir):
        print('  wrote %s' % path)
    return emissivity


def load_absorption(path: str) -> dict:
    """``absorption.npz`` as a dict (0-d arrays kept; strings as ``str``)."""
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    for key, value in out.items():
        if value.dtype.kind in 'US':
            out[key] = str(value)
    return out


def emissivity_from_absorption(
    source_dir: str,
    outdir: str,
    *,
    direct_dielectric_dir: str,
    direct_scissor: str = 'thermal',
    target_gap_ev: float | None = None,
    thickness_um: float = 500.0,
    angles_deg: Sequence[float] = (0.0, 30.0, 60.0),
) -> dict:
    """Recompute the slab emissivity of a finished absorption run with another direct eps.

    Parameters
    ----------
    source_dir : str
        Output directory of :func:`write_absorption` (``absorption.npz``); never modified.
    outdir : str
        New directory for the emissivity files and a copy of ``absorption.npz`` (a
        complete model folder, e.g. for the thermo-optics analysis).
    direct_dielectric_dir : str
        Output directory of the ``dielectric_tensor`` run (``epsr/epsi_*.dat``).
    direct_scissor : {'same', 'thermal', 'none'}
        ``'thermal'`` needs the 0 K scissor: runs since October 2026 store it
        (``scissor_0k_ev``); older ones need ``target_gap_ev``.
    target_gap_ev : float, optional
        0 K target gap of the run scissor (``Delta(0) = target - Hamiltonian gap``).
    thickness_um, angles_deg
        As :func:`slab_emissivity`.

    Returns
    -------
    dict
        The emissivity.

    Raises
    ------
    ValueError
        If ``outdir`` is the source directory, or the 0 K scissor cannot be determined.
    """
    source_dir, outdir = os.path.abspath(source_dir), os.path.abspath(outdir)
    if outdir == source_dir:
        raise ValueError('outdir must differ from the run output (it is never overwritten).')
    results = load_absorption(os.path.join(source_dir, 'absorption.npz'))
    if direct_scissor == 'thermal' and 'scissor_0k_ev' not in results:
        if target_gap_ev is None:
            scissor = np.asarray(results['scissor_ev'], dtype=float)
            if np.any(scissor != 0.0) and str(results.get('scissor_mode', '')) != 'relative':
                raise ValueError(
                    'this run has no scissor_0k_ev: give its 0 K target gap (target_gap_ev).'
                )
            results['scissor_0k_ev'] = 0.0  # no scissor, or the Varshni shift only
        else:
            results['scissor_0k_ev'] = target_gap_ev - float(results['indirect_gap_dft_ev'])
    os.makedirs(outdir, exist_ok=True)
    shutil.copy2(os.path.join(source_dir, 'absorption.npz'), os.path.join(outdir, 'absorption.npz'))
    return slab_emissivity(
        results,
        outdir,
        thickness_um=thickness_um,
        angles_deg=angles_deg,
        direct_dielectric=read_direct_dielectric(direct_dielectric_dir),
        direct_dielectric_label=direct_dielectric_dir,
        direct_scissor=direct_scissor,
    )


def direct_dielectric(
    system: ElphonSystem,
    savedir: str,
    outdir: str,
    *,
    nfft: int = 24,
    emin: float = 0.01,
    emax: float = 10.0,
    ne: int = 1000,
    delta: float = 0.1,
    intraband: bool = True,
    configuration: str = 'extended',
    pthr: float = 0.95,
    smearing: str | None = None,
) -> None:
    """Direct dielectric function on the extended PAO basis (PAOFLOW's optical route).

    Parameters
    ----------
    system : ElphonSystem
        For the basis tables.
    savedir : str
        QE save of an nscf with more bands than extended-basis orbitals
        (``<optics_dir>/<prefix>.save``).
    outdir : str
        Output directory of ``epsr_<c>.dat`` and ``epsi_<c>.dat`` (read by the absorption
        and metal-transport drivers).
    nfft : int
        Interpolated k-grid per axis.
    emin, emax, ne, delta : float
        Energy window, points and broadening of ``dielectric_tensor``.
    intraband : bool
        Include the Drude term (``False`` for the interband part of a metal, whose
        intraband part comes from alpha^2F_tr).
    configuration : str
        PAO basis preset.
    pthr : float
        Projectability threshold.
    smearing : str, optional
        PAOFLOW smearing of the Fermi level (metals: ``'gauss'``).

    Notes
    -----
    Runs ``projections`` -> ``projectability`` -> ``pao_hamiltonian`` ->
    ``interpolated_hamiltonian`` -> ``pao_eigh`` -> ``gradient_and_momenta(position_dipoles=True)``
    -> ``dielectric_tensor(d_tensor='diag')``.
    """
    from ...PAOFLOW import PAOFLOW

    savedir, outdir = os.path.abspath(savedir), os.path.abspath(outdir)
    options = dict(
        workpath=os.path.dirname(outdir),
        outputdir=outdir,
        savedir=savedir,
        save_overlaps=False,
        verbose=False,
    )
    if smearing is not None:
        options['smearing'] = smearing
    pf = PAOFLOW(**options)
    pf.projections(basispath=system.basis_path, configuration=configuration)
    pf.projectability(pthr=pthr)
    pf.pao_hamiltonian()
    pf.interpolated_hamiltonian(nfft1=nfft, nfft2=nfft, nfft3=nfft)
    pf.pao_eigh()
    pf.gradient_and_momenta(position_dipoles=True)
    pf.dielectric_tensor(
        emin=emin, emax=emax, ne=ne, d_tensor='diag', delta=delta, intraband=intraband
    )
    pf.finish_execution()
