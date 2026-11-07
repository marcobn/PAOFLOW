"""Frozen copy of the ``paoflow-gen`` prompt functions before the declarative-questions refactor.

The parity tests (``test_questions_parity.py``) run these and the new
:mod:`PAOFLOW.gen.questions` path on the same answers and require the same configuration, the same
prompts and byte-identical generated files.  Do not edit: this is the reference of 7 November 2026
(``develop-marco`` at e5071171), copied verbatim; only ``_input`` calls go through the module so
``monkeypatch.setattr(pd, '_input', ...)`` reaches them, and relative imports are absolute.
"""

# ruff: noqa
# fmt: off

import glob
import os
from typing import Any, Sequence

import numpy as np

from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen.paoflow_driver import (
    ENERGY_WINDOW_PROPERTIES,
    PROPERTY_MENU,
    _uses_projwfc,
    ask,
    ask_choice,
    ask_float,
    ask_int,
    ask_yes_no,
    detect_prefix,
    detect_savedir,
    detect_upfs,
    detect_v_cutoff,
    parse_laser_list,
)
from PAOFLOW.gen.sscha_driver import DEFAULT_POPULATIONS

def select_properties():
    """Show the property menu and return the ordered set of chosen keys."""
    print('\nAvailable properties:')
    for i, (_key, label) in enumerate(PROPERTY_MENU, 1):
        print(f'  {i}: {label}')
    raw = pd._input("Enter a comma/space separated list (e.g. '1 2 8'): ").strip()
    tokens = raw.replace(',', ' ').split()
    chosen = []
    for tok in tokens:
        try:
            idx = int(tok)
        except ValueError:
            continue
        if 1 <= idx <= len(PROPERTY_MENU):
            key = PROPERTY_MENU[idx - 1][0]
            if key not in chosen:
                chosen.append(key)
    return chosen


def collect_common(args, workdir):
    """Prompt for configuration shared by both workflows."""
    det_save = detect_savedir(workdir)
    det_upfs = detect_upfs(workdir, det_save)
    det_prefix = detect_prefix(workdir, det_save)

    savedir = args.savedir or ask('Save directory (<prefix>.save)', det_save or 'pwscf.save')
    prefix = args.prefix or ask('Prefix', det_prefix)
    if args.upf:
        upfs = [u.strip() for u in args.upf.split(',') if u.strip()]
    else:
        default_upfs = ', '.join(det_upfs) if det_upfs else 'pseudo.UPF'
        answer = ask('Pseudopotential (UPF) file(s), comma-separated', default_upfs)
        upfs = [u.strip() for u in answer.split(',') if u.strip()]
    basisdir = ask('Basis directory name', 'BASIS_PS')
    ibrav = ask_int('ibrav (0 = read cell; needs band path for bands)', 0)
    is_2d = ask_yes_no('Is this a 2D system (slab with vacuum along c)?', False)
    projector = ask_choice(
        'Projector source (internal = PAOFLOW atomic projection; projwfc = QE projwfc.x)',
        ['internal', 'projwfc'],
        'internal',
    )
    return {
        'savedir': savedir,
        'prefix': prefix,
        'upfs': upfs,
        'basisdir': basisdir,
        'ibrav': ibrav,
        'is_2d': is_2d,
        'projector': projector,
        'outputdir': 'output',
        'workdir': workdir,
    }


def collect_run(common):
    """Prompt for the full property-run configuration."""
    props = select_properties()
    if not props:
        print('No valid properties selected; nothing to generate.')
        return None

    std_basis = 'standard'
    if any(p != 'optical' for p in props):
        std_basis = ask_choice(
            '\nBasis configuration for the standard properties:',
            ['minimal', 'standard', 'extended'],
            'standard',
        )

    cfg = dict(common)
    cfg['properties'] = props
    cfg['std_basis'] = std_basis
    cfg['npool'] = ask_int('npool', 1)
    cfg['smearing'] = ask('Smearing (gauss / m-p / None)', 'gauss')
    if cfg['smearing'].lower() == 'none':
        cfg['smearing'] = None
    cfg['spin_orbit'] = ask_yes_no('Spin-orbit (noncollinear) calculation?', False)
    cfg['nk'] = ask_int('Band path k-points (NK)', 400) if 'bands' in props else 400
    if any(p in ENERGY_WINDOW_PROPERTIES for p in props):
        print('\nEnergy grid for DOS / transport / Hall properties (eV, relative to E_F).')
        print(
            '  The full PAO band range is computed at run time and printed as a '
            'suggestion;\n  the values below are what the calculation actually uses.'
        )
        cfg['emin'] = ask_float('  EMIN', -8.0)
        cfg['emax'] = ask_float('  EMAX', 4.0)
        cfg['ne'] = ask_int('  NE (number of energy points)', 1000)
    else:
        cfg['emin'] = -8.0
        cfg['emax'] = 4.0
        cfg['ne'] = 1000
    cfg['do_pdos'] = ask_yes_no('Projected DOS as well?', True) if 'dos' in props else False
    cfg['interpolate'] = ask_yes_no(
        'Use double-grid interpolation (denser FFT)?',
        'optical' in props,
    )
    cfg['nfft'] = ask_int('FFT grid size per direction', 24) if cfg['interpolate'] else 0
    if 'nonlinear' in props:
        collect_nonlinear(cfg)
    cfg['plot'] = ask_yes_no('Also generate a plotting script (plot.py)?', True)
    return cfg


def collect_nonlinear(cfg):
    """Prompt for the non-linear (and coupled) optics run; fills the ``nlo_*`` / ``coupled_*`` keys.

    Parameters
    ----------
    cfg : dict
        Run configuration (``is_2d`` and the projection mode already set); modified in place.

    Returns
    -------
    dict
        ``cfg``.
    """
    internal = not _uses_projwfc(cfg)
    print('\nNon-linear optics (shift / injection currents, SHG):')
    molecule = False
    if not cfg.get('is_2d'):
        molecule = ask_yes_no('  isolated molecule in a box (Gamma point, beta tensor)?', False)
    cfg['nlo_molecule'] = molecule
    if not molecule:
        cfg['nlo_nk'] = ask_int('  dense k-grid per direction', 40)
    cfg['nlo_emin'] = ask_float('  smallest photon energy (eV)', 0.0)
    cfg['nlo_emax'] = ask_float('  largest photon energy (eV)', 4.0)
    cfg['nlo_ne'] = ask_int('  number of photon energies', 401)
    cfg['nlo_eta'] = ask_float('  broadening eta (eV)', 0.05)
    cfg['nlo_scissor'] = ask_float('  scissor shift of the conduction bands (eV)', 0.0)
    if not molecule:
        comps = ask("  tensor components ('all', 'diag' or e.g. xyz,yyy)", 'all')
        cfg['nlo_components'] = [c.strip() for c in comps.split(',')] if ',' in comps else comps
    responses = ask('  responses (any of shift,injection,shg)', 'shift,injection,shg')
    cfg['nlo_responses'] = tuple(r.strip() for r in responses.split(',') if r.strip())
    cfg['nlo_dipoles'] = (
        ask_yes_no('  Löwdin position dipoles (recommended: basis-independent results)?', True)
        if internal
        else False
    )
    cfg['nlo_shg_terms'] = ask_yes_no(
        '  split the SHG into interband / intraband terms at w and 2w?', False
    )
    if molecule:
        cfg['nlo_coupling'] = ask(
            '  coupled response of the molecule (none, monopole, hartree, hartree+alda)',
            'hartree+alda',
        )
    cfg['nlo_coupled'] = False
    if internal and not molecule and not cfg.get('is_2d'):
        cfg['nlo_coupled'] = ask_yes_no(
            '  coupled optical response (local fields, ALDA, long-range kernels)?', False
        )
    if cfg['nlo_coupled']:
        cfg['coupled_nk'] = ask_int('    coupled-response k-grid per direction', 16)
        kernels = ask(
            '    kernel levels (none,hartree,hartree+alda,hartree+alda+lrc,hartree+alda+bootstrap)',
            'none,hartree,hartree+alda,hartree+alda+bootstrap',
        )
        cfg['coupled_kernels'] = tuple(k.strip() for k in kernels.split(',') if k.strip())
        cfg['coupled_emax'] = ask_float(
            '    largest photon energy of the dielectric function (eV)', 5.0
        )
        cfg['coupled_ne'] = ask_int('    number of photon energies', 51)
        shg = ask(
            '    SHG photon energies (eV, comma separated; blank skips the coupled SHG)', '0.1'
        )
        cfg['coupled_shg_energies'] = [float(x) for x in shg.split(',') if x.strip()]
        cfg['coupled_eta'] = ask_float('    broadening eta (eV)', 0.1)
    return cfg


def collect_phonon(common):
    """Prompt for the harmonic-phonon (finite-displacement) configuration."""
    cfg = dict(common)
    cfg['supercell'] = ask_int('Supercell multiplicity (isotropic diagonal)', 2)
    cfg['displacement'] = ask_float('Displacement amplitude (Bohr)', 0.06)
    mesh = ask_int('q-mesh density per axis (DOS / thermal)', 12)
    cfg['mesh'] = [mesh, mesh, mesh]
    cfg['units'] = ask_choice('Frequency units', ['cm-1', 'THz'], 'cm-1')
    cfg['do_thermal'] = ask_yes_no('Compute thermal properties (F, S, Cv)?', True)
    cfg['pp_dir'] = ask('Pseudopotential directory (blank or "HERE" = script directory)', 'HERE')
    cfg['hubbard_file'] = ask(
        'pw.x input with a HUBBARD card to inject (on-site U only; blank = none)', ''
    )
    cfg['born'] = ask_yes_no(
        'Compute Born charges + dielectric tensor (NAC / LO-TO splitting)?', True
    )
    if cfg['born']:
        cfg['born_method'] = ask_choice(
            'Born/epsilon method (dfpt = ph.x, field = lelfield pw.x)',
            ['dfpt', 'field'],
            'dfpt',
        )
    else:
        cfg['born_method'] = 'dfpt'
    cfg['vibdielectric'] = cfg['born'] and ask_yes_no(
        'Also compute the vibrational (ionic) dielectric function eps(w) (reststrahlen band)?',
        True,
    )
    if cfg['vibdielectric']:
        cfg['vibdielectric_gamma'] = ask_float(
            '  Phonon linewidth gamma for eps(w) (in the chosen units)', 4.0
        )
        cfg['vibdielectric_emissivity'] = ask_yes_no(
            '  Also compute the reststrahlen (phonon) emissivity (1 - R)?', True
        )
        if cfg['vibdielectric_emissivity']:
            temp = ask_float('    Temperature for the total hemispherical emissivity (K)', 300.0)
            cfg['vibdielectric_emis_temp'] = [temp]
        else:
            cfg['vibdielectric_emis_temp'] = [300.0]
    else:
        cfg['vibdielectric_gamma'] = 4.0
        cfg['vibdielectric_emissivity'] = False
        cfg['vibdielectric_emis_temp'] = [300.0]
    cfg['qha'] = ask_yes_no(
        'Compute quasi-harmonic properties (thermal expansion, V(T), B(T), Gruneisen)?',
        False,
    )
    if cfg['qha']:
        cfg['qha_nvolumes'] = ask_choice(
            '  Number of sampled volumes (5 = full EOS fit; 3 = parabolic fit)',
            ['5', '3'],
            '5',
        )
        cfg['qha_nvolumes'] = int(cfg['qha_nvolumes'])
        cfg['qha_strain'] = ask_float('  Maximum linear strain of the volume scan', 0.02)
        if cfg['qha_nvolumes'] >= 4:
            cfg['qha_eos'] = ask_choice(
                '  Equation of state',
                ['vinet', 'birch_murnaghan', 'murnaghan'],
                'vinet',
            )
        else:
            cfg['qha_eos'] = 'vinet'
        cfg['qha_tmin'] = ask_float('  Minimum temperature (K)', 0.0)
        cfg['qha_tmax'] = ask_float('  Maximum temperature (K)', 1000.0)
        cfg['qha_tstep'] = ask_float('  Temperature step (K)', 10.0)
        cfg['qha_pressure'] = ask_float('  External pressure (GPa)', 0.0)
    else:
        cfg['qha_nvolumes'] = 5
        cfg['qha_strain'] = 0.02
        cfg['qha_eos'] = 'vinet'
        cfg['qha_tmin'] = 0.0
        cfg['qha_tmax'] = 1000.0
        cfg['qha_tstep'] = 10.0
        cfg['qha_pressure'] = 0.0
    print('\npw.x launch settings for the displaced-supercell SCF runs:')
    cfg['mpi_qe'] = ask('MPI command for pw.x', 'mpirun -np 4')
    cfg['qe_pw'] = ask(
        "pw.x executable/command (may include flags, e.g. 'pw.x', 'pw.x -npool 4', "
        "'/opt/qe/bin/pw.x')",
        'pw.x',
    )
    cfg['plot'] = ask_yes_no('Also generate a plotting script (plot.phonon.py)?', True)
    cfg['raman_method'] = ask_choice(
        'Also generate a Raman workflow (main.raman.py)?  Pick the flavour '
        '(none = skip; all = static + resonance from one run)',
        ['none', 'static', 'resonance', 'all'],
        'none',
    )
    cfg['raman'] = cfg['raman_method'] != 'none'
    if cfg['raman']:
        print('\nRaman finite-difference settings:')
        cfg['raman_delta'] = ask_float('  Mass-weighted displacement step (Bohr*sqrt(amu))', 0.05)
        cfg['raman_nbnd'] = ask_int('  Bands for the displaced SCF (0 = QE default)', 0)
        cfg['raman_npool'] = ask_int('  npool for the per-cell dielectric run', 4)
        cfg['raman_smearing'] = ask('  Smearing for the dielectric run', 'gauss')
        raman_nfft = ask_int('  Double-grid FFT size per direction (0 = skip)', 24)
        cfg['raman_nfft'] = raman_nfft
        cfg['raman_e_static'] = ask_float('  Static-epsilon upper energy (eV)', 0.05)
        cfg['raman_temperature'] = ask_float('  Temperature for the Bose factor (K)', 300.0)
        cfg['raman_gamma'] = ask_float('  Lorentzian FWHM (in the chosen units)', 4.0)
        if cfg['raman_method'] in ('resonance', 'all'):
            laser_str = ask(
                '  Laser wavelength(s) in nm (comma-separated, or a Python list '
                'expression like "[n for n in range(450, 650, 5)]")',
                '532',
            )
            lasers = parse_laser_list(laser_str)
            cfg['raman_laser_nm'] = lasers if len(lasers) > 1 else lasers[0]
            cfg['raman_lifetime'] = ask_float(
                '  Lifetime broadening of the dielectric tensor (eV)', 0.1
            )
        else:
            laser = ask_float('  Laser wavelength (nm; 0 = omit the (wL-wv)^4 factor)', 0.0)
            cfg['raman_laser_nm'] = laser if laser > 0 else None
            cfg['raman_lifetime'] = 0.1
        cfg['raman_pthr'] = 0.95
        cfg['raman_configuration'] = 'extended'
    return cfg


def collect_acbn0(common):
    """Prompt for the ACBN0 / eACBN0 configuration."""
    cfg = dict(common)
    cfg['use_intersite_v'] = ask_yes_no(
        'Include intersite V (eACBN0)? (No = on-site U only, ACBN0)', False
    )
    if cfg['use_intersite_v'] and not _uses_projwfc(cfg):
        print(
            '\nNote: intersite V (eACBN0) requires the true non-orthogonal atomic\n'
            '  overlap S(k), which only the projwfc.x projector provides. The\n'
            '  internal ortho-atomic projection dumps S(k)=I and gives wrong V.\n'
            '  Switching the projector to projwfc for this workflow.'
        )
        cfg['projector'] = 'projwfc'
    cfg['projection'] = ask('Projection scheme', 'ortho-atomic')
    cfg['conv_thr'] = float(ask('U/V convergence threshold (eV)', '0.05'))
    cfg['nk'] = ask_int('Band path k-points (NK)', 400)
    cfg['gaussian_threshold'] = float(
        ask('Gaussian-fit tolerance (loosen for ONCV/SOC pseudos)', '0.05')
    )

    print("\nEnter the Hubbard manifolds, one per line, as '<species>-<nl> <initU>'")
    print("  e.g. 'Si-3p 0.5'.  Leave blank to finish.")
    hubbard = []
    while True:
        line = pd._input('  manifold: ').strip()
        if not line:
            break
        parts = line.split()
        orb = parts[0].strip().strip('\'"')
        try:
            val = float(parts[1]) if len(parts) > 1 else 0.5
        except ValueError:
            val = 0.5
        hubbard.append((orb, val))
    if not hubbard:
        print('No Hubbard manifolds entered; nothing to generate.')
        return None
    cfg['hubbard'] = hubbard

    if cfg['use_intersite_v']:
        v_default = detect_v_cutoff(common.get('workdir', '.')) or 2.6
        cfg['v_cutoff'] = float(ask('Intersite V neighbour cutoff (Angstrom)', repr(v_default)))
        cfg['v_init'] = float(ask('Initial intersite V (eV)', '0.5'))

    print('\nParallel launch commands (edit later in the script if unsure):')
    cfg['mpi_qe'] = ask('MPI command for pw.x', 'mpirun -np 4')
    cfg['mpi_py'] = ask('MPI command for python', 'mpirun -np 1')
    cfg['mpi_hartree'] = ask('MPI command for the Hartree step', 'mpirun -np 4')
    cfg['qe_path'] = ask('Quantum ESPRESSO bin path', '')
    cfg['py_path'] = ask('Python bin path', '')
    return cfg


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.replace(',', ' ').split() if x.strip()]


def ask_divisible(ask_int, prompt: str, default: int, divisor: int, what: str) -> int:
    """Ask for an integer that is a positive multiple of ``divisor`` (re-prompting otherwise)."""
    while True:
        value = ask_int(prompt, default)
        if value > 0 and value % divisor == 0:
            return value
        print('  %s must be a positive multiple of %d.' % (what, divisor))


def ask_workflows(ask, choices: Sequence[str], default: Sequence[str]) -> list[str]:
    """Comma-separated multi-select among ``choices`` (re-prompts on an unknown name)."""
    while True:
        answer = ask('Workflows (comma-separated: %s)' % ', '.join(choices), ', '.join(default))
        names = [x.strip() for x in answer.replace(',', ' ').split() if x.strip()]
        unknown = [n for n in names if n not in choices]
        if not unknown:
            return list(dict.fromkeys(names))
        print('  unknown workflow(s): %s' % ', '.join(unknown))


def collect_elphon_system(args, workdir: str) -> dict[str, Any]:
    """Prompt for the electron-phonon system configuration.

    Parameters
    ----------
    args : argparse.Namespace
        ``prefix`` and ``out`` overrides of ``paoflow-gen``.
    workdir : str
        Directory scanned for a ``pw.x`` input.

    Returns
    -------
    dict
        The configuration of :func:`write_system_folder`.
    """
    from PAOFLOW.gen.epw_inputs import parse_exclude_bands, species_masses

    cfg: dict[str, Any] = {'workdir': workdir}
    template = pd.ask(
        'pw.x input of the system (gives qe/phonon/scf.in and the nscf inputs; blank = none)',
        pd.detect_pw_input(workdir) or '',
    )
    cfg['pw_template'] = template
    cfg['pw_template_text'] = None
    template_pseudo_dir = None
    if template:
        path = os.path.join(workdir, os.path.expanduser(template))
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                cfg['pw_template_text'] = handle.read()
        except OSError:
            print('  (%s not readable; the scf and nscf inputs are not written)' % path)
        else:
            template_pseudo_dir = pd._read_namelist_value(cfg['pw_template_text'], 'pseudo_dir')
            if template_pseudo_dir is not None:
                template_pseudo_dir = os.path.normpath(
                    os.path.join(os.path.dirname(os.path.abspath(path)), template_pseudo_dir)
                )
    default_prefix = (
        pd._read_namelist_value(cfg['pw_template_text'], 'prefix')
        if cfg['pw_template_text']
        else None
    )
    cfg['prefix'] = args.prefix or pd.ask('Prefix', (default_prefix or 'pwscf').strip('\'"'))
    cfg['name'] = pd.ask('System name (titles of the figures)', cfg['prefix'])
    out = os.path.abspath(os.path.expanduser(args.out or workdir))
    cfg['out'] = out
    if template_pseudo_dir is None:
        template_pseudo_dir = workdir
    # qe/phonon, qe/nscf and qe/optics are siblings, so one relative path serves all three.
    default_pseudo = os.path.relpath(template_pseudo_dir, os.path.join(out, 'qe', 'phonon'))
    cfg['pseudo_dir'] = pd.ask(
        'pseudo_dir of the QE run directories', default_pseudo.rstrip('/') + '/'
    )
    source = pd.ask_choice(
        "Coupling source (dfpt = PAOFLOW's vertex from ph.x dvscf, norm-conserving; "
        'epw = EPW epbwrite, any pseudopotential)',
        ['dfpt', 'epw'],
        'dfpt',
    )
    cfg['coupling_source'] = source
    kg = pd.ask_int('Coarse k-grid per axis (nscf)', 6)
    while True:
        qg = pd.ask_int('Coarse q-grid per axis (ph.x; must divide the k-grid)', kg)
        if qg > 0 and kg % qg == 0:
            break
        print('  The q-grid must divide the k-grid (%d).' % kg)
    cfg['kgrid'], cfg['qgrid'] = [kg] * 3, [qg] * 3
    cfg['irreducible'] = True
    if source == 'dfpt':
        cfg['irreducible'] = pd.ask_yes_no(
            'nscf on the irreducible wedge (K_POINTS automatic, unfolded by symmetry)? '
            '(No = explicit full list)',
            True,
        )
    else:
        print('  (EPW needs the explicit full k list: K_POINTS crystal)')
        cfg['irreducible'] = False
    cfg['nbnd'] = pd.ask_int('Bands in the nscf run (nbnd, > number of PAO orbitals)', 0)
    cfg['exclude_bands'] = []
    if source == 'epw':
        while True:
            try:
                cfg['exclude_bands'] = parse_exclude_bands(
                    pd.ask(
                        'Bands to exclude from the coupling (EPW exclude_bands, e.g. 1:5; blank = none)',
                        '',
                    )
                )
                break
            except ValueError as err:
                print('  %s' % err)
    default_masses = species_masses(cfg['pw_template_text'] or '')
    masses = pd.ask(
        'Atomic mass of each species (amu), comma-separated',
        ', '.join('%g' % m for m in default_masses),
    )
    cfg['masses_amu'] = _floats(masses)
    cfg['nelec'] = pd.ask_int('Valence electrons (nelec)', 0)
    cfg['metal'] = pd.ask_yes_no('Is the system a metal?', False)
    cfg['basisdir'] = pd.ask('Basis directory name (paoflow-genbasis-ps)', 'BASIS_PS')
    cfg['configuration'] = 'standard'
    cfg['pthr'] = pd.ask_float('Projectability threshold (pthr)', 0.95)
    cfg['spin_degeneracy'] = None
    cfg['long_range'] = True
    cfg['quadrupoles'] = None
    cfg['cache_dir'] = 'cache'
    if not cfg['metal']:
        print(
            '  (ph.in gets epsil = .true.: the dielectric tensor and Born charges of the '
            'long-range model; quadrupoles come from coupling/quadrupoles.py)'
        )
    default = (
        ['superconductivity', 'transport', 'arpes']
        if cfg['metal']
        else ['optics', 'transport', 'arpes']
    )
    choices = [
        'phonons',
        'coupling',
        'superconductivity',
        'optics',
        'transport',
        'arpes',
        'thermal',
    ]
    cfg['workflows'] = ask_workflows(pd.ask, choices, default)
    cfg['optics_nscf'] = False
    if 'superconductivity' in cfg['workflows']:
        _collect_superconductivity(cfg, pd, kg)
    if 'optics' in cfg['workflows']:
        _collect_optics(cfg, pd, kg)
    if 'transport' in cfg['workflows']:
        _collect_transport(cfg, pd, kg)
    if 'arpes' in cfg['workflows']:
        _collect_arpes(cfg, pd, kg)
    if 'thermal' in cfg['workflows']:
        _collect_thermal(cfg, pd, kg)
    if 'phonons' in cfg['workflows']:
        cfg['phonons'] = dict(units='cm-1', mesh=(24, 24, 24), sigma=3.0)
    if 'coupling' in cfg['workflows']:
        cfg['coupling'] = dict(nk=24, direction=(1.0, 0.0, 0.0))
    if cfg['optics_nscf']:
        cfg['optics_kgrid'] = [pd.ask_int('Optics nscf: Monkhorst-Pack grid per axis', 8)] * 3
        cfg['optics_nbnd'] = pd.ask_int(
            'Optics nscf: bands (more than the orbitals of the extended PAO basis)', 60
        )
    if cfg['nbnd'] <= 0 or not cfg['masses_amu'] or cfg['nelec'] <= 0:
        print(
            '\nNote: nbnd, masses_amu and nelec are system-specific; edit system.py and qe/ before running.'
        )
    return cfg


def _collect_superconductivity(cfg, pd, kg: int) -> None:
    nq = pd.ask_int('Superconductivity: dense phonon q-grid per axis (NQ_DENSE)', 4 * kg)
    nk = ask_divisible(
        pd.ask_int,
        'Superconductivity: dense electron k-grid per axis (NK_DENSE)',
        8 * kg,
        nq,
        'NK_DENSE',
    )
    adaptive = pd.ask_float(
        'Adaptive Fermi-surface smearing prefactor a (Yates; 0.7 production, 0 = fixed width)', 0.7
    )
    sigma = pd.ask_float(
        'Fermi-surface smearing (eV): floor of the adaptive widths, or the fixed width (EPW degaussw)',
        0.001 if adaptive else 0.05,
    )
    mu_star = pd.ask_float('Coulomb pseudopotential mu* (for Tc)', 0.10)
    transport = pd.ask_yes_no(
        'Also the transport Eliashberg function (resistivity, infrared optics, emissivity)?',
        cfg.get('metal', False) and 'transport' in cfg['workflows'],
    )
    fs_coupling = pd.ask_yes_no(
        'Store the Fermi-surface pair coupling for the anisotropic Migdal-Eliashberg equations '
        '(needs nk == nq)?',
        nk == nq,
    )
    fsthick = (
        pd.ask_float('Fermi window of the anisotropic states (eV, EPW fsthick)', 0.2)
        if fs_coupling
        else None
    )
    temps = _floats(pd.ask('Migdal-Eliashberg temperatures (K): Tmin, Tmax, nstemp', '0.5, 10, 20'))
    if len(temps) == 3 and temps[2] >= 2:
        me_temps = [round(float(t), 3) for t in np.linspace(temps[0], temps[1], int(temps[2]))]
    else:
        me_temps = temps or [1.0, 2.0, 4.0]
    cfg['superconductivity'] = dict(
        nk=nk,
        nq=nq,
        sigma_ev=sigma,
        adaptive_smearing=adaptive or None,
        mu_star=mu_star,
        transport=transport,
        fs_coupling=fs_coupling,
        fsthick_ev=fsthick,
        me_temps=me_temps,
        me_linear_temps=(me_temps[0], me_temps[-1], len(me_temps)),
        me_aniso_temps=me_temps,
        wscut=0.1,
        wscut_aniso=0.5,
        degaussq_mev=0.15,
        degaussq_linear_mev=0.5,
        npade=90.0,
    )


def _collect_optics(cfg, pd, kg: int) -> None:
    nq = pd.ask_int('Optics: dense phonon q-grid per axis (EPW nqf)', kg)
    nk = ask_divisible(
        pd.ask_int, 'Optics: dense electron k-grid per axis (EPW nkf)', 2 * nq, nq, 'NK_DENSE'
    )
    electrons_save = (
        pd.ask(
            'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis, whose PAO '
            'Hamiltonian replaces the PBE one (blank = PBE electrons)',
            '',
        )
        or None
    )
    gap = pd.ask('Target indirect gap for a scissor (eV; blank = the Hamiltonian gap)', '')
    gap_reference = None
    if not electrons_save:
        gap_reference = (
            pd.ask(
                'QE save of an eACBN0 (DFT+U+V) run whose PAO gap is the target (blank = none)', ''
            )
            or None
        )
    varshni = _floats(
        pd.ask('Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)', '')
    )
    eps = pd.ask(
        'Screening correction of the coupling (Poncé 2018): target static dielectric constant '
        '(Si 11.94; blank = DFPT screening)',
        '',
    )
    emissivity = pd.ask_yes_no(
        'Also compute the thermal emissivity of a slab vs temperature (Kirchhoff)?', False
    )
    thickness, angles, direct_dir = 500.0, (0.0, 30.0, 60.0), None
    if emissivity:
        thickness = pd.ask_float('Slab thickness (um)', 500.0)
        angles = tuple(
            _floats(pd.ask('Emission angles (deg), comma-separated', '0, 30, 60')) or [0.0]
        )
        cfg['optics_nscf'] = pd.ask_yes_no(
            'Direct dielectric function from an extended-basis run (qe/optics nscf + optics/dielectric.py)?',
            True,
        )
        direct_dir = '../qe/optics/output' if cfg['optics_nscf'] else None
    omega = (
        pd.ask_float('Lowest photon energy (eV, EPW omegamin)', 0.01 if emissivity else 0.05),
        pd.ask_float('Highest photon energy (eV, EPW omegamax)', 3.0),
        pd.ask_float('Photon-energy step (eV, EPW omegastep)', 0.01 if emissivity else 0.05),
    )
    temps = _floats(
        pd.ask(
            'Temperatures (K), comma-separated',
            '300, 500, 700, 900, 1100, 1300, 1500' if emissivity else '300',
        )
    ) or [300.0]
    degauss = pd.ask_float('Delta smearing (eV, EPW degaussw)', 0.01 if emissivity else 0.05)
    fsthick = pd.ask_float('Energy window around E_F (eV, EPW fsthick)', 4.0)
    n_r = pd.ask(
        'Constant refractive index n_r for alpha (blank = n(omega) of the direct run, else 3.4)', ''
    )
    cfg['optics'] = dict(
        nk=nk,
        nq=nq,
        omega_ev=omega,
        temps_k=temps,
        degauss_ev=degauss,
        adaptive_smearing=None,
        fsthick_ev=fsthick,
        refractive_index=float(n_r) if n_r.strip() else None,
        position_dipoles=pd.ask_yes_no(
            'Position dipoles of the PAO orbitals in the velocity operator?', True
        ),
        target_gap_ev=float(gap) if gap.strip() else None,
        gap_reference_save=gap_reference,
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=electrons_save,
        screening_eps=float(eps) if eps.strip() else None,
        emissivity=emissivity,
        thickness_um=thickness,
        angles=angles,
        direct_dielectric_dir=direct_dir,
        kk_omega_max_ev=10.0,
        dielectric=dict(nfft=24, emin=0.01, emax=10.0, ne=1000, delta=0.1, intraband=True),
    )


def _collect_transport(cfg, pd, kg: int) -> None:
    if cfg.get('metal'):
        temps = _floats(
            pd.ask('Resistivity: temperatures (K)', '20, 50, 100, 200, 300, 400, 500, 600')
        )
        rho0 = pd.ask_float('Residual resistivity (micro-ohm cm)', 0.0)
        b_ee = pd.ask_float('Electron-electron (Fermi-liquid) coefficient b (1/eV)', 0.0)
        interband = pd.ask_yes_no(
            'Interband dielectric function from an extended-basis run (qe/optics nscf + '
            'optics/dielectric.py --interband-only) for the emissivity?',
            False,
        )
        if interband:
            cfg['optics_nscf'] = True
        cfg['transport'] = dict(
            temps_k=temps or [300.0],
            rho0_uohm_cm=rho0,
            b_ee_per_ev=b_ee,
            omega_ev=(0.005, 2.0, 0.005),
            thickness_um=float('inf'),
            angles=(0.0, 30.0, 60.0),
            interband_dielectric_dir='../qe/optics/output' if interband else None,
            extended_drude=True,
        )
        if interband and 'optics' not in cfg['workflows']:
            cfg['optics'] = dict(
                dielectric=dict(
                    nfft=36,
                    emin=0.002,
                    emax=6.0,
                    ne=3000,
                    delta=0.1,
                    intraband=False,
                    smearing='gauss',
                )
            )
        return
    nq = pd.ask_int('Mobility: dense phonon q-grid per axis (mobilities need >= 48)', 8 * kg)
    nk = ask_divisible(pd.ask_int, 'Mobility: dense electron k-grid per axis', nq, nq, 'NK_DENSE')
    temps = _floats(
        pd.ask('Temperatures (K), comma-separated', '200, 250, 300, 350, 400, 500')
    ) or [300.0]
    doping = pd.ask_float('Net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)', 0.0)
    window = pd.ask_float('Energy window from the band edges (eV, EPW fsthick)', 0.3)
    smearing = pd.ask_float('Delta smearing (eV, EPW degaussw)', 0.01)
    electrons_save = (
        pd.ask(
            'QE save of an eACBN0 (DFT+U+V, ortho-atomic) nscf in the same PAO basis (blank = PBE electrons)',
            '',
        )
        or None
    )
    gap = pd.ask(
        'Target indirect gap for the carrier statistics (eV; blank = the Hamiltonian gap)', ''
    )
    varshni = []
    if gap.strip() or electrons_save:
        varshni = _floats(
            pd.ask('Temperature-dependent gap: Varshni alpha (eV/K), beta (K) (blank = fixed)', '')
        )
    eps = pd.ask(
        'Screening correction of the coupling (Poncé 2018): target static dielectric constant '
        '(Si 11.94; blank = DFPT screening)',
        '',
    )
    cfg['transport'] = dict(
        nk=nk,
        nq=nq,
        temps_k=temps,
        doping_cm3=doping,
        window_ev=window,
        smearing_ev=smearing,
        adaptive_smearing=None,
        target_gap_ev=float(gap) if gap.strip() else None,
        varshni=tuple(varshni) if len(varshni) == 2 else None,
        electrons_save=electrons_save,
        screening_eps=float(eps) if eps.strip() else None,
    )


def _collect_arpes(cfg, pd, kg: int) -> None:
    metal = cfg.get('metal', False)
    photon = pd.ask_float('ARPES: photon energy (eV; He I 21.2)', 21.2)
    work = pd.ask_float('Work function (eV)', 4.5)
    direction = _floats(
        pd.ask('Path: end point of the line from Gamma (crystal coordinates)', '0.5 0 0.5')
    )
    label = pd.ask('Its label', 'X')
    sc = cfg.get('superconductivity', {})
    if metal:
        kinds = ['isotropic'] if 'superconductivity' in cfg['workflows'] else ['fan-migdal']
        if sc.get('fs_coupling'):
            kinds.append('anisotropic')
    else:
        kinds = ['fan-migdal']
    cfg['arpes'] = dict(
        photon_ev=photon,
        work_function_ev=work,
        polarization=(1.0, 0.0, 1.0),
        direction=tuple(direction) if len(direction) == 3 else (0.5, 0.0, 0.5),
        direction_label=label,
        window=(-0.1, 0.02, 401) if metal else (-13.0, 1.5, 581),
        resolution=(0.002, 0.0) if metal else (0.05, 0.0),
        self_energy=kinds,
        temps_k=[10.0] if metal else [10.0, 300.0, 600.0],
        segment_ev=0.12 if metal else None,
        npts=81 if metal else 41,
        nq=18 if metal else 12,
        eta_ev=0.005 if metal else 0.02,
        adaptive_smearing=0.7 if metal else None,
        intensity_power=1.0 if metal else 0.3,
    )


def _collect_thermal(cfg, pd, kg: int) -> None:
    metal = cfg.get('metal', False)
    temps = _floats(pd.ask('Thermal (ZG supercells): temperatures (K), comma-separated', '0, 300'))
    dims = pd.ask_int('ZG supercell per axis (3 for tests, 4 and more for production)', 3)
    kgrid = pd.ask_int('k-grid per axis of the supercell scf', max(1, -(-kg // dims)))
    nat = pd._read_namelist_value(cfg.get('pw_template_text') or '', 'nat')
    try:
        nat = max(1, int(float(str(nat).strip(' ,\'"'))))
    except (TypeError, ValueError):
        nat = 1
    default_bands = int(np.ceil(cfg['nbnd'] / nat)) if cfg.get('nbnd', 0) > 0 else 12
    nbnd_per_atom = pd.ask_int(
        'Bands per atom of the supercell scf (above the PAO orbitals per atom)',
        max(4, default_bands),
    )
    nconf = pd.ask_int('Configurations per temperature (seeds of the sign search)', 1)
    cfg['thermal'] = dict(
        temps_k=temps or [0.0, 300.0],
        supercell=(dims,) * 3,
        kgrid=(kgrid,) * 3,
        nbnd_per_atom=nbnd_per_atom,
        force_constants=None,
        seeds=list(range(max(1, nconf))),
        error_thresh=0.05,
        reference=True,
        path='G-X' if metal else 'L-G-X',
        sym_points=None,
        nk_per_seg=100,
        energy_window=(-0.3, 0.3, 601) if metal else (-2.0, 3.0, 1001),
        broadening_ev=0.005 if metal else 0.02,
        crossing_ev=0.0,
        half_window_ev=0.25 if metal else 0.5,
        dielectric=None if metal else dict(nfft=24, emin=0.05, emax=6.0, ne=1200, delta=0.05),
        fd_mode='random',
        fd_nconfig=60,
        fd_distance=0.03,
        lambda_nk=12,
        lambda_sigma_ev=0.05,
    )


def _detect_harmonic(workdir: str, prefix: str) -> str:
    """A ``<prefix>.dyn`` prefix whose ``dyn0`` exists under ``workdir`` (relative), else ``harmonic/<prefix>.dyn``."""
    found = sorted(glob.glob(os.path.join(workdir, '**', '*.dyn0'), recursive=True))
    preferred = [path for path in found if os.path.basename(path) == prefix + '.dyn0'] or found
    if preferred:
        return os.path.relpath(preferred[0][:-1], workdir)
    return os.path.join('harmonic', prefix + '.dyn')


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.replace(',', ' ').split()]


def collect_sscha(args, workdir: str) -> dict:
    """Prompts of the ``sscha`` workflow.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line options of ``paoflow-gen`` (``savedir``, ``prefix`` overrides).
    workdir : str
        Directory scanned for the scf save and existing ``ph.x`` dynamical matrices.

    Returns
    -------
    dict
        The configuration of :func:`build_sscha_script`.
    """
    det_save = detect_savedir(workdir)
    cfg = {'outputdir': 'output', 'workdir': workdir}
    cfg['savedir'] = args.savedir or ask(
        'Save directory of the scf (<prefix>.save)', det_save or 'pwscf.save'
    )
    cfg['prefix'] = args.prefix or ask('Prefix', detect_prefix(workdir, det_save))
    cfg['pp_dir'] = ask('Pseudopotential directory (blank or "HERE" = script directory)', 'HERE')
    temperatures = ask('Temperatures (K, comma-separated)', '300')
    cfg['temperatures'] = [float(t) for t in temperatures.replace(',', ' ').split()]
    cfg['supercell'] = ask_int('Supercell multiplicity (isotropic diagonal)', 2)
    cfg['harmonic_dyn'] = ask(
        'ph.x dynamical matrices (fildyn prefix <dir>/<prefix>.dyn; written by the harmonic phase if missing)',
        _detect_harmonic(workdir, cfg['prefix']),
    )
    qgrid = ask_int(
        'q-grid of the harmonic ph.x run if it is generated (multiple of the supercell; larger = better '
        'harmonic dispersion)',
        cfg['supercell'],
    )
    cfg['harmonic_qgrid'] = [qgrid] * 3
    kgrid = ask(
        'k-grid of the supercell force runs (e.g. "6 6 6"; blank = scf grid / supercell)', ''
    )
    cfg['kgrid'] = _ints(kgrid) if kgrid.strip() else None
    electrons = ask_choice(
        'Electrons of the force runs (parent = occupations of the scf; ionic = Fermi-Dirac at k_B T of each '
        'temperature, hot electrons; fixed = Fermi-Dirac at one temperature)',
        ['parent', 'ionic', 'fixed'],
        'parent',
    )
    if electrons == 'ionic':
        cfg['electronic_temperature'] = 'ionic'
    elif electrons == 'fixed':
        cfg['electronic_temperature'] = ask_float('  Electronic temperature (K)', 300.0)
    else:
        cfg['electronic_temperature'] = None
    sizes = ask('Configurations per population (comma-separated; the last repeats)',
                ', '.join(map(str, DEFAULT_POPULATIONS)))  # fmt: skip
    cfg['population_sizes'] = _ints(sizes)
    cfg['hessian_size'] = ask_int('Configurations of the free-energy Hessian population', 800)
    cfg['launcher'] = ask_choice(
        'Launcher of the pw.x runs (local = here; slurm = job arrays chained by step jobs; manual = write inputs)',
        ['local', 'slurm', 'manual'],
        'local',
    )
    cfg['qe_pw'] = ask("pw.x command (may include flags, e.g. 'pw.x -nk 4')", 'pw.x')
    cfg['qe_ph'] = cfg['qe_pw'].replace('pw.x', 'ph.x') if 'pw.x' in cfg['qe_pw'] else 'ph.x'
    if cfg['launcher'] == 'slurm':
        print('\nSlurm settings of one configuration (array task) and of the step job:')
        cfg['mpi_qe'] = ask('MPI launcher inside the jobs', 'srun')
        partition = ask('  partition', 'normal')
        nodes = ask_int('  nodes per configuration', 1)
        ntasks = ask_int('  MPI tasks per configuration', 32)
        walltime = ask('  wall time per configuration', '02:00:00')
        account = ask('  account (blank = default)', '')
        directives = ['#SBATCH -p %s' % partition, '#SBATCH -N %d' % nodes, '#SBATCH -n %d' % ntasks,
                      '#SBATCH -t %s' % walltime]  # fmt: skip
        if account:
            directives.append('#SBATCH -A %s' % account)
        cfg['slurm_directives'] = directives
        step_tasks = ask_int('  cores of the step job (python minimisation)', 4)
        step_time = ask('  wall time of the step job', '02:00:00')
        step = ['#SBATCH -p %s' % partition, '#SBATCH -N 1', '#SBATCH -n %d' % step_tasks,
                '#SBATCH -t %s' % step_time]  # fmt: skip
        if account:
            step.append('#SBATCH -A %s' % account)
        cfg['slurm_step_directives'] = step
        cfg['slurm_concurrent'] = ask_int('  array tasks running at the same time', 40)
        setup = ask('  shell line that loads QE and PAOFLOW in a job (blank = none)', '')
        cfg['env_setup'] = [setup] if setup else []
        cfg['python'] = ask('  python of the step jobs', 'python')
    else:
        cfg['mpi_qe'] = ask('MPI command for pw.x / ph.x', 'mpirun -np 4')
    return cfg
