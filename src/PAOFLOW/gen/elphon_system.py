"""``paoflow-gen elphon``: write an electron-phonon system folder from a ``pw.x`` input.

The folder has the layout of ``examples/elphon/<System>``: ``system.py`` (the only file with
values, an :class:`~PAOFLOW.elphon.workflows.ElphonSystem` plus the per-workflow defaults),
``qe/`` (scf, ``ph.x``, nscf and, optionally, optics-nscf inputs with ``run_qe.sh``) and one
subfolder per selected workflow (``phonons/``, ``coupling/``, ``superconductivity/``,
``optics/``, ``transport/``, ``arpes/``, ``thermal/``) holding copies of the drivers of
``PAOFLOW/gen/elphon_templates`` (the same files as the examples).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elphon_templates')

#: Driver files of every workflow folder (relative to the template directory).
WORKFLOWS: dict[str, tuple[str, ...]] = {
    'phonons': ('phonons/phonons.py',),
    'coupling': ('coupling/quadrupoles.py', 'coupling/long_range_check.py'),
    'superconductivity': (
        'superconductivity/eliashberg.py',
        'superconductivity/me.py',
        'superconductivity/me_aniso.py',
        'superconductivity/plot_me.py',
    ),
    'optics': (
        'optics/absorption.py',
        'optics/emissivity.py',
        'optics/dielectric.py',
        'optics/plot_absorption.py',
    ),
    'optics-metal': ('optics/dielectric.py',),
    'transport': ('transport/mobility.py', 'transport/plot_mobility.py'),
    'transport-metal': ('transport/resistivity.py', 'transport/plot_resistivity.py'),
    'arpes': ('arpes/arpes.py',),
    'thermal': ('thermal/zg.py',),
}
QE_HELPERS = ('qe/kmesh.py',)

WRITE_UKK_TEMPLATE = '''#!/usr/bin/env python3
"""Write {ukk} for epw.x with wannierize = .false. (run in this directory before epw.x).

EPW reads the band bookkeeping from this file only: the {nbnd} nscf bands, {exclusion}
No Wannier functions are built; the identity rotations only feed
EPW's Wannier stage, which PAOFLOW does not use.  The empty {stem}.bvec /
{stem}.mmn written alongside let that stage finish.
"""

from PAOFLOW.gen.epw_inputs import write_placeholder_ukk

write_placeholder_ukk('{ukk}', nbnd={nbnd}, nk_total={nk_total}, exclude_bands={excluded}, nelec={nelec})
print('Wrote {ukk} (and the empty wannier90 stubs {stem}.bvec, {stem}.mmn)')
'''


# ----------------------------------------------------------------------------- rendering
def _value(value: Any) -> str:
    """``repr`` of a configuration value (tuples for grids, lists for sequences)."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, float) and value == float('inf'):
        return "float('inf')"
    return repr(value)


def _format_dict(name: str, values: dict[str, Any], comments: dict[str, str] | None = None) -> str:
    """``NAME = dict(\\n    key=value,  # comment\\n)`` with one entry per line."""
    comments = comments or {}
    if not values:
        return '%s = {}\n' % name
    lines = ['%s = dict(' % name]
    for key, value in values.items():
        if isinstance(value, dict):
            inner = ', '.join('%s=%s' % (k, _value(v)) for k, v in value.items())
            entry = '    %s=dict(%s),' % (key, inner)
        else:
            entry = '    %s=%s,' % (key, _value(value))
        comment = comments.get(key)
        lines.append(entry + ('  # %s' % comment if comment else ''))
    lines.append(')')
    return '\n'.join(lines) + '\n'


SYSTEM_COMMENTS = {
    'coupling_source': "'dfpt': PAOFLOW's vertex from ph.x dvscf + the nscf save; 'epw': .epb files; 'fd': finite differences of a supercell (fd_dir)",
    'qe_dir': 'qe/phonon (scf, ph.x), qe/nscf (<prefix>.save), qe/optics',
    'coarse_grid': 'nscf k-grid',
    'qgrid': 'ph.x q-grid (divides the k-grid)',
    'masses_amu': 'one per species, ATOMIC_SPECIES order',
    'nelec': 'valence electrons',
    'basis_dir': 'paoflow-genbasis-ps tables',
    'pthr': 'projectability threshold',
    'long_range': 'dipole (Fröhlich) + quadrupole model when polar or quadrupoles is set',
    'quadrupoles': "'coupling/quadrupole.fmt' from coupling/quadrupoles.py",
    'cache_dir': 'coarse vertex g(R_e, R_p), shared by every workflow',
    'spin_degeneracy': '2 without spin-orbit coupling, 1 with it (None: from the save)',
    'fd_dir': 'displaced runs of the supercell (qe/finite_differences.py --write, run_fd.sh)',
    'save_dir': 'save of the undisplaced supercell: the electrons of the coupling',
}
#: ``system.py`` comments of the finite-difference source, where the cell is the supercell.
FD_SYSTEM_COMMENTS = {
    'qe_dir': 'qe/phonon (scf of the primitive cell), qe/fd (the supercell runs)',
    'coarse_grid': 'k-grid of the supercell runs',
    'qgrid': 'Gamma only: the modes of the supercell (the primitive q-grid of the supercell)',
    'nelec': 'valence electrons of the supercell',
    'long_range': 'no dipole model: the supercell vertex is used as computed',
}
FD_COMMENTS = {
    'supercell': 'ideal supercell (primitive cells per axis): the unit cell of the finite differences',
    'mode': "'cartesian': every atom along +-x, +-y, +-z (6 N cells); 'random': 2 nconfig cells",
    'distance': 'displacement amplitude (bohr)',
    'nconfig': "random configurations ('random' mode; None: 3 N)",
    'seed': 'seed of the random directions',
    'nbnd': 'bands of the supercell runs (None: the PAO basis of the supercell)',
    'primitive_save': 'scf save of the primitive cell (run_qe.sh): structure and settings of the inputs',
    'pseudo_dir': 'pseudo_dir of the supercell inputs, as seen from qe/fd',
    'prefix': 'prefix of the primitive cell (the runs are <prefix>_sc_ref, <prefix>_sc_dispNNN)',
}
SUPERCONDUCTIVITY_COMMENTS = {
    'nk': 'dense electron grid per axis (EPW nkf)',
    'nq': 'dense phonon grid per axis (EPW nqf); nk % nq == 0',
    'sigma_ev': 'Fermi-surface smearing (EPW degaussw): the floor of the adaptive widths',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (0.7 production; None: fixed smearing)',
    'mu_star': 'Coulomb pseudopotential',
    'transport': 'also alpha^2F_tr (transport/resistivity.py)',
    'fs_coupling': 'Fermi-surface pair coupling for me_aniso.py (needs nk == nq)',
    'fsthick_ev': 'Fermi window of the anisotropic states (EPW fsthick)',
    'me_temps': 'temperatures (K) of the isotropic Migdal-Eliashberg equations',
    'me_linear_temps': '(Tmin, Tmax, nstemp) of the linearised kernel',
    'me_aniso_temps': 'temperatures (K) of the anisotropic equations',
    'wscut': 'Matsubara cutoff (eV, EPW wscut)',
    'wscut_aniso': 'Matsubara cutoff of the anisotropic equations (eV)',
    'degaussq_mev': 'phonon smearing of alpha^2F for the equations (meV, EPW degaussq)',
    'degaussq_linear_mev': '... for the linearised kernel (meV)',
    'npade': 'percentage of Matsubara points of the Padé approximant',
    'real_axis_max_mev': 'upper limit of the real-axis panels of plot_me.py',
}
OPTICS_COMMENTS = {
    'nk': 'dense electron grid per axis (EPW nkf)',
    'nq': 'dense phonon grid per axis (EPW nqf); nk % nq == 0',
    'omega_ev': 'photon energies (min, max, step)',
    'temps_k': 'temperatures (K)',
    'degauss_ev': 'delta smearing (EPW degaussw); the floor with adaptive_smearing',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (None: fixed smearing)',
    'fsthick_ev': 'states within this window of the Fermi level (EPW fsthick)',
    'refractive_index': 'constant n_r of alpha (None: n(omega) of the direct run, else 3.4)',
    'position_dipoles': 'position dipoles of the PAO orbitals in the velocity',
    'target_gap_ev': 'scissor target (eV); None: the Hamiltonian gap',
    'gap_reference_save': 'QE save of an eACBN0 run whose PAO gap is the target',
    'varshni': '(alpha eV/K, beta K) of a temperature-dependent gap',
    'electrons_save': 'QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)',
    'screening_eps': 'target dielectric constant of the Poncé screening correction',
    'emissivity': 'also the thermal emissivity of a slab',
    'thickness_um': 'slab thickness (um)',
    'angles': 'emission angles (deg)',
    'direct_dielectric_dir': 'output of optics/dielectric.py (extended basis)',
    'kk_omega_max_ev': 'Kramers-Kronig range of the direct term without an external eps',
    'dielectric': 'optics/dielectric.py: interpolated grid, window and broadening',
}
TRANSPORT_COMMENTS = {
    'nk': 'dense electron grid per axis (mobilities need >= 48)',
    'nq': 'dense phonon grid per axis; nk % nq == 0',
    'temps_k': 'temperatures (K)',
    'doping_cm3': 'net doping (cm^-3; > 0 n type, < 0 p type, 0 intrinsic)',
    'window_ev': 'states within this window of the band edges (EPW fsthick)',
    'smearing_ev': 'delta smearing (EPW degaussw); the floor with adaptive_smearing',
    'adaptive_smearing': 'Yates prefactor of the adaptive widths (None: fixed smearing)',
    'target_gap_ev': 'scissor of the carrier statistics (eV)',
    'varshni': '(alpha eV/K, beta K) of a temperature-dependent gap',
    'electrons_save': 'QE save of an eACBN0 nscf in the same basis (eACBN0 electrons)',
    'screening_eps': 'target dielectric constant of the Poncé screening correction',
    'rho0_uohm_cm': 'residual resistivity (micro-ohm cm)',
    'b_ee_per_ev': 'Fermi-liquid coefficient b (1/eV)',
    'omega_ev': 'photon energies (min, max, step) of the metal optics',
    'thickness_um': 'film thickness (um; inf = opaque)',
    'angles': 'emission angles (deg)',
    'interband_dielectric_dir': 'output of optics/dielectric.py --interband-only',
    'extended_drude': 'frequency-dependent rate + mass enhancement (False: DC rate)',
}
ARPES_COMMENTS = {
    'photon_ev': 'photon energy (He I: 21.2)',
    'work_function_ev': 'work function (eV)',
    'polarization': 'light polarisation (Cartesian)',
    'direction': 'end point of the line from Gamma (crystal coordinates)',
    'direction_label': 'its name',
    'window': '(E_min, E_max, n) of the energy window (eV, relative to E_F)',
    'resolution': '(energy, momentum) resolution (eV, 1/A)',
    'self_energy': "any of 'isotropic', 'anisotropic', 'fan-migdal'",
    'temps_k': 'temperatures (K)',
    'segment_ev': 'segment of +- this energy across the Fermi crossing (None: the full line)',
    'npts': 'points of the path',
    'nq': 'dense q-grid of the Fan-Migdal run',
    'eta_ev': 'Lorentzian width / floor (eV) of the Fan-Migdal run',
    'adaptive_smearing': 'adaptive widths a |v| dq of the Fan-Migdal run (metals)',
    'intensity_power': 'power scale of the band maps (weak bands next to strong ones)',
}
THERMAL_COMMENTS = {
    'temps_k': 'temperatures (K) of the ZG supercells',
    'supercell': 'ZG supercell (diagonal, in primitive cells)',
    'kgrid': 'k-grid of the supercell scf (None: the coarse grid divided by the supercell)',
    'nbnd_per_atom': 'bands per atom of the supercell scf (above the PAO orbitals per atom)',
    'force_constants': 'q2r.x force constants (None: phonons/output/<prefix>.fc of phonons.py)',
    'seeds': 'seeds of the sign search: one cell per seed and temperature (ensemble average)',
    'error_thresh': 'accepted relative error of the mean-square displacements (ZG.x error_thresh)',
    'reference': 'also the undisplaced supercell (offset of the supercell bands)',
    'path': 'k-path of the unfolded spectral function',
    'sym_points': 'its points in the primitive reciprocal basis (None: the lattice defaults)',
    'nk_per_seg': 'k-points per path segment',
    'energy_window': '(E_min, E_max, n) of the spectral function (eV, relative to E_F)',
    'broadening_ev': 'Gaussian width of the spectral function (eV)',
    'crossing_ev': 'energy (eV) at which the width of the crossing band is measured',
    'half_window_ev': 'half width (eV) of the window of the peak and moment analysis',
    'dielectric': 'dielectric function of the ZG cells (nfft, emin, emax, ne, delta, electronic_temperature; None: skip)',
    'fd_mode': "displaced runs of a ZG cell (--displace): 'random' (2 fd_nconfig cells) or 'cartesian' (6 N)",
    'fd_nconfig': 'random configurations per ZG cell',
    'fd_distance': 'displacement amplitude (bohr) of the displaced runs',
    'lambda_nk': 'dense k per axis of the supercell for lambda(T) of the ZG cells (--lambda)',
    'lambda_sigma_ev': 'Fermi-surface smearing (eV) of lambda(T)',
}
PHONONS_COMMENTS = {
    'units': "'cm-1' or 'THz'",
    'mesh': 'q-mesh of the DOS',
    'sigma': 'DOS broadening',
}
COUPLING_COMMENTS = {
    'nk': 'dense grid of the path points of long_range_check.py',
    'direction': 'Cartesian direction of q (2 pi / alat)',
}


def render_system_py(cfg: dict[str, Any]) -> str:
    """Text of ``system.py``: ``NAME``, ``SYSTEM`` and the per-workflow default dictionaries.

    Parameters
    ----------
    cfg : dict
        ``name``, ``prefix``, ``coupling_source``, ``kgrid``, ``qgrid``, ``masses_amu``,
        ``nelec``, ``basisdir``, ``configuration``, ``pthr``, ``long_range``,
        ``quadrupoles`` (path or ``None``), ``cache_dir``, ``spin_degeneracy`` and the
        optional dictionaries ``superconductivity``, ``optics``, ``transport``, ``arpes``,
        ``thermal``, ``phonons``, ``coupling`` (missing ones are written empty).

    Returns
    -------
    str
        The module text.
    """
    fd = cfg.get('coupling_source') == 'fd'
    prefix = cfg['prefix']
    system = dict(
        coupling_source=cfg.get('coupling_source', 'dfpt'),
        qe_dir='qe',
        coarse_grid=tuple(int(n) for n in cfg['kgrid']),
        qgrid=tuple(int(n) for n in cfg['qgrid']),
        masses_amu=[float(m) for m in cfg['masses_amu']],
        nelec=cfg['nelec'],
        basis_dir=cfg.get('basisdir', 'BASIS_PS'),
        configuration=cfg.get('configuration', 'standard'),
        pthr=float(cfg.get('pthr', 0.95)),
        long_range=bool(cfg.get('long_range', True)),
        quadrupoles=cfg.get('quadrupoles'),
        cache_dir=cfg.get('cache_dir', 'cache'),
        spin_degeneracy=cfg.get('spin_degeneracy'),
    )
    comments = dict(SYSTEM_COMMENTS)
    if fd:
        # The cell of the coupling is the ideal supercell: its runs, its save, its electrons.
        from ..elphon.workflows.finite_differences import reference_save, supercell_prefix

        settings = cfg['finite_differences']
        cells = int(np.prod(settings['supercell']))
        prefix = supercell_prefix(cfg['prefix']) + '_ref'
        system['qgrid'] = (1, 1, 1)
        system['nelec'] = cfg['nelec'] * cells
        system['long_range'] = False
        del system['quadrupoles']  # no dipole or quadrupole model on the supercell vertex
        system['fd_dir'] = 'qe/fd'
        system['save_dir'] = reference_save('qe/fd', cfg['prefix'])
        system['fd_options'] = dict(cfg.get('fd_options') or {})
        comments.update(FD_SYSTEM_COMMENTS)
        comments['nelec'] = '%s per primitive cell x %d cells' % (_value(cfg['nelec']), cells)
    lines = [
        '#!/usr/bin/env python3',
        '"""%s: configuration of the electron-phonon workflows (one folder per property).'
        % cfg['name'],
        '',
        'Edit the values here; every driver of the workflow folders reads them through',
        '``PAOFLOW.elphon.workflows.load_system``.  Paths are relative to this directory.',
        '"""',
        '',
        'import os',
        '',
        'from PAOFLOW.elphon.workflows import ElphonSystem',
        '',
        'HERE = os.path.dirname(os.path.abspath(__file__))',
        'NAME = %r' % cfg['name'],
        '',
        'SYSTEM = ElphonSystem(',
        '    prefix=%r,%s' % (prefix, '  # the undisplaced supercell (qe/fd)' if fd else ''),
        '    root=HERE,',
    ]
    for key, value in system.items():
        comment = comments.get(key)
        lines.append('    %s=%s,%s' % (key, _value(value), '  # %s' % comment if comment else ''))
    lines += [')', '']
    if fd:
        lines += [
            '# The displaced supercells of qe/finite_differences.py (coupling_source fd).',
            _format_dict('FINITE_DIFFERENCES', cfg['finite_differences'], FD_COMMENTS).rstrip('\n'),
            '',
        ]
    lines += ['# Defaults of the workflow drivers (command-line options override them).']
    for name, comments in (
        ('SUPERCONDUCTIVITY', SUPERCONDUCTIVITY_COMMENTS),
        ('OPTICS', OPTICS_COMMENTS),
        ('TRANSPORT', TRANSPORT_COMMENTS),
        ('ARPES', ARPES_COMMENTS),
        ('THERMAL', THERMAL_COMMENTS),
        ('PHONONS', PHONONS_COMMENTS),
        ('COUPLING', COUPLING_COMMENTS),
    ):
        lines.append(_format_dict(name, cfg.get(name.lower(), {}) or {}, comments).rstrip('\n'))
    return '\n'.join(lines) + '\n'


def render_run_qe(cfg: dict[str, Any]) -> str:
    """Text of ``qe/run_qe.sh``: scf, ``ph.x``, the nscf (and the EPW / optics steps).

    Parameters
    ----------
    cfg : dict
        ``prefix``, ``coupling_source``, ``optics_nscf`` (bool), ``metal`` (bool).

    Returns
    -------
    str
        The shell script (``NP`` processes, run from ``qe/``).
    """
    prefix = cfg['prefix']
    epw = cfg.get('coupling_source', 'dfpt') == 'epw'
    if cfg.get('coupling_source') == 'fd':
        return render_fd_run_qe(cfg)
    lines = [
        '#!/bin/bash',
        '# QE steps of the electron-phonon workflows (run from this directory): NP=8 bash run_qe.sh',
        '# The PAOFLOW_* variables (set by the paoflow-studio Runs tab) take precedence: the MPI',
        '# prefix, the Quantum ESPRESSO bin folder and the pw.x / ph.x options (PAOFLOW.utils.qe_launch).',
        'set -e',
        'NP=${NP:-8}',
        'MPI=${PAOFLOW_MPI_QE-mpirun -np $NP}',
        'BIN=${PAOFLOW_QE_BIN:+$PAOFLOW_QE_BIN/}',
        'PW=${PW:-${BIN}pw.x}; PH=${PH:-${BIN}ph.x}',
        'OPT=${PAOFLOW_QE_OPTIONS:-}',
        'log() { echo "$(date +%T) $*"; }',
        '',
        'cd phonon',
        'log scf;  $MPI $PW $OPT -in scf.in > scf.out',
        'log ph.x; $MPI $PH $OPT -in ph.in > ph.out        # irreducible q, fildvscf -> _ph0/',
    ]
    if epw:
        lines.append(
            'log pp.py; python3 ${EPW_PP:-pp.py}                 # <q-e>/EPW/bin/pp.py: collects '
            'dvscf, patterns, dyn into save/'
        )
    lines += [
        'cd ../nscf',
        'mkdir -p %s.save' % prefix,
        'cp ../phonon/%s.save/charge-density.dat ../phonon/%s.save/data-file-schema.xml %s.save/'
        % (prefix, prefix, prefix),
        'log nscf; $MPI $PW $OPT -in nscf.in > nscf.out',
    ]
    if epw:
        lines += [
            'log ukk;  python3 write_ukk.py                         # %s.ukk for wannierize = .false.'
            % prefix,
            'log epw;  $MPI ${EPW:-${BIN}epw.x} -nk $NP -in epw.in > epw.out   # writes %s.epb*'
            % prefix,
        ]
    if cfg.get('optics_nscf'):
        lines += [
            'cd ../optics',
            'mkdir -p %s.save' % prefix,
            'cp ../phonon/%s.save/charge-density.dat ../phonon/%s.save/data-file-schema.xml '
            '%s.save/' % (prefix, prefix, prefix),
            'log optics nscf; $MPI $PW -nk $NP -in nscf.in > nscf.out   # one k point per pool',
        ]
    lines += ['cd ..', 'log done']
    return '\n'.join(lines) + '\n'


def render_fd_run_qe(cfg: dict[str, Any]) -> str:
    """Text of ``qe/run_qe.sh`` for the finite-difference source: the scf of the primitive cell.

    The displaced supercells come next (``finite_differences.py --write``, then
    ``fd/run_fd.sh``); there is no ``ph.x`` and no nscf.

    Parameters
    ----------
    cfg : dict
        The generator configuration (``prefix``).

    Returns
    -------
    str
    """
    return (
        '\n'.join(
            [
                '#!/bin/bash',
                '# scf of the primitive cell (run from this directory): NP=8 bash run_qe.sh',
                '# Its save gives the structure and settings of the displaced supercells:',
                '#   python finite_differences.py --write; (cd fd && NP=8 bash run_fd.sh)',
                '# The PAOFLOW_* variables (set by the paoflow-studio Runs tab) take precedence: the MPI',
                '# prefix, the Quantum ESPRESSO bin folder and the pw.x options (PAOFLOW.utils.qe_launch).',
                'set -e',
                'NP=${NP:-8}',
                'MPI=${PAOFLOW_MPI_QE-mpirun -np $NP}',
                'BIN=${PAOFLOW_QE_BIN:+$PAOFLOW_QE_BIN/}',
                'PW=${PW:-${BIN}pw.x}',
                'OPT=${PAOFLOW_QE_OPTIONS:-}',
                'log() { echo "$(date +%T) $*"; }',
                '',
                'cd phonon',
                'log scf;  $MPI $PW $OPT -in scf.in > scf.out',
                'cd ..',
                'log "done: python finite_differences.py --write"',
            ]
        )
        + '\n'
    )


def write_ukk_script(cfg: dict[str, Any]) -> str:
    """Text of ``qe/nscf/write_ukk.py`` (placeholder ``<prefix>.ukk`` for ``wannierize = .false.``)."""
    from .epw_inputs import exclude_bands_string

    excluded = [int(b) for b in cfg.get('exclude_bands', [])]
    if excluded:
        exclusion = (
            'of which\nbands %s are excluded from the electron-phonon calculation, matching\n'
            '``bands_skipped`` in epw.in.' % exclude_bands_string(excluded).split('= ')[1]
        )
    else:
        exclusion = 'all of them kept in the\nelectron-phonon calculation.'
    return WRITE_UKK_TEMPLATE.format(
        ukk=cfg['prefix'] + '.ukk',
        stem=cfg['prefix'],
        nbnd=int(cfg['nbnd']),
        exclusion=exclusion,
        nk_total=' * '.join(str(int(n)) for n in cfg['kgrid']),
        excluded=repr(excluded),
        nelec=int(cfg['nelec']),
    )


def qe_inputs(cfg: dict[str, Any]) -> dict[str, str]:
    """QE (and EPW) inputs of the system folder, ``{relative path: text}``.

    Parameters
    ----------
    cfg : dict
        ``prefix``, ``kgrid``, ``qgrid``, ``nbnd``, ``nelec``, ``masses_amu``,
        ``pw_template_text`` (a ``pw.x`` input; without it only ``qe/nscf/nscf.kpoints`` is
        written), ``pseudo_dir``, ``coupling_source``, ``irreducible`` (nscf on the
        irreducible wedge), ``metal`` (no ``epsil``), ``exclude_bands``, ``optics_nscf``,
        ``optics_kgrid``, ``optics_nbnd``.

    Returns
    -------
    dict
        ``qe/phonon/scf.in``, ``qe/phonon/ph.in``, ``qe/nscf/nscf.in`` (or ``nscf.kpoints``),
        ``qe/run_qe.sh`` and, for the EPW source, ``qe/nscf/epw.in`` and
        ``qe/nscf/write_ukk.py``; with ``optics_nscf`` also ``qe/optics/nscf.in``.
    """
    from .epw_inputs import (
        automatic_kpoints_card,
        epw_input,
        kpoints_card,
        nscf_input,
        optics_nscf_input,
        ph_input,
        scf_input,
    )

    prefix = cfg['prefix']
    kgrid, qgrid = tuple(cfg['kgrid']), tuple(cfg['qgrid'])
    epw = cfg.get('coupling_source', 'dfpt') == 'epw'
    explicit = epw or not cfg.get('irreducible', True)
    template = cfg.get('pw_template_text')
    pseudo_dir = cfg.get('pseudo_dir') or None
    files = {}
    if cfg.get('coupling_source') == 'fd':
        # The scf of the primitive cell only: the supercells come from finite_differences.py.
        if template:
            files['qe/phonon/scf.in'] = scf_input(template, prefix, pseudo_dir)
        files['qe/run_qe.sh'] = render_run_qe(cfg)
        return files
    if template:
        files['qe/phonon/scf.in'] = scf_input(template, prefix, pseudo_dir)
        files['qe/nscf/nscf.in'] = nscf_input(
            template, prefix, kgrid, int(cfg['nbnd']), pseudo_dir, explicit=explicit
        )
        if cfg.get('optics_nscf'):
            files['qe/optics/nscf.in'] = optics_nscf_input(
                template, prefix, tuple(cfg.get('optics_kgrid', (8, 8, 8))),
                int(cfg.get('optics_nbnd', 60)), pseudo_dir,
            )  # fmt: skip
    else:
        files['qe/nscf/nscf.kpoints'] = (
            kpoints_card(kgrid) if explicit else automatic_kpoints_card(kgrid)
        )
    files['qe/phonon/ph.in'] = ph_input(
        prefix, qgrid, epsil=not cfg.get('metal', False), tr2_ph=cfg.get('tr2_ph', 1.0e-16)
    )
    if epw:
        excluded = [int(b) for b in cfg.get('exclude_bands', [])]
        files['qe/nscf/epw.in'] = epw_input(
            prefix, cfg['masses_amu'], kgrid, qgrid, int(cfg['nbnd']) - len(excluded),
            dvscf_dir='../phonon/save', exclude_bands=excluded,
        )  # fmt: skip
        files['qe/nscf/write_ukk.py'] = write_ukk_script(cfg)
    files['qe/run_qe.sh'] = render_run_qe(cfg)
    return files


def workflow_files(cfg: dict[str, Any]) -> list[str]:
    """Template files (relative paths) of the workflows selected in ``cfg['workflows']``."""
    files = list(QE_HELPERS)
    if cfg.get('coupling_source') == 'fd':
        files.append('qe/finite_differences.py')
    for name in cfg.get('workflows', ()):
        key = name
        if name in ('transport', 'optics') and cfg.get('metal'):
            key = name + '-metal'  # resistivity instead of mobility; the interband eps only
        files += list(WORKFLOWS[key])
    return files


def system_folder_files(cfg: dict[str, Any]) -> list:
    """The files of the system folder, before anything is written.

    Parameters
    ----------
    cfg : dict
        The generator configuration (:func:`render_system_py`, :func:`qe_inputs`,
        ``workflows``, ``metal``).

    Returns
    -------
    list of PAOFLOW.gen.outputs.GeneratedFile
        ``system.py``, the QE inputs (shell scripts executable) and the copied drivers, with
        paths relative to the system folder.
    """
    from .outputs import GeneratedFile

    texts = {'system.py': render_system_py(cfg)}
    texts.update(qe_inputs(cfg))
    files = [
        GeneratedFile(rel, text=text, executable=rel.endswith('.sh')) for rel, text in texts.items()
    ]
    # The plot_*.py scripts of the workflows come with them unless cfg['plot'] is False.
    plots = cfg.get('plot', True)
    files += [
        GeneratedFile(rel, template=os.path.join(TEMPLATE_DIR, rel))
        for rel in workflow_files(cfg)
        if plots or not os.path.basename(rel).startswith('plot_')
    ]
    return files


#: Empty folders of every system folder (the shared vertex cache and the figures).
SYSTEM_FOLDERS = ('cache', 'figures')


def write_system_folder(cfg: dict[str, Any], out: str, force: bool = False) -> list[str]:
    """Write ``system.py``, the QE inputs and the workflow drivers of a system folder.

    Parameters
    ----------
    cfg : dict
        The generator configuration (:func:`render_system_py`, :func:`qe_inputs`,
        ``workflows``, ``metal``).
    out : str
        The system folder (created).
    force : bool, optional
        Overwrite existing files (otherwise they are skipped with a message).

    Returns
    -------
    list of str
        Written files.
    """
    out = os.path.abspath(out)
    written = []
    for generated in system_folder_files(cfg):
        path = os.path.join(out, generated.path)
        if os.path.exists(path) and not force:
            print('  %s exists (use --force to overwrite)' % path)
            continue
        written.append(generated.write(out))
    for rel in SYSTEM_FOLDERS:
        os.makedirs(os.path.join(out, rel), exist_ok=True)
    return written


# ------------------------------------------------------------------------------ prompts


def collect_elphon_system(args, workdir: str) -> dict[str, Any]:
    """Prompt for the electron-phonon system configuration.

    The questions are :data:`PAOFLOW.gen.questions.elphon.ELPHON_ITEMS` (shared with the GUI).

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
    from .questions import collect

    return collect('elphon', args, workdir)


def main_elphon(args, workdir: str) -> int:
    """Run the electron-phonon generator: prompts, then the system folder.

    Parameters
    ----------
    args : argparse.Namespace
        ``paoflow-gen`` arguments (``prefix``, ``out``, ``force``).
    workdir : str
        Directory scanned for QE inputs.

    Returns
    -------
    int
        Exit status.
    """
    cfg = collect_elphon_system(args, workdir)
    written = write_system_folder(cfg, cfg['out'], force=args.force)
    print('\nWrote %d files under %s:' % (len(written), cfg['out']))
    for path in written:
        print('  %s' % os.path.relpath(path, cfg['out']))
    print('\nNext steps:')
    if not cfg.get('pw_template_text'):
        print(
            '  0) write qe/phonon/scf.in and qe/nscf/nscf.in (K_POINTS card of qe/nscf/nscf.kpoints, nbnd = %d)'
            % cfg['nbnd']
        )
    print(
        '  1) copy the pseudopotentials into qe/ and run the QE steps: (cd qe && NP=8 bash run_qe.sh)'
    )
    print('  2) paoflow-genbasis-ps --pseudo-dir qe --out %s --preset standard' % cfg['basisdir'])
    for name in cfg['workflows']:
        if name == 'phonons':
            print('  3) python phonons/phonons.py')
        elif name == 'coupling':
            print(
                '  3) python coupling/quadrupoles.py --write-inputs; (cd qe/phonon/quadrupoles && bash run.sh); '
                'python coupling/quadrupoles.py --extract; then set quadrupoles in system.py'
            )
        elif name == 'superconductivity':
            print(
                '  3) mpirun -np N python superconductivity/eliashberg.py; python superconductivity/me.py'
                + (
                    '; python superconductivity/me_aniso.py'
                    if cfg['superconductivity'].get('fs_coupling')
                    else ''
                )
                + '; python superconductivity/plot_me.py'
            )
        elif name == 'optics':
            print(
                '  3) mpirun -np N python optics/absorption.py; python optics/plot_absorption.py'
                + (
                    '   (optics/dielectric.py first, for --emissivity)'
                    if cfg['optics'].get('emissivity')
                    else ''
                )
            )
        elif name == 'transport' and cfg.get('metal'):
            print(
                '  3) python transport/resistivity.py (after eliashberg.py --transport); python transport/plot_resistivity.py'
            )
        elif name == 'transport':
            print(
                '  3) mpirun -np N python transport/mobility.py; python transport/plot_mobility.py'
            )
        elif name == 'arpes':
            print('  3) mpirun -np N python arpes/arpes.py')
        elif name == 'thermal':
            print(
                '  3) python thermal/zg.py --write; (cd thermal/zg && NP=8 bash run_zg.sh); '
                'mpirun -np N python thermal/zg.py --spectral --plot'
            )
    print(
        '  (every driver reads %s/system.py; outputs go to <workflow>/output/, the vertex cache to cache/)'
        % os.path.basename(cfg['out'])
    )
    return 0
