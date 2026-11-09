"""Questions of the ``phonon`` workflow: finite-displacement phonons, Born charges, QHA and Raman."""

from __future__ import annotations

from .common import COMMON_ITEMS, COMMON_OVERRIDES, assemble_common
from .model import Answers, Context, Note, Question, Workflow, gated

PHONONS = 'Phonons'
BORN = 'Born charges and dielectric response'
QHA = 'Quasi-harmonic approximation'
LAUNCH = 'pw.x launch'
RAMAN = 'Raman'


def _laser_list(text: str) -> list[float]:
    from ..paoflow_driver import parse_laser_list

    return parse_laser_list(text)


def _vibdielectric(answers: Answers) -> bool:
    return bool(answers.get('born')) and bool(answers.get('vibdielectric'))


def _qha(answers: Answers) -> bool:
    return bool(answers.get('qha'))


def _raman(answers: Answers) -> bool:
    return answers.get('raman_method', 'none') != 'none'


def _resonance(answers: Answers) -> bool:
    return answers.get('raman_method') in ('resonance', 'all')


PHONON_ITEMS = (
    Question(
        'supercell',
        'Supercell multiplicity (isotropic diagonal)',
        kind='int',
        default=2,
        section=PHONONS,
    ),
    Question(
        'displacement',
        'Displacement amplitude (Bohr)',
        kind='float',
        default=0.06,
        unit='Bohr',
        section=PHONONS,
    ),
    Question(
        'mesh',
        'q-mesh density per axis (DOS / thermal)',
        kind='int',
        default=12,
        section=PHONONS,
    ),
    Question(
        'units',
        'Frequency units',
        kind='choice',
        choices=['cm-1', 'THz'],
        default='cm-1',
        section=PHONONS,
    ),
    Question(
        'do_thermal',
        'Compute thermal properties (F, S, Cv)?',
        kind='bool',
        default=True,
        section=PHONONS,
    ),
    Question(
        'pp_dir',
        'Pseudopotential directory (blank or "HERE" = script directory)',
        default='HERE',
        section=PHONONS,
    ),
    Question(
        'hubbard_file',
        'pw.x input with a HUBBARD card to inject (on-site U only; blank = none)',
        default='',
        section=PHONONS,
        level='advanced',
    ),
    Question(
        'born',
        'Compute Born charges + dielectric tensor (NAC / LO-TO splitting)?',
        kind='bool',
        default=True,
        section=BORN,
    ),
    Question(
        'born_method',
        'Born/epsilon method (dfpt = ph.x, field = lelfield pw.x)',
        kind='choice',
        choices=['dfpt', 'field'],
        default='dfpt',
        when=lambda a, c: bool(a.get('born')),
        section=BORN,
    ),
    Question(
        'vibdielectric',
        'Also compute the vibrational (ionic) dielectric function eps(w) (reststrahlen band)?',
        kind='bool',
        default=True,
        when=lambda a, c: bool(a.get('born')),
        section=BORN,
    ),
    Question(
        'vibdielectric_gamma',
        '  Phonon linewidth gamma for eps(w) (in the chosen units)',
        kind='float',
        default=4.0,
        when=lambda a, c: _vibdielectric(a),
        section=BORN,
    ),
    Question(
        'vibdielectric_emissivity',
        '  Also compute the reststrahlen (phonon) emissivity (1 - R)?',
        kind='bool',
        default=True,
        when=lambda a, c: _vibdielectric(a),
        section=BORN,
    ),
    Question(
        'vibdielectric_emis_temp',
        '    Temperature for the total hemispherical emissivity (K)',
        kind='float',
        default=300.0,
        when=lambda a, c: _vibdielectric(a) and bool(a.get('vibdielectric_emissivity')),
        unit='K',
        section=BORN,
    ),
    Question(
        'qha',
        'Compute quasi-harmonic properties (thermal expansion, V(T), B(T), Gruneisen)?',
        kind='bool',
        default=False,
        section=QHA,
    ),
    *gated(
        (
            Question(
                'qha_nvolumes',
                '  Number of sampled volumes (5 = full EOS fit; 3 = parabolic fit)',
                kind='choice',
                choices=['5', '3'],
                default='5',
                section=QHA,
            ),
            Question(
                'qha_strain',
                '  Maximum linear strain of the volume scan',
                kind='float',
                default=0.02,
                section=QHA,
            ),
            Question(
                'qha_eos',
                '  Equation of state',
                kind='choice',
                choices=['vinet', 'birch_murnaghan', 'murnaghan'],
                default='vinet',
                when=lambda a, c: int(a.get('qha_nvolumes', '5')) >= 4,
                section=QHA,
            ),
            Question(
                'qha_tmin',
                '  Minimum temperature (K)',
                kind='float',
                default=0.0,
                unit='K',
                section=QHA,
            ),
            Question(
                'qha_tmax',
                '  Maximum temperature (K)',
                kind='float',
                default=1000.0,
                unit='K',
                section=QHA,
            ),
            Question(
                'qha_tstep',
                '  Temperature step (K)',
                kind='float',
                default=10.0,
                unit='K',
                section=QHA,
            ),
            Question(
                'qha_pressure',
                '  External pressure (GPa)',
                kind='float',
                default=0.0,
                unit='GPa',
                section=QHA,
            ),
        ),
        lambda a, c: _qha(a),
    ),
    # Asked by the CLI only: paoflow-studio sets the ranks, the QE build and the pw.x options of
    # the run phase in its Runs tab, and the drivers read them at run time (PAOFLOW.utils.qe_launch).
    Note('\npw.x launch settings for the displaced-supercell SCF runs:', section=LAUNCH, gui=False),
    Question('mpi_qe', 'MPI command for pw.x', default='mpirun -np 4', section=LAUNCH, gui=False),
    Question(
        'qe_pw',
        "pw.x executable/command (may include flags, e.g. 'pw.x', 'pw.x -npool 4', "
        "'/opt/qe/bin/pw.x')",
        default='pw.x',
        section=LAUNCH,
        gui=False,
    ),
    Question(
        'plot',
        'Also generate a plotting script (plot.phonon.py)?',
        kind='bool',
        default=True,
        section='Output',
    ),
    Question(
        'raman_method',
        'Also generate a Raman workflow (main.raman.py)?  Pick the flavour '
        '(none = skip; all = static + resonance from one run)',
        kind='choice',
        choices=['none', 'static', 'resonance', 'all'],
        default='none',
        label='Raman workflow (main.raman.py)',
        section=RAMAN,
    ),
    *gated(
        (
            Note('\nRaman finite-difference settings:', section=RAMAN),
            Question(
                'raman_delta',
                '  Mass-weighted displacement step (Bohr*sqrt(amu))',
                kind='float',
                default=0.05,
                section=RAMAN,
            ),
            Question(
                'raman_nbnd',
                '  Bands for the displaced SCF (0 = QE default)',
                kind='int',
                default=0,
                section=RAMAN,
            ),
            Question(
                'raman_npool',
                '  npool for the per-cell dielectric run',
                kind='int',
                default=4,
                section=RAMAN,
            ),
            Question(
                'raman_smearing',
                '  Smearing for the dielectric run',
                default='gauss',
                section=RAMAN,
            ),
            Question(
                'raman_nfft',
                '  Double-grid FFT size per direction (0 = skip)',
                kind='int',
                default=24,
                section=RAMAN,
            ),
            Question(
                'raman_e_static',
                '  Static-epsilon upper energy (eV)',
                kind='float',
                default=0.05,
                unit='eV',
                section=RAMAN,
            ),
            Question(
                'raman_temperature',
                '  Temperature for the Bose factor (K)',
                kind='float',
                default=300.0,
                unit='K',
                section=RAMAN,
            ),
            Question(
                'raman_gamma',
                '  Lorentzian FWHM (in the chosen units)',
                kind='float',
                default=4.0,
                section=RAMAN,
            ),
            Question(
                'raman_lasers',
                '  Laser wavelength(s) in nm (comma-separated, or a Python list '
                'expression like "[n for n in range(450, 650, 5)]")',
                default='532',
                parse=_laser_list,
                when=lambda a, c: _resonance(a),
                unit='nm',
                section=RAMAN,
            ),
            Question(
                'raman_lifetime',
                '  Lifetime broadening of the dielectric tensor (eV)',
                kind='float',
                default=0.1,
                when=lambda a, c: _resonance(a),
                unit='eV',
                section=RAMAN,
            ),
            Question(
                'raman_laser',
                '  Laser wavelength (nm; 0 = omit the (wL-wv)^4 factor)',
                kind='float',
                default=0.0,
                when=lambda a, c: not _resonance(a),
                unit='nm',
                section=RAMAN,
            ),
        ),
        lambda a, c: _raman(a),
    ),
)

#: Values of the optional blocks when they are not asked.
QHA_DEFAULTS = {
    'qha_nvolumes': 5,
    'qha_strain': 0.02,
    'qha_eos': 'vinet',
    'qha_tmin': 0.0,
    'qha_tmax': 1000.0,
    'qha_tstep': 10.0,
    'qha_pressure': 0.0,
}


def assemble_phonon_from_common(common: dict, answers: Answers) -> dict:
    """Configuration of the phonon driver on top of the common configuration.

    Parameters
    ----------
    common : dict
        The common configuration.
    answers : dict
        Answers of :data:`PHONON_ITEMS`.

    Returns
    -------
    dict
    """
    cfg = dict(common)
    cfg['supercell'] = answers['supercell']
    cfg['displacement'] = answers['displacement']
    cfg['mesh'] = [answers['mesh']] * 3
    for key in ('units', 'do_thermal', 'pp_dir', 'hubbard_file', 'born'):
        cfg[key] = answers[key]
    cfg['born_method'] = answers['born_method'] if cfg['born'] else 'dfpt'
    cfg['vibdielectric'] = _vibdielectric(answers)
    if cfg['vibdielectric']:
        cfg['vibdielectric_gamma'] = answers['vibdielectric_gamma']
        cfg['vibdielectric_emissivity'] = answers['vibdielectric_emissivity']
        cfg['vibdielectric_emis_temp'] = (
            [answers['vibdielectric_emis_temp']] if cfg['vibdielectric_emissivity'] else [300.0]
        )
    else:
        cfg['vibdielectric_gamma'] = 4.0
        cfg['vibdielectric_emissivity'] = False
        cfg['vibdielectric_emis_temp'] = [300.0]
    cfg['qha'] = answers['qha']
    if cfg['qha']:
        cfg['qha_nvolumes'] = int(answers['qha_nvolumes'])
        cfg['qha_strain'] = answers['qha_strain']
        cfg['qha_eos'] = answers['qha_eos'] if cfg['qha_nvolumes'] >= 4 else 'vinet'
        for key in ('qha_tmin', 'qha_tmax', 'qha_tstep', 'qha_pressure'):
            cfg[key] = answers[key]
    else:
        cfg.update(QHA_DEFAULTS)
    cfg['mpi_qe'] = answers['mpi_qe']
    cfg['qe_pw'] = answers['qe_pw']
    cfg['plot'] = answers['plot']
    cfg['raman_method'] = answers['raman_method']
    cfg['raman'] = cfg['raman_method'] != 'none'
    if cfg['raman']:
        for key in (
            'raman_delta',
            'raman_nbnd',
            'raman_npool',
            'raman_smearing',
            'raman_nfft',
            'raman_e_static',
            'raman_temperature',
            'raman_gamma',
        ):
            cfg[key] = answers[key]
        if _resonance(answers):
            lasers = answers['raman_lasers']
            cfg['raman_laser_nm'] = lasers if len(lasers) > 1 else lasers[0]
            cfg['raman_lifetime'] = answers['raman_lifetime']
        else:
            laser = answers['raman_laser']
            cfg['raman_laser_nm'] = laser if laser > 0 else None
            cfg['raman_lifetime'] = 0.1
        cfg['raman_pthr'] = 0.95
        cfg['raman_configuration'] = 'extended'
    return cfg


def assemble_phonon(answers: Answers, context: Context) -> dict:
    """Configuration of the phonon driver (:func:`PAOFLOW.gen.paoflow_driver.build_phonon_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`COMMON_ITEMS` and :data:`PHONON_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
    """
    return assemble_phonon_from_common(assemble_common(answers, context), answers)


PHONON = Workflow(
    name='phonon',
    title='Harmonic phonons',
    description='Finite-displacement phonons with phonopy, Born charges, vibrational dielectric '
    'function, QHA and Raman (main.phonon.py, plot.phonon.py, main.raman.py).',
    items=COMMON_ITEMS + PHONON_ITEMS,
    assemble=assemble_phonon,
    overrides=COMMON_OVERRIDES,
)
