"""Questions of the ``sscha`` workflow: anharmonic phonons at temperature (``main.sscha.py``)."""

from __future__ import annotations

from .common import detected_savedir
from .model import Answers, Context, Note, Question, Workflow, gated, int_list

SYSTEM = 'System'
ENSEMBLE = 'Ensembles'
LAUNCH = 'Launcher'
SLURM = 'Slurm settings'
MINIMISER = 'Minimiser'


def _temperatures(text: str) -> list[float]:
    try:
        return [float(t) for t in text.replace(',', ' ').split()]
    except ValueError:
        raise ValueError('not a list of temperatures: %r' % text) from None


def _optional_grid(text: str) -> list[int] | None:
    return int_list(text) if text.strip() else None


def _default_prefix(answers: Answers, context: Context) -> str:
    from ..paoflow_driver import detect_prefix

    return detect_prefix(context.workdir, detected_savedir(context))


def _default_harmonic(answers: Answers, context: Context) -> str:
    from ..sscha_driver import _detect_harmonic

    return _detect_harmonic(context.workdir, answers['prefix'])


def _default_populations(answers: Answers, context: Context) -> str:
    from ..sscha_driver import DEFAULT_POPULATIONS

    return ', '.join(map(str, DEFAULT_POPULATIONS))


def _slurm(answers: Answers) -> bool:
    return answers.get('launcher') == 'slurm'


SSCHA_ITEMS = (
    Question(
        'savedir',
        'Save directory of the scf (<prefix>.save)',
        default=lambda a, c: detected_savedir(c) or 'pwscf.save',
        override='savedir',
        section=SYSTEM,
        widget='folder',
    ),
    Question('prefix', 'Prefix', default=_default_prefix, override='prefix', section=SYSTEM),
    Question(
        'pp_dir',
        'Pseudopotential directory (blank or "HERE" = script directory)',
        default='HERE',
        section=SYSTEM,
    ),
    Question(
        'temperatures',
        'Temperatures (K, comma-separated)',
        default='300',
        parse=_temperatures,
        unit='K',
        section=ENSEMBLE,
    ),
    Question(
        'supercell',
        'Supercell multiplicity (isotropic diagonal)',
        kind='int',
        default=2,
        section=ENSEMBLE,
    ),
    Question(
        'harmonic_dyn',
        'ph.x dynamical matrices (fildyn prefix <dir>/<prefix>.dyn; written by the harmonic '
        'phase if missing)',
        default=_default_harmonic,
        section=ENSEMBLE,
    ),
    Question(
        'harmonic_qgrid_axis',
        'q-grid of the harmonic ph.x run if it is generated (multiple of the supercell; larger '
        '= better harmonic dispersion)',
        kind='int',
        default=lambda a, c: a['supercell'],
        section=ENSEMBLE,
    ),
    Question(
        'kgrid',
        'k-grid of the supercell force runs (e.g. "6 6 6"; blank = scf grid / supercell)',
        default='',
        parse=_optional_grid,
        section=ENSEMBLE,
    ),
    Question(
        'electrons',
        'Electrons of the force runs (parent = occupations of the scf; ionic = Fermi-Dirac at '
        'k_B T of each temperature, hot electrons; fixed = Fermi-Dirac at one temperature)',
        kind='choice',
        choices=['parent', 'ionic', 'fixed'],
        default='parent',
        label='Electrons of the force runs',
        section=ENSEMBLE,
    ),
    Question(
        'electronic_temperature_k',
        '  Electronic temperature (K)',
        kind='float',
        default=300.0,
        when=lambda a, c: a.get('electrons') == 'fixed',
        unit='K',
        section=ENSEMBLE,
    ),
    Question(
        'population_sizes',
        'Configurations per population (comma-separated; the last repeats)',
        default=_default_populations,
        parse=int_list,
        section=ENSEMBLE,
    ),
    Question(
        'hessian_size',
        'Configurations of the free-energy Hessian population',
        kind='int',
        default=800,
        section=ENSEMBLE,
    ),
    Question(
        'launcher',
        'Launcher of the pw.x runs (local = here; slurm = job arrays chained by step jobs; '
        'manual = write inputs)',
        kind='choice',
        choices=['local', 'slurm', 'manual'],
        default='local',
        label='Launcher of the pw.x runs',
        section=LAUNCH,
    ),
    Question(
        'qe_pw',
        "pw.x command (may include flags, e.g. 'pw.x -nk 4')",
        default='pw.x',
        section=LAUNCH,
    ),
    *gated(
        (
            Note(
                '\nSlurm settings of one configuration (array task) and of the step job:',
                section=SLURM,
            ),
            Question('slurm_mpi_qe', 'MPI launcher inside the jobs', default='srun', section=SLURM),
            Question('slurm_partition', '  partition', default='normal', section=SLURM),
            Question(
                'slurm_nodes', '  nodes per configuration', kind='int', default=1, section=SLURM
            ),
            Question(
                'slurm_ntasks',
                '  MPI tasks per configuration',
                kind='int',
                default=32,
                section=SLURM,
            ),
            Question(
                'slurm_walltime', '  wall time per configuration', default='02:00:00', section=SLURM
            ),
            Question('slurm_account', '  account (blank = default)', default='', section=SLURM),
            Question(
                'slurm_step_tasks',
                '  cores of the step job (python minimisation)',
                kind='int',
                default=4,
                section=SLURM,
            ),
            Question(
                'slurm_step_time', '  wall time of the step job', default='02:00:00', section=SLURM
            ),
            Question(
                'slurm_concurrent',
                '  array tasks running at the same time',
                kind='int',
                default=40,
                section=SLURM,
            ),
            Question(
                'slurm_env_setup',
                '  shell line that loads QE and PAOFLOW in a job (blank = none)',
                default='',
                section=SLURM,
            ),
            Question('slurm_python', '  python of the step jobs', default='python', section=SLURM),
        ),
        lambda a, c: _slurm(a),
    ),
    Question(
        'mpi_qe',
        'MPI command for pw.x / ph.x',
        default='mpirun -np 4',
        when=lambda a, c: not _slurm(a),
        section=LAUNCH,
    ),
    *(
        Question(
            key,
            prompt,
            kind='float',
            default=default,
            section=MINIMISER,
            level='advanced',
            cli=False,
        )
        for key, prompt, default in (
            ('min_step_dyn', 'Minimisation step of the dynamical matrices', 0.05),
            ('kong_liu_ratio', 'Kong-Liu effective-sample-size ratio that ends a population', 0.5),
            ('meaningful_factor', 'Gradient threshold relative to its stochastic error', 1.0e-3),
            ('conv_thr', 'pw.x conv_thr of the force runs (Ry)', 1.0e-10),
            ('mixing_beta', 'pw.x mixing_beta of the force runs', 0.3),
        )
    ),
)

#: The SSCHA minimiser settings the CLI leaves at their defaults (GUI-only).
MINIMISER_KEYS = ('min_step_dyn', 'kong_liu_ratio', 'meaningful_factor', 'conv_thr', 'mixing_beta')


def assemble_sscha(answers: Answers, context: Context) -> dict:
    """Configuration of ``main.sscha.py`` (:func:`PAOFLOW.gen.sscha_driver.build_sscha_script`).

    Parameters
    ----------
    answers : dict
        Answers of :data:`SSCHA_ITEMS`.
    context : Context
        The session context.

    Returns
    -------
    dict
    """
    cfg = {'outputdir': 'output', 'workdir': context.workdir}
    for key in ('savedir', 'prefix', 'pp_dir', 'temperatures', 'supercell', 'harmonic_dyn'):
        cfg[key] = answers[key]
    cfg['harmonic_qgrid'] = [answers['harmonic_qgrid_axis']] * 3
    cfg['kgrid'] = answers['kgrid']
    electrons = answers['electrons']
    if electrons == 'ionic':
        cfg['electronic_temperature'] = 'ionic'
    elif electrons == 'fixed':
        cfg['electronic_temperature'] = answers['electronic_temperature_k']
    else:
        cfg['electronic_temperature'] = None
    cfg['population_sizes'] = answers['population_sizes']
    cfg['hessian_size'] = answers['hessian_size']
    cfg['launcher'] = answers['launcher']
    cfg['qe_pw'] = answers['qe_pw']
    cfg['qe_ph'] = cfg['qe_pw'].replace('pw.x', 'ph.x') if 'pw.x' in cfg['qe_pw'] else 'ph.x'
    if _slurm(answers):
        cfg['mpi_qe'] = answers['slurm_mpi_qe']
        partition, account = answers['slurm_partition'], answers['slurm_account']
        directives = [
            '#SBATCH -p %s' % partition,
            '#SBATCH -N %d' % answers['slurm_nodes'],
            '#SBATCH -n %d' % answers['slurm_ntasks'],
            '#SBATCH -t %s' % answers['slurm_walltime'],
        ]
        if account:
            directives.append('#SBATCH -A %s' % account)
        cfg['slurm_directives'] = directives
        step = [
            '#SBATCH -p %s' % partition,
            '#SBATCH -N 1',
            '#SBATCH -n %d' % answers['slurm_step_tasks'],
            '#SBATCH -t %s' % answers['slurm_step_time'],
        ]
        if account:
            step.append('#SBATCH -A %s' % account)
        cfg['slurm_step_directives'] = step
        cfg['slurm_concurrent'] = answers['slurm_concurrent']
        setup = answers['slurm_env_setup']
        cfg['env_setup'] = [setup] if setup else []
        cfg['python'] = answers['slurm_python']
    else:
        cfg['mpi_qe'] = answers['mpi_qe']
    for key in MINIMISER_KEYS:  # GUI-only, when changed
        if key in answers:
            cfg[key] = answers[key]
    return cfg


SSCHA = Workflow(
    name='sscha',
    title='Anharmonic phonons (SSCHA)',
    description='Stochastic self-consistent harmonic approximation at a list of temperatures '
    '(main.sscha.py, resumable phases).',
    items=SSCHA_ITEMS,
    assemble=assemble_sscha,
    overrides=('savedir', 'prefix'),
)
