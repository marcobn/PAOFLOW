"""The files a ``paoflow-gen`` workflow writes, before anything touches the disk.

:func:`workflow_outputs` lists, for a configuration, every file of a workflow with its content (or
the template it copies).  The CLI writes the list (:func:`PAOFLOW.gen.paoflow_driver.main`,
:func:`PAOFLOW.gen.elphon_system.write_system_folder`); the browser GUI previews it, shows the
differences against existing files and writes it after confirmation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratedFile:
    """One generated file.

    Parameters
    ----------
    path : str
        Path relative to the output folder (``main.py``, ``qe/phonon/ph.in``, ...).
    text : str, optional
        The content (generated files).
    template : str, optional
        Absolute path of a file copied verbatim (the electron-phonon drivers).
    executable : bool, optional
        Set the executable bits when written (shell scripts).
    """

    path: str
    text: str | None = None
    template: str | None = None
    executable: bool = False

    def content(self) -> str:
        """The text of the file (read from the template for copies).

        Returns
        -------
        str
        """
        if self.text is not None:
            return self.text
        with open(self.template, 'r', encoding='utf-8') as handle:
            return handle.read()

    def write(self, folder: str) -> str:
        """Write the file under ``folder`` (directories created).

        Parameters
        ----------
        folder : str
            The output folder.

        Returns
        -------
        str
            The absolute path written.
        """
        import shutil

        path = os.path.abspath(os.path.join(folder, self.path))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if self.text is not None:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(self.text)
        else:
            shutil.copyfile(self.template, path)
        if self.executable:
            os.chmod(path, 0o755)
        return path


#: Default name of the driver of each single-script workflow.
DRIVER_NAMES = {
    'run': 'main.py',
    'acbn0': 'main.acbn0.py',
    'phonon': 'main.phonon.py',
    'sscha': 'main.sscha.py',
    'pyskeaf': 'main.pyskeaf.py',
}

#: Whether a workflow writes its plot scripts by default (``cfg['plot']`` decides otherwise): the
#: defaults of ``paoflow-gen``, which the *Write the plot scripts* switch of the Studio starts from.
#: ``run``, ``phonon`` and ``pyskeaf`` ask ("Also generate a plotting script?", default yes),
#: ``acbn0`` and ``elphon`` always wrote them, ``sscha`` only with ``--plot``.
PLOT_DEFAULTS = {
    'run': True,
    'acbn0': True,
    'phonon': True,
    'elphon': True,
    'sscha': False,
    'pyskeaf': True,
}

#: Plot scripts that save their figures (no display): the Runs tab can run them.
SAVING_PLOT_SCRIPTS = ('plot.sscha.py', 'plot.pyskeaf.py')


def workflow_outputs(
    name: str,
    cfg: dict,
    plot: bool = False,
    out: str | None = None,
    plot_out: str | None = None,
) -> list[GeneratedFile]:
    """The files of a workflow for a configuration, the driver first.

    Parameters
    ----------
    name : str
        Workflow name (``run``, ``acbn0``, ``phonon``, ``sscha``, ``elphon``).
    cfg : dict
        The configuration of the workflow (:func:`PAOFLOW.gen.questions.collect`).
    plot : bool, optional
        Force the plotting script (``--plot``); otherwise ``cfg['plot']`` decides.
    out : str, optional
        Driver path (``-o/--out``) for the single-script workflows.
    plot_out : str, optional
        Plotting-script path (``--plot-out``) of the ``run`` and ``acbn0`` workflows.

    Returns
    -------
    list of GeneratedFile
        For ``elphon`` the files of the system folder (paths relative to it).
    """
    from . import paoflow_driver as pd

    if name == 'elphon':
        from .elphon_system import system_folder_files

        return system_folder_files(cfg)
    driver = GeneratedFile(out or DRIVER_NAMES[name], text=_driver_text(name, cfg))
    files = [driver]
    wants_plot = plot or bool(cfg.get('plot', PLOT_DEFAULTS.get(name, False)))
    if name == 'run' and wants_plot:
        files.append(GeneratedFile(plot_out or 'plot.py', text=pd.build_plot_script(cfg)))
    elif name == 'acbn0' and wants_plot:
        files.append(
            GeneratedFile(plot_out or 'plot.acbn0.py', text=pd.build_acbn0_plot_script(cfg))
        )
    elif name in ('sscha', 'pyskeaf') and wants_plot:
        from .results_plot import build_results_plot_script

        title, kinds, outputdir = {
            'sscha': ('anharmonic phonons (SSCHA)', ('sscha',), 'output'),
            'pyskeaf': (
                'quantum oscillations (pyskeaf)',
                ('dhva',),
                cfg.get('outputdir', 'output'),
            ),
        }[name]
        files.append(
            GeneratedFile(
                plot_out or 'plot.%s.py' % name,
                text=build_results_plot_script(title, DRIVER_NAMES[name], kinds, outputdir),
            )
        )
    elif name == 'phonon':
        if wants_plot:
            files.append(
                GeneratedFile(plot_out or 'plot.phonon.py', text=pd.build_phonon_plot_script(cfg))
            )
        if cfg.get('raman'):
            files.append(GeneratedFile('main.raman.py', text=pd.build_raman_script(cfg)))
            if wants_plot:
                files.append(GeneratedFile('plot.raman.py', text=pd.build_raman_plot_script(cfg)))
    return files


def _driver_text(name: str, cfg: dict) -> str:
    from . import paoflow_driver as pd

    if name == 'run':
        return pd.build_run_script(cfg)
    if name == 'acbn0':
        return pd.build_acbn0_script(cfg)
    if name == 'phonon':
        return pd.build_phonon_script(cfg)
    if name == 'sscha':
        from .sscha_driver import build_sscha_script

        return build_sscha_script(cfg)
    if name == 'pyskeaf':
        from .pyskeaf_driver import build_pyskeaf_script

        return build_pyskeaf_script(cfg)
    raise ValueError('unknown workflow %r' % name)


# --------------------------------------------------------------------------------- run plan
@dataclass(frozen=True)
class RunStep:
    """One command of the run plan of a generated workflow.

    Parameters
    ----------
    label : str
        What the step does.
    argv : tuple of str
        Program and arguments (``python`` is replaced by the running interpreter when the GUI
        launches the step).
    cwd : str, optional
        Folder of the step, relative to the output folder.
    mpi : bool, optional
        The step may run under ``mpirun -np N``.
    ranks_env : str, optional
        Environment variable that carries the number of MPI ranks instead (``NP`` of
        ``run_qe.sh``).
    interactive : bool, optional
        The step needs a terminal or a display (the plotting scripts): it is shown, not run by the
        GUI (the Results tab replaces it).
    manual : bool, optional
        Not a single command (an instruction for the user).
    note : str, optional
        Remark shown with the step.
    stdout : str, optional
        File (in ``cwd``) that receives the standard output: the result of programs that print
        it, such as ``pw.x > scf.out``.
    action : str, optional
        A step done in the GUI rather than by a command: ``'setup:<workflow>'`` opens that
        workflow of the Setup tab on the folder.
    qe_launch : bool, optional
        The driver or script runs Quantum ESPRESSO itself (ACBN0, the phonon, Raman and local
        SSCHA phases, ``run_qe.sh``, ``run_zg.sh``): the Runs tab passes the ranks, the QE build
        and the ``pw.x`` options in the variables of :mod:`PAOFLOW.utils.qe_launch`, which it
        reads before its own defaults.
    """

    label: str
    argv: tuple[str, ...]
    cwd: str = '.'
    mpi: bool = False
    ranks_env: str | None = None
    interactive: bool = False
    manual: bool = False
    note: str = ''
    stdout: str | None = None
    action: str | None = None
    qe_launch: bool = False

    def command(self, ranks: int | None = None) -> str:
        """Shell text of the step (with ``mpirun -np N`` when ``ranks`` is given).

        Parameters
        ----------
        ranks : int, optional
            MPI ranks.

        Returns
        -------
        str
        """
        import shlex

        text = ' '.join(shlex.quote(token) for token in self.argv)
        if self.mpi and ranks:
            text = 'mpirun -np %d %s' % (ranks, text)
        if self.ranks_env and ranks:
            text = '%s=%d %s' % (self.ranks_env, ranks, text)
        if self.stdout:
            text = '%s > %s' % (text, shlex.quote(self.stdout))
        if self.cwd not in ('', '.'):
            text = '(cd %s && %s)' % (shlex.quote(self.cwd), text)
        return text

    def to_json(self) -> dict:
        """JSON-ready description (the browser's run plan).

        Returns
        -------
        dict
        """
        return {
            'label': self.label,
            'argv': list(self.argv),
            'command': self.command() if self.argv else '',
            'cwd': self.cwd,
            'mpi': self.mpi,
            'ranks_env': self.ranks_env,
            'interactive': self.interactive,
            'manual': self.manual,
            'note': self.note,
            'stdout': self.stdout,
            'action': self.action,
            'qe_launch': self.qe_launch,
        }


def qe_input_plan(filename: str) -> list[RunStep]:
    """What follows a QE input written from a database entry (``paoflow-gen-qe``).

    Parameters
    ----------
    filename : str
        The ``pw.x`` input (``<compound>.scf.in``).

    Returns
    -------
    list of RunStep
        The ``pw.x`` self-consistent run (output ``<compound>.scf.out``), then the PAOFLOW drivers
        in the Setup tab.
    """
    stem = filename[:-3] if filename.endswith('.in') else filename
    return [
        RunStep(
            'Self-consistent run (pw.x)',
            ('pw.x', '-in', filename),
            mpi=True,
            stdout=stem + '.out',
            note='pw.x must be on the PATH of paoflow-studio; the run writes <prefix>.save here',
        ),
        RunStep(
            'PAOFLOW drivers for the new save',
            (),
            manual=True,
            action='setup:run',
            note='Setup tab, Property run (or ACBN0, phonons, SSCHA) on this folder, once the run '
            'has finished',
        ),
    ]


def run_plan(name: str, cfg: dict, files: list[GeneratedFile] | None = None) -> list[RunStep]:
    """The commands that run a generated workflow, in order.

    Parameters
    ----------
    name : str
        Workflow name.
    cfg : dict
        Its configuration.
    files : list of GeneratedFile, optional
        Its files (:func:`workflow_outputs`); the driver names are read from them.

    Returns
    -------
    list of RunStep
    """
    files = files if files is not None else workflow_outputs(name, cfg)
    names = [f.path for f in files]
    if name == 'elphon':
        return _elphon_plan(cfg)
    driver = names[0]
    steps: list[RunStep] = []
    if name == 'run':
        steps.append(RunStep('PAOFLOW properties', ('python', driver), mpi=True))
    elif name == 'acbn0':
        steps.append(
            RunStep(
                'ACBN0 self-consistency (runs pw.x and projwfc.x itself)',
                ('python', driver),
                qe_launch=True,
                note='pw.x and the Hartree step run on the ranks, Quantum ESPRESSO build and pw.x '
                'options of this step (in a terminal: the PAOFLOW_* variables of the command, or the '
                'launch commands at the top of the script).',
            )
        )
    elif name == 'phonon':
        # The run phases start pw.x (and ph.x) themselves: ranks, build and options of the step.
        for phase, label in (
            ('generate', 'displaced supercells'),
            ('run', 'pw.x on the supercells'),
            ('analyse', 'phonons, Born charges, QHA'),
        ):
            steps.append(
                RunStep('Phonons: %s' % label, ('python', driver, phase), qe_launch=phase == 'run')
            )
        if 'main.raman.py' in names:
            for phase in ('generate', 'run', 'analyse'):
                steps.append(
                    RunStep(
                        'Raman: %s' % phase,
                        ('python', 'main.raman.py', phase),
                        qe_launch=phase == 'run',
                    )
                )
    elif name == 'sscha':
        # The local launcher runs ph.x / pw.x here (ranks, build and options of the step); Slurm
        # jobs and manual runs take the settings of the script.
        local = cfg.get('launcher', 'local') == 'local'
        steps.append(
            RunStep(
                'Harmonic ph.x matrices (only if missing)',
                ('python', driver, 'harmonic'),
                qe_launch=local,
            )
        )
        steps.append(
            RunStep(
                'SSCHA populations and minimisation', ('python', driver, 'run'), qe_launch=local
            )
        )
        steps.append(RunStep('Status of every temperature', ('python', driver, 'status')))
    elif name == 'pyskeaf':
        steps.append(
            RunStep(
                'Quantum oscillations (pyskeaf)',
                ('python', driver),
                mpi=True,
                note='the field directions spread over the ranks; the Results tab plots the '
                'frequencies against the angle',
            )
        )
    for plot in names[1:]:
        if plot.startswith('plot'):
            saving = plot in SAVING_PLOT_SCRIPTS
            steps.append(
                RunStep(
                    'Figures (%s)' % plot,
                    ('python', plot),
                    interactive=not saving,
                    note='PDF files in output/figures; or the Results tab'
                    if saving
                    else 'or the Results tab',
                )
            )
    return steps


def _elphon_plan(cfg: dict) -> list[RunStep]:
    steps = []
    plots = cfg.get('plot', True)  # the plot_*.py of the workflows (see system_folder_files)
    if not cfg.get('pw_template_text'):
        steps.append(
            RunStep(
                'Write qe/phonon/scf.in and qe/nscf/nscf.in',
                (),
                manual=True,
                note='K_POINTS card of qe/nscf/nscf.kpoints, nbnd = %d' % cfg.get('nbnd', 0),
            )
        )
    steps.append(
        RunStep(
            'QE: scf, ph.x (dvscf), nscf',
            ('bash', 'run_qe.sh'),
            cwd='qe',
            ranks_env='NP',
            qe_launch=True,
            note='copy the pseudopotentials into qe/ first',
        )
    )
    steps.append(
        RunStep(
            'Pseudo-atom basis',
            (
                'paoflow-genbasis-ps',
                '--pseudo-dir',
                'qe',
                '--out',
                cfg.get('basisdir', 'BASIS_PS'),
                '--preset',
                'standard',
            ),
        )
    )
    metal = cfg.get('metal')
    for workflow in cfg.get('workflows', ()):
        if workflow == 'phonons':
            steps.append(
                RunStep('Phonon dispersion', ('python', 'phonons/phonons.py', '--no-plot'))
            )
        elif workflow == 'coupling':
            steps.append(
                RunStep(
                    'Quadrupoles: ph.x inputs',
                    ('python', 'coupling/quadrupoles.py', '--write-inputs'),
                )
            )
            steps.append(
                RunStep('Quadrupoles: ph.x runs', ('bash', 'run.sh'), cwd='qe/phonon/quadrupoles')
            )
            steps.append(
                RunStep(
                    'Quadrupoles: fit',
                    ('python', 'coupling/quadrupoles.py', '--extract'),
                    note='then set quadrupoles in system.py',
                )
            )
        elif workflow == 'superconductivity':
            steps.append(
                RunStep(
                    'Eliashberg function', ('python', 'superconductivity/eliashberg.py'), mpi=True
                )
            )
            steps.append(
                RunStep('Migdal-Eliashberg (isotropic)', ('python', 'superconductivity/me.py'))
            )
            if cfg.get('superconductivity', {}).get('fs_coupling'):
                steps.append(
                    RunStep(
                        'Migdal-Eliashberg (anisotropic)',
                        ('python', 'superconductivity/me_aniso.py'),
                    )
                )
            if plots:
                steps.append(
                    RunStep(
                        'Figures (gap)',
                        ('python', 'superconductivity/plot_me.py'),
                        interactive=True,
                        note='or the Results tab',
                    )
                )
        elif workflow == 'optics' and not metal:
            if cfg.get('optics_nscf'):
                steps.append(
                    RunStep(
                        'Direct dielectric function (extended basis)',
                        ('python', 'optics/dielectric.py'),
                    )
                )
            steps.append(
                RunStep('Phonon-assisted absorption', ('python', 'optics/absorption.py'), mpi=True)
            )
            if plots:
                steps.append(
                    RunStep(
                        'Figures (absorption)',
                        ('python', 'optics/plot_absorption.py'),
                        interactive=True,
                        note='or the Results tab',
                    )
                )
        elif workflow == 'optics' and metal:
            steps.append(
                RunStep(
                    'Interband dielectric function',
                    ('python', 'optics/dielectric.py', '--interband-only'),
                )
            )
        elif workflow == 'transport' and metal:
            steps.append(
                RunStep(
                    'Resistivity',
                    ('python', 'transport/resistivity.py'),
                    note='after eliashberg.py with the transport Eliashberg function',
                )
            )
            if plots:
                steps.append(
                    RunStep(
                        'Figures (resistivity)',
                        ('python', 'transport/plot_resistivity.py'),
                        interactive=True,
                        note='or the Results tab',
                    )
                )
        elif workflow == 'transport':
            steps.append(RunStep('Carrier mobility', ('python', 'transport/mobility.py'), mpi=True))
            if plots:
                steps.append(
                    RunStep(
                        'Figures (mobility)',
                        ('python', 'transport/plot_mobility.py'),
                        interactive=True,
                        note='or the Results tab',
                    )
                )
        elif workflow == 'arpes':
            steps.append(RunStep('ARPES maps', ('python', 'arpes/arpes.py', '--no-plot'), mpi=True))
        elif workflow == 'thermal':
            steps.append(RunStep('ZG supercell inputs', ('python', 'thermal/zg.py', '--write')))
            steps.append(
                RunStep(
                    'ZG supercell scf runs',
                    ('bash', 'run_zg.sh'),
                    cwd='thermal/zg',
                    ranks_env='NP',
                    qe_launch=True,
                )
            )
            steps.append(
                RunStep(
                    'Unfolded spectral functions',
                    ('python', 'thermal/zg.py', '--spectral'),
                    mpi=True,
                )
            )
    return steps
