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
}


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
    wants_plot = plot or bool(cfg.get('plot'))
    if name == 'run' and wants_plot:
        files.append(GeneratedFile(plot_out or 'plot.py', text=pd.build_plot_script(cfg)))
    elif name == 'acbn0':
        files.append(
            GeneratedFile(plot_out or 'plot.acbn0.py', text=pd.build_acbn0_plot_script(cfg))
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
    """

    label: str
    argv: tuple[str, ...]
    cwd: str = '.'
    mpi: bool = False
    ranks_env: str | None = None
    interactive: bool = False
    manual: bool = False
    note: str = ''

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
        if self.cwd not in ('', '.'):
            text = '(cd %s && %s)' % (shlex.quote(self.cwd), text)
        return text


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
                note='MPI commands of pw.x and python are set inside the script.',
            )
        )
    elif name == 'phonon':
        for phase, label in (
            ('generate', 'displaced supercells'),
            ('run', 'pw.x on the supercells'),
            ('analyse', 'phonons, Born charges, QHA'),
        ):
            steps.append(RunStep('Phonons: %s' % label, ('python', driver, phase)))
        if 'main.raman.py' in names:
            for phase in ('generate', 'run', 'analyse'):
                steps.append(RunStep('Raman: %s' % phase, ('python', 'main.raman.py', phase)))
    elif name == 'sscha':
        steps.append(
            RunStep(
                'Harmonic ph.x matrices (only if missing)',
                ('python', driver, 'harmonic'),
            )
        )
        steps.append(RunStep('SSCHA populations and minimisation', ('python', driver, 'run')))
        steps.append(RunStep('Status of every temperature', ('python', driver, 'status')))
    for plot in names[1:]:
        if plot.startswith('plot'):
            steps.append(
                RunStep(
                    'Figures (%s)' % plot,
                    ('python', plot),
                    interactive=True,
                    note='or the Results tab',
                )
            )
    return steps


def _elphon_plan(cfg: dict) -> list[RunStep]:
    steps = []
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
                    'ZG supercell scf runs', ('bash', 'run_zg.sh'), cwd='thermal/zg', ranks_env='NP'
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
