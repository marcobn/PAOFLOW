"""The file lists of the generator workflows and the CLI that writes them."""

from __future__ import annotations

import os

import pytest

from PAOFLOW.gen import elphon_system as es
from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen.outputs import GeneratedFile, workflow_outputs


def _feed(monkeypatch, answers):
    queue = list(answers)
    monkeypatch.setattr(pd, '_input', lambda prompt: queue.pop(0) if queue else '')


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    save = tmp_path / 'si.save'
    save.mkdir()
    (save / 'Si.upf').write_text('x')
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_run_cli_writes_the_listed_files(workdir, monkeypatch, capsys):
    # workflow, savedir, prefix, upfs, basis, ibrav, 2D, projector, properties, then defaults
    _feed(monkeypatch, ['run', '', '', '', '', '', '', '', '1 2'])
    assert pd.main(['-d', str(workdir)]) == 0
    out = capsys.readouterr().out
    assert 'Wrote %s' % os.path.join(str(workdir), 'main.py') in out
    assert (workdir / 'main.py').read_text().startswith('#!/usr/bin/env python3')
    assert 'def plot_band_structure' in (workdir / 'plot.py').read_text()


def test_cli_refuses_to_overwrite_the_driver(workdir, monkeypatch, capsys):
    (workdir / 'main.acbn0.py').write_text('keep')
    _feed(monkeypatch, ['acbn0'])
    assert pd.main(['-d', str(workdir)]) == 1
    assert 'Refusing to overwrite main.acbn0.py' in capsys.readouterr().err
    assert (workdir / 'main.acbn0.py').read_text() == 'keep'


def test_cli_keeps_existing_companions_without_force(workdir, monkeypatch, capsys):
    (workdir / 'plot.acbn0.py').write_text('keep')
    _feed(monkeypatch, ['acbn0', '', '', '', '', '', '', '', '', '', '', '', '', 'Si-3p 0.5', ''])
    assert pd.main(['-d', str(workdir)]) == 0
    assert 'Refusing to overwrite plot.acbn0.py' in capsys.readouterr().err
    assert (workdir / 'plot.acbn0.py').read_text() == 'keep'
    assert (workdir / 'main.acbn0.py').exists()


def test_phonon_outputs_follow_plot_and_raman():
    cfg = {'plot': False, 'raman': True}
    monkey = {
        'build_phonon_script': 'driver',
        'build_phonon_plot_script': 'plot',
        'build_raman_script': 'raman',
        'build_raman_plot_script': 'raman plot',
    }
    saved = {name: getattr(pd, name) for name in monkey}
    try:
        for name, text in monkey.items():
            setattr(pd, name, lambda cfg, text=text: text)
        assert [f.path for f in workflow_outputs('phonon', cfg)] == [
            'main.phonon.py',
            'main.raman.py',
        ]
        names = [f.path for f in workflow_outputs('phonon', cfg, plot=True, plot_out='p.py')]
        assert names == ['main.phonon.py', 'p.py', 'main.raman.py', 'plot.raman.py']
    finally:
        for name, function in saved.items():
            setattr(pd, name, function)


def test_generated_file_copies_templates_and_sets_executable(tmp_path):
    template = tmp_path / 'template.py'
    template.write_text('print(1)\n')
    copied = GeneratedFile('sub/driver.py', template=str(template))
    script = GeneratedFile('run.sh', text='#!/bin/bash\n', executable=True)
    assert copied.content() == 'print(1)\n'
    path = copied.write(str(tmp_path / 'out'))
    assert open(path).read() == 'print(1)\n'
    assert os.access(script.write(str(tmp_path / 'out')), os.X_OK)


def test_system_folder_files_match_what_is_written(tmp_path):
    from test_elphon_system import _cfg

    cfg = _cfg()
    files = es.system_folder_files(cfg)
    paths = [f.path for f in files]
    assert paths[0] == 'system.py' and 'arpes/arpes.py' in paths
    assert [f.path for f in files if f.executable] == ['qe/run_qe.sh']
    written = es.write_system_folder(cfg, str(tmp_path / 'sys'))
    assert written == [os.path.join(str(tmp_path / 'sys'), p) for p in paths]
    for generated in files:
        assert (tmp_path / 'sys' / generated.path).read_text() == generated.content()
    assert (tmp_path / 'sys' / 'cache').is_dir() and (tmp_path / 'sys' / 'figures').is_dir()


def test_qe_input_plan_and_step_json():
    from PAOFLOW.gen.outputs import RunStep, qe_input_plan

    run, setup = qe_input_plan('TaC.scf.in')
    assert run.command(4) == 'mpirun -np 4 pw.x -in TaC.scf.in > TaC.scf.out'
    assert setup.to_json()['action'] == 'setup:run' and setup.to_json()['command'] == ''
    step = RunStep('basis', ('paoflow-genbasis-ps', '--out', 'B'), cwd='qe')
    assert step.to_json()['command'] == '(cd qe && paoflow-genbasis-ps --out B)'


def test_drivers_abort_every_rank_on_an_error():
    """The drivers end with run_main (an error on one MPI rank stops the job, see
    PAOFLOW.utils.mpi_guard); the plot scripts, single-process, do not need it."""
    from PAOFLOW.gen.questions import WORKFLOWS
    from PAOFLOW.gen.questions.model import Context
    from PAOFLOW.gen.questions.resolve import resolve_form

    si = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
    )
    answers = {
        'run': {'properties': ['bands']},
        'acbn0': {'hubbard': [['Si-3p', '0.5']]},
        'phonon': {'raman_method': 'resonance'},
        'sscha': {},
    }
    drivers = set()
    for name, given in answers.items():
        form = resolve_form(WORKFLOWS[name], given, Context(workdir=os.path.abspath(si)))
        assert form.complete, (name, form.errors)
        for generated in workflow_outputs(name, form.config, plot=True):
            text = generated.content()
            compile(text, generated.path, 'exec')
            guarded = 'run_main(main)' in text
            assert guarded == (
                not os.path.basename(generated.path).startswith('plot')
            ), generated.path
            if guarded:
                drivers.add(generated.path)
                assert 'from PAOFLOW.utils.mpi_guard import run_main' in text
    assert drivers == {
        'main.py',
        'main.acbn0.py',
        'main.phonon.py',
        'main.raman.py',
        'main.sscha.py',
    }
    for rel in ('arpes/arpes.py', 'optics/absorption.py', 'transport/mobility.py', 'thermal/zg.py',
                'superconductivity/eliashberg.py', 'coupling/long_range_check.py'):  # fmt: skip
        with open(os.path.join(es.TEMPLATE_DIR, rel)) as handle:
            assert handle.read().rstrip().endswith('run_main(main)'), rel


def test_acbn0_launch_settings_come_from_the_runs_tab(monkeypatch):
    """The ACBN0 launch commands are CLI questions only: the GUI form hides them and the driver
    reads the PAOFLOW_* variables that the Runs tab sets, before the CLI answers."""
    from PAOFLOW.gen.outputs import run_plan
    from PAOFLOW.gen.questions import WORKFLOWS
    from PAOFLOW.gen.questions.model import Context
    from PAOFLOW.gen.questions.resolve import resolve_form

    si = os.path.join(
        os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
    )
    answers = {'hubbard': [['Si-3p', '0.5']], 'mpi_qe': 'srun -n 64'}  # a GUI cannot set it
    form = resolve_form(WORKFLOWS['acbn0'], answers, Context(workdir=os.path.abspath(si)))
    launch = ('mpi_qe', 'mpi_py', 'mpi_hartree', 'qe_path', 'py_path')
    items = {i['key']: i for i in form.items if i.get('type') != 'note' and 'key' in i}
    assert all(items[key]['visible'] is False for key in launch)
    assert not any(
        i['visible']
        for i in form.items
        if i.get('type') == 'note' and 'launch' in i['text'].lower()
    )
    assert form.config['mpi_qe'] == 'mpirun -np 4'  # the default, whatever the payload says
    driver = workflow_outputs('acbn0', form.config)[0].content()
    assert "MPI_QE = mpi_prefix('mpirun -np 4')" in driver
    assert "QE_PATH = qe_bin('')" in driver
    assert driver.count('qe_options=QE_OPTIONS') >= 1 and "qe_options=''" not in driver
    namespace = {}
    monkeypatch.setenv('PAOFLOW_MPI_QE', 'mpirun -np 8')
    monkeypatch.setenv('PAOFLOW_QE_OPTIONS', '-nk 4')
    head = driver[: driver.index('PROJECTION =')]
    exec(
        compile(
            head.replace('HERE = os.path.dirname(os.path.abspath(__file__))', 'HERE = "."'),
            'head',
            'exec',
        ),
        namespace,
    )  # noqa: S102
    assert namespace['MPI_QE'] == 'mpirun -np 8' and namespace['QE_OPTIONS'] == '-nk 4'
    assert namespace['MPI_HARTREE'] == 'mpirun -np 8'  # follows the ranks of pw.x
    assert namespace['MPI_PY'] == 'mpirun -np 1'  # not set: the script's value
    (step,) = run_plan('acbn0', form.config)[:1]
    assert step.qe_launch and step.to_json()['qe_launch'] and not step.mpi


def test_every_qe_run_takes_the_runs_tab_settings():
    """The run phases that start QE carry qe_launch, and their launch questions are CLI-only."""
    from PAOFLOW.gen.outputs import RunStep, run_plan
    from PAOFLOW.gen.questions import WORKFLOWS
    from PAOFLOW.gen.questions.model import Context
    from PAOFLOW.gen.questions.resolve import resolve_form

    si = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
        )
    )

    def resolved(name, answers):
        form = resolve_form(WORKFLOWS[name], answers, Context(workdir=si))
        assert form.complete, form.errors
        return form, {i['key']: i for i in form.items if 'key' in i}

    form, items = resolved('phonon', {'raman_method': 'static'})
    assert not items['mpi_qe']['visible'] and not items['qe_pw']['visible']
    launch = {s.label: s.qe_launch for s in run_plan('phonon', form.config)}
    assert launch['Phonons: pw.x on the supercells'] and launch['Raman: run']
    assert not launch['Phonons: displaced supercells'] and not launch['Raman: analyse']
    driver = workflow_outputs('phonon', form.config)[0].content()
    assert "QE_PW = qe_command('pw.x', 'pw.x')" in driver and "QE_PH = qe_command('ph.x'" in driver

    form, items = resolved('sscha', {})
    assert form.config['launcher'] == 'local'
    assert not items['mpi_qe']['visible'] and not items['qe_pw']['visible']
    assert all(s.qe_launch for s in run_plan('sscha', form.config) if 'status' not in s.argv)
    slurm, items = resolved('sscha', {'launcher': 'slurm', 'slurm_account': 'x'})
    assert items['qe_pw']['visible']  # inside Slurm jobs: the cluster's command
    assert not any(s.qe_launch for s in run_plan('sscha', slurm.config))
    text = workflow_outputs('sscha', form.config)[0].content()
    assert "if LAUNCHER == 'local':" in text and 'qe_command(' in text

    assert RunStep('x', ('bash', 'run_qe.sh'), ranks_env='NP', qe_launch=True).to_json()[
        'qe_launch'
    ]
