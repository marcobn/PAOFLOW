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
