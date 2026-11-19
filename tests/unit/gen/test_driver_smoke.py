"""Generated drivers run: a property main.py on the silicon save of tests/data, end to end.

The closed-form tests check the properties on models; this checks the wiring of what the generator
writes (the order of the calls, the constants, the Hamiltonian changes, the restart branch) on a real
save. The save has projwfc.x projections and no wavefunctions, so the run takes the projwfc route.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from PAOFLOW.gen.outputs import workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

pytestmark = pytest.mark.unit

SILICON = Path(__file__).resolve().parents[2] / 'data' / 'silicon'


def _run(folder: Path, *script: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, OMP_NUM_THREADS='1', MPLBACKEND='Agg')
    return subprocess.run(
        [sys.executable, *script], cwd=folder, env=env, capture_output=True, text=True, timeout=600
    )


def test_property_run_on_the_silicon_save(tmp_path):
    shutil.copytree(SILICON / 'silicon.save', tmp_path / 'silicon.save')
    shutil.copy(SILICON / 'scf.in', tmp_path / 'scf.in')
    answers = {
        'savedir': 'silicon.save',
        'prefix': 'silicon',
        'projector': 'projwfc',
        'ibrav': 2,
        'properties': [
            'bands',
            'dos',
            'transport',
            'site_projected_bands',
            'hamiltonian_changes',
            'restart',
        ],
        'site_atoms': ['1'],
        'nk': 40,
        'emin': -3.0,
        'emax': 3.0,
        'ne': 301,
        'interpolate': True,
        'nfft': 12,
        'efield': '0 0 0.001',
    }
    form = resolve_form(WORKFLOWS['run'], answers, Context(workdir=str(tmp_path)))
    assert form.complete, form.errors
    files = workflow_outputs('run', form.config)
    assert [f.path for f in files] == ['main.py', 'proj.in', 'plot.py']
    for generated in files:
        generated.write(str(tmp_path))
    first = _run(tmp_path, 'main.py')
    assert first.returncode == 0, first.stdout[-3000:] + first.stderr[-3000:]
    output = tmp_path / 'output'
    for name in ('bands_0.dat', 'dosdk_0.dat', 'sigmagauss_0.dat', 'site-projected-bands_0.dat'):
        assert (output / name).stat().st_size > 0, name
    assert (tmp_path / 'paoflow_dump_0.json').is_file()
    dos = (output / 'dosdk_0.dat').read_bytes()
    (output / 'dosdk_0.dat').unlink()
    second = _run(tmp_path, 'main.py')  # from the restart dump: no projections, same DOS
    assert second.returncode == 0, second.stdout[-3000:] + second.stderr[-3000:]
    assert (output / 'dosdk_0.dat').read_bytes() == dos
    plots = subprocess.run(
        [sys.executable, 'plot.py', '--all'], cwd=tmp_path, capture_output=True, text=True,
        env=dict(os.environ, MPLBACKEND='Agg'), timeout=300,
    )  # fmt: skip
    assert plots.returncode == 0, plots.stdout[-3000:] + plots.stderr[-3000:]


def test_sparse_property_run_on_the_silicon_save(tmp_path):
    shutil.copytree(SILICON / 'silicon.save', tmp_path / 'silicon.save')
    shutil.copy(SILICON / 'scf.in', tmp_path / 'scf.in')
    answers = {
        'savedir': 'silicon.save',
        'prefix': 'silicon',
        'projector': 'projwfc',
        'ibrav': 2,
        'properties': ['bands', 'dos', 'transport', 'hamiltonian_changes', 'sparse'],
        'nk': 40,
        'emin': -3.0,
        'emax': 3.0,
        'ne': 301,
        'interpolate': True,
        'nfft': 12,
        'doubling': '1 0 0',
    }
    form = resolve_form(WORKFLOWS['run'], answers, Context(workdir=str(tmp_path)))
    assert form.complete, form.errors
    for generated in workflow_outputs('run', form.config):
        generated.write(str(tmp_path))
    run = _run(tmp_path, 'main.py')
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-3000:]
    output = tmp_path / 'output'
    for name in ('sparse.log', 'bands_0.dat', 'dosdk_0.dat', 'sigmagauss_0.dat'):
        assert (output / name).stat().st_size > 0, name
    plots = _run(tmp_path, 'plot.py', '--all')  # GPAO reads the sparse files as the dense ones
    assert plots.returncode == 0, plots.stdout[-3000:] + plots.stderr[-3000:]
