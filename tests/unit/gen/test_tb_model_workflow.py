"""The ``tb_model`` workflow of paoflow-gen: a property run on a tight-binding model, no DFT."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import numpy as np
import pytest

from PAOFLOW.gen.outputs import run_plan, workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form
from PAOFLOW.gen.questions.tb_model import (
    TB_MODELS,
    build_model,
    model_band_path,
    model_function,
)

pytestmark = pytest.mark.unit


def _form(tmp_path, **answers):
    return resolve_form(WORKFLOWS['tb_model'], answers, Context(workdir=str(tmp_path)))


def _choices(form, key):
    item = next(i for i in form.items if i.get('key') == key)
    return [c[0] if isinstance(c, list) else c for c in item['choices']]


@pytest.mark.parametrize('name', sorted(TB_MODELS))
def test_every_model_resolves_with_its_defaults(name, tmp_path):
    form = _form(tmp_path, tb_model=name, properties=['bands'])
    assert not form.errors, form.errors
    cfg = form.config
    assert cfg['model_function'] == TB_MODELS[name][0]
    # one question per keyword parameter of the model function, with its default
    signature = inspect.signature(model_function(name)).parameters
    assert set(cfg['model_arguments']) == {
        k for k, p in signature.items() if p.default is not p.empty
    }
    model = model_function(name)(**cfg['model_arguments'])
    assert cfg['spinful'] == model.spinful and cfg['ibrav'] == 0
    assert cfg['band_path'] and set(cfg['high_sym']) >= set(
        cfg['band_path'].replace('|', '-').split('-')
    )


def test_properties_follow_the_model(tmp_path):
    honeycomb = _choices(_form(tmp_path, tb_model='honeycomb'), 'properties')
    assert 'spin_Hall' not in honeycomb and 'transport' not in honeycomb  # spinless, 2D
    assert {'bands', 'dos', 'optical', 'nonlinear'} <= set(honeycomb)
    kane_mele = _choices(_form(tmp_path, tb_model='kane_mele'), 'properties')
    assert {'spin_texture', 'spin_Hall', 'anomalous_Hall'} <= set(kane_mele)
    weyl = _choices(_form(tmp_path, tb_model='weyl'), 'properties')
    assert 'transport' in weyl  # 3D
    # a property the model cannot run (spin Hall of a spinless model) is left out
    switched = _form(tmp_path, tb_model='honeycomb', properties=['bands', 'spin_Hall'])
    assert not switched.errors and switched.config['properties'] == ['bands']


def test_band_paths_from_the_lattice(tmp_path):
    for name, expected in (
        ('rice_mele', 'gG-X'),
        ('kane_mele', 'gG-M-K-gG'),
        ('weyl', 'gG-X-M-gG-R-X|M-R'),
    ):
        model = build_model({'tb_model': name})
        path, points = model_band_path(model)
        assert path == expected
    # K of the hexagonal lattice is a corner of the zone: as far from Γ as from two other points of
    # the reciprocal lattice, and nearer to none
    model = build_model({'tb_model': 'kane_mele'})
    path, points = model_band_path(model)
    reciprocal = np.linalg.inv(model.lattice_vectors).T[:2]
    k = np.array(dict((p[0], p[1:]) for p in points)['K']) @ np.linalg.inv(model.lattice_vectors).T
    lattice = [i * reciprocal[0] + j * reciprocal[1] for i in range(-2, 3) for j in range(-2, 3)]
    distances = np.sort([np.linalg.norm(k - g) for g in lattice])
    assert np.isclose(distances[0], np.linalg.norm(k)) and np.allclose(distances[:3], distances[0])
    assert distances[3] > distances[0] + 1e-6
    # the 2-band parabolic model is a chain, the 4-band one cubic
    two = _form(tmp_path, tb_model='parabolic', parabolic_n_bands='2', properties=['bands'])
    assert two.config['band_path'] == 'gG-X' and two.config['periodic'] == (True, False, False)


def test_parameters_reach_the_model_and_errors_are_reported(tmp_path):
    form = _form(
        tmp_path,
        tb_model='honeycomb',
        honeycomb_t2='0.1',
        honeycomb_phi='1.5708',
        properties=['bands'],
    )
    assert form.config['model_arguments']['t2'] == 0.1
    tmd = _form(tmp_path, tb_model='tmd', tmd_material='WSe2', tmd_soc='0.2', properties=['bands'])
    assert tmd.config['model_arguments']['material'] == 'WSe2'
    assert tmd.config['model_arguments']['soc'] == 0.2
    note = [
        i['text'] for i in tmd.items if i.get('type') == 'note' and 'orbitals' in i.get('text', '')
    ]
    assert note and '6 orbitals' in note[0] and 'spinful' in note[0]
    bad = _form(
        tmp_path, tb_model='tmd', tmd_material='MoS2', tmd_functional='LDA', properties=['bands']
    )
    if bad.errors:  # LDA with the tabulated (GGA) spin-orbit coupling is refused by the model
        assert 'model' in ' '.join(bad.errors.values()) or '_assemble' in bad.errors


def test_generated_driver_runs_on_the_model(tmp_path):
    form = _form(
        tmp_path,
        tb_model='kane_mele',
        properties=['bands', 'dos', 'spin_Hall', 'nonlinear'],
        nk='40',
        nfft='24',
        nlo_nk='24',
        nlo_ne='21',
    )
    assert not form.errors, form.errors
    files = workflow_outputs('tb_model', form.config)
    assert [f.path for f in files] == ['main.py', 'plot.py']
    text = files[0].content()
    assert 'from PAOFLOW.models.models import kane_mele_model' in text
    assert 'NLO_NK = (24, 24, 1)' in text and 'NFFT = (24, 24, 1)' in text
    assert 'savedir' not in text.lower() and 'projections' not in text
    steps = run_plan('tb_model', form.config, files)
    assert steps[0].argv == ('python', 'main.py') and steps[0].mpi
    for f in files:
        f.write(str(tmp_path))
    result = subprocess.run(
        [sys.executable, 'main.py'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    output = os.listdir(tmp_path / 'output')
    for name in ('bands_0.dat', 'dosdk_0.dat', 'shcEf_z_xy.dat', 'shgr_xxx.dat'):
        assert name in output, name
