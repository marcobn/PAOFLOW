"""The linear-response (Kubo) property of paoflow-gen and paoflow-studio: one question set mapped onto
one PAOFLOW.kubo_response call (every observable, every Kubo-Bastin term)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from PAOFLOW.gen.outputs import workflow_outputs
from PAOFLOW.gen.paoflow_driver import ENERGY_WINDOW_PROPERTIES, PROPERTY_MENU
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.linear_response import (
    kubo_terms,
    linear_response_calls,
    parse_components,
)
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

pytestmark = pytest.mark.unit


def _cfg(**overrides):
    cfg = {
        'lr_observable': 'charge_current',
        'lr_term': 'fermi_sea',
        'lr_sea': 'gamma_to_zero',
        'lr_surface': 'total',
        'lr_components': ['xy'],
        'lr_gamma': 0.01,
        'lr_spinful': True,
        'is_2d': False,
    }
    cfg.update(overrides)
    return cfg


def _form(tmp_path, **answers):
    return resolve_form(WORKFLOWS['tb_model'], answers, Context(workdir=str(tmp_path)))


def test_the_property_is_appended_to_the_menu():
    # appended last: the menu numbers of the other properties (and the frozen scenarios) stay
    assert PROPERTY_MENU[-1][0] == 'linear_response'
    assert 'linear_response' in ENERGY_WINDOW_PROPERTIES


@pytest.mark.parametrize(
    'overrides, terms',
    [
        ({}, ('intrinsic',)),
        ({'lr_sea': 'finite_gamma'}, ('sea',)),
        ({'lr_term': 'fermi_surface'}, ('surface',)),
        ({'lr_term': 'fermi_surface', 'lr_surface': 'interband'}, ('surface_interband',)),
        ({'lr_term': 'total'}, ('total',)),
        (
            {'lr_term': 'both', 'lr_surface': 'intraband'},
            ('total', 'surface_intraband', 'intrinsic'),
        ),
        ({'lr_term': 'both', 'lr_sea': 'finite_gamma'}, ('total', 'surface', 'sea')),
    ],
)
def test_terms_map_onto_kubo_response(overrides, terms):
    assert kubo_terms(_cfg(**overrides)) == terms
    ((method, kwargs, _comment),) = linear_response_calls(_cfg(**overrides))
    assert method == 'kubo_response'
    assert kwargs['terms'] == terms and kwargs['observable'] == 'charge_current'
    assert kwargs['components'] == ['xy'] and kwargs['eta'] == 0.01 and kwargs['twoD'] is False


def test_every_observable_reaches_kubo_response():
    cfg = _cfg(
        lr_observable='spin_torque', lr_components=['yx', 'xx'], lr_temperature=300.0, is_2d=True
    )
    ((method, kwargs, _comment),) = linear_response_calls(cfg)
    assert kwargs == {
        'observable': 'spin_torque',
        'components': ['yx', 'xx'],
        'terms': ('intrinsic',),
        'eta': 0.01,
        'temperature': 300.0,
        'twoD': True,
    }


def test_questions_follow_the_model_and_check_components(tmp_path):
    spinless = _form(tmp_path, tb_model='honeycomb', properties=['linear_response'])
    choices = next(i for i in spinless.items if i.get('key') == 'lr_observable')['choices']
    assert choices == ['charge_current']
    spinful = _form(tmp_path, tb_model='kane_mele', properties=['linear_response'])
    choices = next(i for i in spinful.items if i.get('key') == 'lr_observable')['choices']
    assert choices == [
        'charge_current',
        'spin_current',
        'spin_density',
        'spin_torque',
    ]  # no orbital observables on models
    bad = _form(
        tmp_path,
        tb_model='kane_mele',
        properties=['linear_response'],
        lr_observable='spin_current',
        lr_components='xy',
    )
    assert 'lr_components' in bad.errors
    with pytest.raises(ValueError):
        parse_components('xq')
    terms = _form(
        tmp_path, tb_model='kane_mele', properties=['linear_response'], lr_observable='spin_current'
    )
    assert next(i for i in terms.items if i.get('key') == 'lr_term')['choices'] == [
        'fermi_sea',
        'fermi_surface',
        'both',
        'total',
    ]


def test_haldane_chern_number_from_the_generated_driver(tmp_path):
    """Haldane model in its gap: the intrinsic (Fermi-sea) charge Hall response gives C = +1 per spin
    (Kubo-Bastin sign; the spin-degenerate bands give 2 pi sigma_xy = 2C)."""
    form = _form(
        tmp_path,
        tb_model='honeycomb',
        honeycomb_t2='0.1',
        honeycomb_phi='1.5708',
        properties=['linear_response'],
        lr_observable='charge_current',
        lr_term='both',
        nfft='36',
        ne='41',
    )
    assert not form.errors, form.errors
    for f in workflow_outputs('tb_model', form.config):
        f.write(str(tmp_path))
    text = (tmp_path / 'main.py').read_text()
    assert "p.kubo_response(emin=EMIN, emax=EMAX, ne=NE, observable='charge_current'" in text
    assert "terms=('total', 'surface', 'intrinsic')" in text and 'Chern number' in text
    result = subprocess.run(
        [sys.executable, 'main.py'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    data = np.loadtxt(tmp_path / 'output' / 'kubo_charge_current_xy.dat')
    gap = np.argmin(np.abs(data[:, 0]))
    assert data.shape[1] == 4  # mu, total, surface, intrinsic
    chern = np.pi * data[gap, 3]
    assert abs(chern - 1.0) < 0.03
    assert abs(np.pi * data[gap, 1] - 1.0) < 0.03  # the full response at finite broadening too


def test_optical_response_maps_onto_kubo_optical_response():
    cfg = _cfg(lr_frequency=True, lr_omega_max=0.0, lr_nomega=201, lr_gamma=0.05, is_2d=True)
    ((method, kwargs, _comment),) = linear_response_calls(cfg)
    assert method == 'kubo_optical_response'
    assert kwargs == {
        'observable': 'charge_current',
        'components': ['xy'],
        'terms': ('total', 'interband', 'intraband'),
        'eta': 0.05,
        'omega_max': None,  # 0: the energy range of the bands
        'nomega': 201,
        'temperature': 0.0,
        'twoD': True,
    }
    assert (
        linear_response_calls(_cfg(lr_frequency=True, lr_omega_max=4.0))[0][1]['omega_max'] == 4.0
    )


def test_optical_questions_replace_the_static_terms(tmp_path):
    static = _form(tmp_path, tb_model='honeycomb', properties=['linear_response'])
    shown = {i.get('key') for i in static.items if i.get('visible')}
    assert 'lr_term' in shown and 'lr_omega_max' not in shown
    optical = _form(
        tmp_path, tb_model='honeycomb', properties=['linear_response'], lr_frequency='true'
    )
    shown = {i.get('key') for i in optical.items if i.get('visible')}
    assert {'lr_omega_max', 'lr_nomega', 'lr_gamma'} <= shown and 'lr_term' not in shown
    assert optical.config['lr_frequency'] is True


def test_optical_driver_runs(tmp_path):
    """The generated kubo_optical_response call runs on a model and writes the photon-energy table."""
    form = _form(
        tmp_path,
        tb_model='honeycomb',
        properties=['linear_response'],
        lr_frequency='true',
        lr_nomega='41',
        lr_omega_max='6',
        nfft='12',
        ne='21',
    )
    assert not form.errors, form.errors
    for f in workflow_outputs('tb_model', form.config):
        f.write(str(tmp_path))
    text = (tmp_path / 'main.py').read_text()
    assert (
        "p.kubo_optical_response(observable='charge_current'" in text
        and 'emin=EMIN' not in text.split('kubo_optical_response')[1].split('\n')[0]
    )
    result = subprocess.run(
        [sys.executable, 'main.py'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    files = sorted((tmp_path / 'output').glob('kubo_optical_charge_current_*.dat'))
    assert files
    data = np.loadtxt(files[0])
    assert data.shape[0] == 41 and data[-1, 0] == pytest.approx(6.0)
