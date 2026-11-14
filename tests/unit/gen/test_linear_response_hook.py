"""The linear-response (Kubo) property of paoflow-gen and paoflow-studio: one question set mapped onto
the existing PAOFLOW calls (spin_Hall, orbital_Hall, anomalous_Hall, linear_response)."""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np
import pytest

from PAOFLOW.gen.outputs import workflow_outputs
from PAOFLOW.gen.paoflow_driver import ENERGY_WINDOW_PROPERTIES, PROPERTY_MENU
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.linear_response import linear_response_calls, parse_components
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
    'overrides, expected',
    [
        ({}, [('anomalous_Hall', {'a_tensor': [[0, 1]], 'do_ac': True})]),
        ({'lr_spinful': False}, [('linear_response', {'response': 'ahc'})]),
        ({'lr_sea': 'finite_gamma'}, [('linear_response', {'response': 'ahc', 'full_chi2': True})]),
        ({'lr_term': 'fermi_surface'}, [('linear_response', {'response': 'cond'})]),
        (
            {'lr_term': 'fermi_surface', 'lr_surface': 'interband'},
            [('linear_response', {'response': 'cond', 'interband': True})],
        ),
        (
            {'lr_observable': 'spin_current', 'lr_components': ['zxy'], 'lr_term': 'both'},
            [
                ('spin_Hall', {'s_tensor': [[0, 1, 2]]}),
                ('linear_response', {'response': 'shc', 't_odd': True, 's_tensor': [[2, 0, 1]]}),
            ],
        ),
        (
            {'lr_observable': 'orbital_current', 'lr_components': ['zxy']},
            [('orbital_Hall', {'o_tensor': [[0, 1, 2]]})],
        ),
        (
            {'lr_observable': 'spin_density', 'lr_components': ['yx'], 'lr_term': 'both'},
            [
                ('linear_response', {'response': 'ree', 't_odd': True, 'full_chi2': False}),
                ('linear_response', {'response': 'ree', 'a_tensor': [[1, 0]]}),
            ],
        ),
    ],
)
def test_terms_map_onto_the_existing_calls(overrides, expected):
    calls = linear_response_calls(_cfg(**overrides))
    assert [c[0] for c in calls] == [e[0] for e in expected]
    for (method, kwargs, _comment), (_m, wanted) in zip(calls, expected):
        for key, value in wanted.items():
            assert kwargs[key] == value, (method, key, kwargs[key])
        if method == 'linear_response':
            # linear_response reads a_tensor only when s_tensor is given: both always passed
            assert kwargs['s_tensor'] and kwargs['a_tensor']


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
    ]  # no orbital current on models
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
    ]


def test_haldane_chern_number_from_the_generated_driver(tmp_path):
    """Haldane model in its gap: the Fermi-sea term of the charge current gives C = -1."""
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
    assert "p.linear_response(eminH=EMIN, emaxH=EMAX, esize=NE, response='ahc'" in text
    assert "response='cond'" in text and 'Chern number' in text
    result = subprocess.run(
        [sys.executable, 'main.py'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env={**os.environ, 'OMP_NUM_THREADS': '1'},
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    data = np.loadtxt(tmp_path / 'output' / 'AHC_xy.dat')
    chern = 2 * np.pi * data[np.argmin(np.abs(data[:, 0])), 1]
    assert abs(chern + 1.0) < 0.03
    assert (tmp_path / 'output' / 'Cond_xy.dat').is_file()
