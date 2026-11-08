"""Band paths of the generator: defaults, validation, the custom path in main.py, the preview."""

from __future__ import annotations

import os
import shutil

import pytest

from PAOFLOW.gen import paoflow_driver as pd
import numpy as np

from PAOFLOW.gen.band_paths import (
    camera_eye,
    check_band_path,
    default_band_path,
    path_figure,
    path_labels,
)
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

SI = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example01'
)


def test_default_paths():
    path, points = default_band_path(SI, 'silicon.save', 2, False)
    assert path == 'gG-X-W-K-gG-L-U-W-L-K|U-X'
    assert points[0] == ('gG', 0.0, 0.0, 0.0) and {p[0] for p in points} == set(path_labels(path))
    assert default_band_path(SI, 'silicon.save', 4, True)[0] == 'gG-M-K-gG'
    assert default_band_path(SI, 'silicon.save', 0, False) == ('', [])
    assert default_band_path(SI, 'missing.save', 2, False) == ('', [])


def test_path_checks():
    points = [('gG', 0, 0, 0), ('X', 0, 0.5, 0.5)]
    assert check_band_path('gG-X', points) is None
    assert check_band_path('gG-X|X-gG', points) is None
    assert 'Q' in check_band_path('gG-Q', points)
    assert check_band_path('gG', points) and check_band_path('', points)


def test_preview_geometry():
    path, points = default_band_path(SI, 'silicon.save', 2, False)
    result = path_figure(SI, 'silicon.save', path, points)
    assert result['length'] > 0 and len(result['figure']['data']) == 3
    labels = result['figure']['data'][2]['text']
    assert labels[0] == 'Γ' and 'X' in labels
    assert 'error' in path_figure(SI, 'silicon.save', 'gG-Q', points)


def test_camera_keeps_the_points_apart():
    """fcc seen along about [111] (the old fixed camera) put Γ under L; the chosen view does not."""
    path, points = default_band_path(SI, 'silicon.save', 2, False)
    figure = path_figure(SI, 'silicon.save', path, points)['figure']
    trace = figure['data'][2]
    xyz = np.array([trace['x'], trace['y'], trace['z']]).T

    def smallest_gap(eye):
        direction = np.array([eye['x'], eye['y'], eye['z']])
        direction /= np.linalg.norm(direction)
        projected = xyz - np.outer(xyz @ direction, direction)
        i, j = np.triu_indices(len(xyz), 1)
        return np.linalg.norm(projected[i] - projected[j], axis=1).min()

    chosen = figure['layout']['scene']['camera']['eye']
    span = np.ptp(xyz, axis=0).max()
    assert smallest_gap(chosen) > 0.15 * span
    assert smallest_gap(chosen) > 1.5 * smallest_gap({'x': 1.05, 'y': 1.05, 'z': 0.7})
    assert camera_eye(xyz[:1]) == camera_eye(np.zeros((0, 3)))  # one point: the default view


def test_custom_path_reaches_main_py():
    context = Context(workdir=SI)
    default = resolve_form(WORKFLOWS['run'], {'properties': ['bands']}, context)
    assert 'band_path' not in default.config and 'BAND_PATH' not in pd.build_run_script(
        default.config
    )
    custom = resolve_form(
        WORKFLOWS['run'], {'properties': ['bands'], 'band_path': 'gG-X|L-gG'}, context
    )
    script = pd.build_run_script(custom.config)
    assert "BAND_PATH = 'gG-X|L-gG'" in script and "'L': (0.0, 0.5, 0.0)" in script
    assert 'band_path=BAND_PATH' in script
    rows = [['gG', '0', '0', '0'], ['Z', '0', '0', '0.5']]
    points = resolve_form(
        WORKFLOWS['run'], {'properties': ['bands'], 'high_sym': rows, 'band_path': 'gG-Z'}, context
    )
    assert points.config['high_sym'] == {'gG': (0.0, 0.0, 0.0), 'Z': (0.0, 0.0, 0.5)}
    wrong = resolve_form(WORKFLOWS['run'], {'properties': ['bands'], 'band_path': 'gG-Q'}, context)
    assert wrong.errors['band_path'] == 'no coordinates for Q'


@pytest.mark.parametrize(
    'rows, message',
    [([['gG', '0', '0']], 'label and three'), ([['X', 'a', '0', '0']], 'not numbers')],
)
def test_point_rows_errors(rows, message):
    form = resolve_form(
        WORKFLOWS['run'], {'properties': ['bands'], 'high_sym': rows}, Context(workdir=SI)
    )
    assert message in form.errors['high_sym']


def test_save_problems_name_the_cause(tmp_path):
    from PAOFLOW.gen.band_paths import save_problem, saves_in

    workdir = str(tmp_path)
    assert 'run pw.x here first' in save_problem(workdir, 'pwscf.save')
    (tmp_path / 'Si.save').mkdir()  # pw.x still running: no data-file-schema.xml yet
    assert saves_in(workdir) == [] and 'has not finished' in save_problem(workdir, 'Si.save')
    shutil.copy(os.path.join(SI, 'silicon.save', 'data-file-schema.xml'), tmp_path / 'Si.save')
    assert saves_in(workdir) == ['Si.save'] and save_problem(workdir, 'Si.save') is None
    stale = save_problem(workdir, 'gaas.save')
    assert 'gaas.save does not exist' in stale and 'found here: Si.save' in stale
    result = path_figure(workdir, 'gaas.save', 'gG-X', [('gG', 0, 0, 0), ('X', 0, 0.5, 0.5)])
    assert result['saves'] == ['Si.save'] and 'gaas.save does not exist' in result['error']
