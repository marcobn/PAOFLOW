"""The ``ep_model`` workflow of paoflow-gen: electron-phonon coupling of a model crystal, no DFT."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys

import pytest

from PAOFLOW.gen.outputs import run_plan, workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.ep_model import EP_MODELS, build_model, model_function
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

pytestmark = pytest.mark.unit


def _form(tmp_path, **answers):
    return resolve_form(WORKFLOWS['ep_model'], answers, Context(workdir=str(tmp_path)))


def _choices(form, key):
    item = next(i for i in form.items if i.get('key') == key)
    return [c[0] if isinstance(c, list) else c for c in item['choices']]


@pytest.mark.parametrize('name', sorted(EP_MODELS))
def test_every_model_resolves_with_its_defaults(name, tmp_path):
    form = _form(tmp_path, ep_model=name, properties=['bands', 'phonons'])
    assert not form.errors, form.errors
    cfg = form.config
    assert cfg['model_function'] == EP_MODELS[name][0]
    signature = inspect.signature(model_function(name)).parameters
    assert set(cfg['model_arguments']) == {
        k for k, p in signature.items() if p.default is not p.empty
    }
    assert cfg['band_path'] and set(cfg['high_sym']) >= set(
        cfg['band_path'].replace('|', '-').split('-')
    )
    note = [i['text'] for i in form.items if i.get('type') == 'note']
    assert note and 'phonons up to' in note[0]


def test_properties_follow_metal_or_semiconductor(tmp_path):
    metal = _choices(_form(tmp_path, ep_model='cubic_metal'), 'properties')
    assert {'superconductivity', 'migdal_eliashberg', 'resistivity', 'self_energy'} <= set(metal)
    assert 'mobility' not in metal and 'absorption' not in metal
    semiconductor = _choices(_form(tmp_path, ep_model='indirect_semiconductor'), 'properties')
    assert {'mobility', 'absorption'} <= set(semiconductor)
    assert 'superconductivity' not in semiconductor
    # the Migdal-Eliashberg equations read the Eliashberg function: the coupling step comes first
    form = _form(tmp_path, ep_model='cubic_metal', properties=['migdal_eliashberg'])
    assert form.config['properties'] == ['superconductivity', 'migdal_eliashberg']


def test_grids_and_path_follow_the_model(tmp_path):
    chain = _form(
        tmp_path,
        ep_model='indirect_semiconductor',
        indirect_semiconductor_n_bands='2',
        properties=['bands', 'mobility', 'absorption'],
    )
    assert not chain.errors, chain.errors
    assert chain.config['band_path'] == 'gG-X'
    assert chain.config['mobility_nk'] == 1200 and chain.config['absorption_nk'] == 400
    assert build_model(chain.answers).dimension == 1
    square = _form(tmp_path, ep_model='light_atom_metal', properties=['superconductivity'])
    assert (square.config['ep_nk'], square.config['ep_nq']) == (48, 12)
    bad = _form(
        tmp_path, ep_model='cubic_metal', properties=['superconductivity'], ep_nk='40', ep_nq='12'
    )
    assert 'ep_nq' in bad.errors


def test_generated_drivers_run_on_the_models(tmp_path):
    cases = {
        'metal': dict(
            ep_model='cubic_metal',
            properties=['bands', 'phonons', 'migdal_eliashberg', 'resistivity', 'self_energy'],
            ep_nk='16',
            ep_nq='4',
            me_temperatures='2 6 10 14',
            rho_temperatures='100 300',
        ),
        'semiconductor': dict(
            ep_model='indirect_semiconductor',
            indirect_semiconductor_n_bands='2',
            properties=['phonons', 'mobility', 'absorption'],
            mobility_nk='300',
            absorption_nk='120',
            mobility_temperatures='300',
        ),
    }
    expected = {
        'metal': (
            'bands_0.dat',
            'cubic_metal_band.dat',
            'alpha2F.dat',
            'eliashberg.npz',
            'resistivity.dat',
            'self_energy_0.0K.dat',
            os.path.join('migdal_eliashberg', 'gap_vs_T.dat'),
        ),
        'semiconductor': ('indirect_2_band.dat', 'mobility.dat', 'alpha_300.0K.dat'),
    }
    for case, answers in cases.items():
        folder = tmp_path / case
        folder.mkdir()
        form = _form(folder, **answers)
        assert not form.errors, form.errors
        files = workflow_outputs('ep_model', form.config)
        assert [f.path for f in files] == ['main.ep_model.py', 'plot.ep_model.py']
        text = files[0].content()
        assert 'from PAOFLOW.models.electron_phonon import' in text and 'savedir' not in text
        steps = run_plan('ep_model', form.config, files)
        assert steps[0].argv == ('python', 'main.ep_model.py') and not steps[0].mpi
        assert not steps[1].interactive  # the plot script saves its figures
        for f in files:
            f.write(str(folder))
        result = subprocess.run(
            [sys.executable, 'main.ep_model.py'],
            cwd=folder,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, 'OMP_NUM_THREADS': '1'},
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
        for name in expected[case]:
            assert (folder / 'output' / name).is_file(), name
        from PAOFLOW.results import ResultsFolder

        kinds = {ref.kind for ref in ResultsFolder(str(folder / 'output')).datasets}
        wanted = (
            {
                'bands',
                'phonon_bands',
                'alpha2f',
                'gap_vs_t',
                'resistivity',
                'eliashberg_self_energy',
            }
            if case == 'metal'
            else {'phonon_bands', 'mobility', 'phonon_assisted_absorption', 'indirect_eps2'}
        )
        assert wanted <= kinds, wanted - kinds
