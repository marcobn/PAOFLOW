"""The Landauer workflow of paoflow-gen and paoflow-studio: questions, main.landauer.py, its plot script
and its run step."""

from __future__ import annotations

import pytest

from PAOFLOW.gen.outputs import LANDAUER_KINDS, run_plan, workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.landauer import (
    parse_biases,
    parse_charge_widths,
    parse_transverse_mesh,
)
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form
from PAOFLOW.results.catalogue import KINDS

pytestmark = pytest.mark.unit


@pytest.fixture
def save_dir(tmp_path):
    save = tmp_path / 'al.save'
    save.mkdir()
    (save / 'Al.upf').write_text('<UPF version="2.0.1"></UPF>\n')
    return tmp_path


def _form(workdir, **answers):
    form = resolve_form(WORKFLOWS['landauer'], answers, Context(workdir=str(workdir)))
    shown = {i['key'] for i in form.items if 'key' in i and i['visible']}
    return form, shown


def _driver(form):
    files = workflow_outputs('landauer', form.config)
    return files, files[0].content()


def test_bulk_defaults(save_dir):
    form, shown = _form(save_dir)
    assert not form.errors, form.errors
    assert {'calculation', 'transport_direction', 'transverse_nk', 'current'} <= shown
    assert not {'lead_layers', 'finite_bias', 'bias_values', 'electrostatics'} & shown
    files, text = _driver(form)
    assert [f.path for f in files] == ['main.landauer.py', 'plot.landauer.py']
    compile(text, 'main.landauer.py', 'exec')
    assert "CALCULATION = 'bulk'" in text and 'TRANSVERSE_NK = (0, 0)' in text
    assert 'from PAOFLOW.Transport import Transport' in text
    assert 'p.projections(basispath=BASISPATH, configuration=BASIS)' in text
    assert 'PTHR = 0.95' in text and "ensure_basis(preset='extended')" in text
    assert 't.compute_transmission()' in text and 't.compute_dos()' in text
    assert 'mu_L=0.5, mu_R=-0.5' in text  # positive current for a positive bias
    assert 'FINITE_BIAS' not in text and 'configure_bias_self_consistency' not in text


def test_conductor_with_a_three_dimensional_finite_bias_scan(save_dir):
    form, shown = _form(
        save_dir,
        calculation='conductor',
        lead_layers='3',
        transverse_nk='1 1',
        finite_bias='true',
        bias_values='0 0.2 0.6',
        electrostatics='3d',
        charge_widths='Au 1.9, S 1.1',
        linear_ramp='true',
    )
    assert not form.errors, form.errors
    assert {'bias_values', 'electrostatics', 'charge_widths', 'potential_grid'} <= shown
    cfg = form.config
    assert cfg['finite_bias'] and cfg['electrostatics'] == '3d'
    assert cfg['charge_widths'] == {'Au': 1.9, 'S': 1.1} and cfg['transverse_nk'] == (1, 1)
    _files, text = _driver(form)
    compile(text, 'main.landauer.py', 'exec')
    assert 'LEAD_LAYERS = 3' in text and 'FINITE_BIAS = [0.0, 0.2, 0.6]' in text
    assert "CHARGE_WIDTHS = {'Au': 1.9, 'S': 1.1}" in text
    assert "PROFILES = ('linear', 'self_consistent')" in text
    assert 'POTENTIAL_GRID_BIASES = [0.6]' in text
    assert 't.configure_bias_self_consistency(' in text
    assert 't.compute_current_self_consistent(bias=FINITE_BIAS' in text


def test_finite_bias_needs_a_conductor(save_dir):
    form, shown = _form(save_dir, calculation='bulk', finite_bias='true')
    assert 'finite_bias' not in shown and form.config['finite_bias'] is False
    planar, _ = _form(save_dir, calculation='conductor', finite_bias='true')
    _files, text = _driver(planar)
    assert "ELECTROSTATICS = 'planar'" in text and 'POTENTIAL_GRID_BIASES = None' in text


def test_projwfc_route_and_doublings(save_dir):
    form, _ = _form(save_dir, projector='projwfc', doublings='1', transport_direction='x')
    _files, text = _driver(form)
    assert 'p.read_atomic_proj_QE()' in text and 'ensure_basis' not in text
    assert 'DOUBLINGS = 1' in text and "TRANSPORT_DIRECTION = 'x'" in text


def test_parsers():
    assert parse_transverse_mesh('') == (0, 0) and parse_transverse_mesh('6, 6') == (6, 6)
    with pytest.raises(ValueError):
        parse_transverse_mesh('6')
    assert parse_biases('0, 0.1 0.5') == [0.0, 0.1, 0.5]
    with pytest.raises(ValueError):
        parse_biases(' ')
    assert parse_charge_widths('') is None
    assert parse_charge_widths('Au 1.9,S 1.1') == {'Au': 1.9, 'S': 1.1}
    with pytest.raises(ValueError):
        parse_charge_widths('Au')
    with pytest.raises(ValueError):
        parse_charge_widths('Au -1')


def test_plot_script_and_run_step(save_dir):
    form, _ = _form(save_dir, calculation='conductor', finite_bias='true')
    files = workflow_outputs('landauer', form.config)
    plot = files[1].content()
    compile(plot, 'plot.landauer.py', 'exec')
    known = {kind.name for kind in KINDS}
    assert set(LANDAUER_KINDS) <= known
    steps = run_plan('landauer', form.config, files)
    assert steps[0].argv == ('python', 'main.landauer.py') and steps[0].mpi
    assert any(step.argv == ('python', 'plot.landauer.py') for step in steps)


def test_principal_layer_and_eigenchannels(save_dir):
    """A lead cut from a junction (bulk on its first layers) and the eigenchannels of a conductor;
    without them the script keeps its default lines."""
    form, shown = _form(save_dir, principal_layers=3)
    assert 'principal_layers' in shown and 'eigenchannels' not in shown
    _, text = _driver(form)
    compile(text, 'main.landauer.py', 'exec')
    assert 'PRINCIPAL_LAYERS = 3' in text and 'central_layers=PRINCIPAL_LAYERS' in text
    form, shown = _form(
        save_dir, calculation='conductor', eigenchannels=4, eigenchannel_energy='0.0'
    )
    assert 'principal_layers' not in shown and 'eigenchannel_energy' in shown
    _, text = _driver(form)
    compile(text, 'main.landauer.py', 'exec')
    assert 'EIGENCHANNELS = 4' in text and 'EIGENCHANNEL_ENERGY = 0.0' in text
    assert 't.configure_eigenchannels(do_eigenchannels=True, neigchnx=EIGENCHANNELS,' in text
    _, plain = _driver(_form(save_dir, calculation='conductor')[0])
    assert 'EIGENCHANNELS' not in plain and 'PRINCIPAL_LAYERS' not in plain
