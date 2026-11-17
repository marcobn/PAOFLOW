"""paoflow-gen: the spin-susceptibility property (questions, assembly and main.py block)."""

from __future__ import annotations

import pytest

from PAOFLOW.gen import paoflow_driver as d
from PAOFLOW.gen.questions.spin_response import assemble_spin_response, spin_methods

pytestmark = pytest.mark.unit


def _cfg(**kw):
    cfg = {
        'savedir': 'Pd.save', 'prefix': 'Pd', 'upfs': ['Pd.upf'], 'basisdir': 'BASIS_PS',
        'outputdir': 'output', 'ibrav': 2, 'is_2d': False, 'properties': ['dos', 'spin_susceptibility'],
        'std_basis': 'standard', 'npool': 1, 'smearing': 'm-v', 'spin_orbit': False, 'nk': 400,
        'emin': -8.0, 'emax': 4.0, 'ne': 1000, 'do_pdos': False, 'interpolate': False, 'nfft': 0,
        'projector': 'internal',
    }  # fmt: skip
    cfg.update(kw)
    return cfg


def test_menu_appends_the_property():
    keys = [key for key, _ in d.PROPERTY_MENU]
    # menu numbers are what CLI users type: entries are only ever appended
    assert keys.index('spin_susceptibility') == 13 and keys.index('linear_response') == 12


def test_methods_are_parsed_and_checked():
    assert spin_methods('bare, stoner') == ('bare', 'stoner')
    with pytest.raises(ValueError):
        spin_methods('bare, rpa')


def test_assembly_drops_what_the_run_cannot_do():
    answers = {'spin_methods': ('bare', 'stoner', 'alda'), 'spin_nk': 12, 'spin_degauss': 0.2,
               'spin_field': 5.0, 'spin_ordered': True}  # fmt: skip
    assert assemble_spin_response(_cfg(), answers)['spin_methods'] == ('bare', 'stoner', 'alda')
    assert assemble_spin_response(_cfg(spin_orbit=True), answers)['spin_methods'] == (
        'bare',
        'stoner',
    )
    projwfc = assemble_spin_response(_cfg(projector='projwfc'), answers)
    assert projwfc['spin_methods'] == ('bare',) and projwfc['spin_ordered'] is False


def test_main_script_runs_the_spin_response_on_its_own_instance():
    cfg = assemble_spin_response(
        _cfg(), {'spin_methods': ('bare', 'stoner', 'alda'), 'spin_nk': 12, 'spin_ordered': True}
    )
    text = d.build_run_script(cfg)
    compile(text, 'main.py', 'exec')
    assert "SPIN_METHODS = ('bare', 'stoner', 'alda')" in text and 'SPIN_NK = (12, 12, 12)' in text
    block = text[text.index('def run_spin_response():') :].split('def main')[0]
    assert 'p.pao_hamiltonian()' in block and 'p.stoner_parameters()' in block
    assert 'p.spin_susceptibility(' in block and 'p.self_consistent_spin(' in block
    assert 'p.projections(' in block and 'STD_BASIS' in block
    assert '    run_spin_response()' in text and '    run_properties()' in text
    # the standard run does not compute it
    standard = text[text.index('def run_properties():') : text.index('def run_spin_response():')]
    assert 'spin_susceptibility' not in standard
