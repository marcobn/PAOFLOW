"""The run properties added for the example cases: questions, ``main.py`` and ``plot.py``."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form
from PAOFLOW.gen.questions.run_extras import (
    EXTRA_PROPERTY_MENU,
    float_tuple,
    hubbard_row,
    int_triplet,
    optional_vector,
    soc_row,
    top_valence_band,
)
from PAOFLOW.gen.run_extras_driver import hubbard_shifts

pytestmark = pytest.mark.unit

SILICON = Path(__file__).resolve().parents[2] / 'data' / 'silicon'
EVERYTHING = [key for key, _label in EXTRA_PROPERTY_MENU]


@pytest.fixture
def workdir(tmp_path):
    shutil.copytree(SILICON / 'silicon.save', tmp_path / 'silicon.save')
    shutil.copy(SILICON / 'scf.in', tmp_path / 'scf.in')
    return tmp_path


def _config(workdir, answers):
    form = resolve_form(WORKFLOWS['run'], answers, Context(workdir=str(workdir)))
    assert form.complete, form.errors
    return form.config


def test_menu_numbers_stay():
    keys = [key for key, _ in pd.PROPERTY_MENU]
    assert keys.index('spin_susceptibility') == 13 and keys[14:] == EVERYTHING


def test_parsers():
    assert hubbard_row('Al-3p 0.1') == ('Al-3p', 0.1)
    assert soc_row('Pt, 0.0, 0.553') == ('Pt', 0.0, 0.553)
    assert optional_vector('') is None and optional_vector('0 0 0') is None
    assert optional_vector('0 0 0.1') == (0.0, 0.0, 0.1)
    assert int_triplet('1, 0, 0') == (1, 0, 0) and float_tuple('0, 0.5') == (0.0, 0.5)
    for parse, text in ((hubbard_row, 'Al 0.1'), (hubbard_row, 'Al-3p x'), (soc_row, 'Pt 1'),
                        (optional_vector, '1 2'), (int_triplet, '1 -1 0'), (float_tuple, '')):  # fmt: skip
        with pytest.raises(ValueError):
            parse(text)


def test_hubbard_shifts():
    assert hubbard_shifts([]) is None
    assert hubbard_shifts([('Al-3p', 0.1), ('P-3p', 2.31), ('Al-3p', 0.2)]) == {
        'Al-3p': 0.2, 'P-3p': 2.31,
    }  # fmt: skip


def test_top_valence_band(workdir):
    cfg = {'workdir': str(workdir), 'savedir': 'silicon.save'}
    assert top_valence_band(cfg) == 3  # 8 electrons, two per band
    assert top_valence_band({**cfg, 'spin_orbit': True}) == 7
    assert top_valence_band({**cfg, 'savedir': 'missing.save'}) == 0


def test_every_extra_in_main_and_plot(workdir):
    answers = {
        'properties': EVERYTHING,
        'site_atoms': ['2'],
        'scattering': ['acoustic', 'impurity'],
        'hubbard_shifts': [['Si-3p', '0.1']],
        'adhoc_soc': [['Si', '0.0', '0.1']],
        'doubling': '1 0 0',
    }
    cfg = _config(workdir, answers)
    assert cfg['properties'][0] == 'bands'  # the site projection reads the band path
    assert cfg['site_atoms'] == [1] and cfg['doping_emin'] == -13.0
    assert set(cfg['tau_parameters']) == {'ms', 'rho', 'v', 'D_ac', 'eps_0', 'eps_inf', 'nI', 'Zi'}
    assert cfg['tau_parameters']['nI'] == 1.0e18 and cfg['top_valence_band'] == 3
    main = pd.build_run_script(cfg)
    compile(main, 'main.py', 'exec')
    for line in (
        'p.site_projected_bands(site_proj=SITE_ATOMS)',
        'p.jdos(delta=JDOS_DELTA, emin=0.0, emax=JDOS_EMAX, ne=JDOS_NE)',
        'p.rashba_edelstein(emin=EMIN, emax=EMAX, ne=NE)',
        'p.find_weyl_points(search_grid=WEYL_GRID, symmetrize=WEYL_SYMMETRIZE)',
        'p.z2_invariant(k3_planes=Z2_PLANES)',
        'mirror = p.mirror_chern_number()',
        'def run_doped_transport():',
        "ADHOC_SOC = {'Si': [0.0, 0.1]}",
        "HUBBARD_U = {'Si-3p': 0.1}",
        'p.restart_dump(fname_prefix=RESTART)',
    ):
        assert line in main, line
    # every Hamiltonian is changed, and pao_eigh knows the band of the Hubbard reference
    assert main.count('p.pao_hamiltonian()') == main.count('modify_hamiltonian(p)\n') == 2
    calls = [line.strip() for line in main.splitlines() if line.strip().startswith('p.pao_eigh(')]
    assert calls == ['p.pao_eigh(bval=TOP_VALENCE_BAND)'] * 2
    plot = pd.build_plot_script(cfg)
    compile(plot, 'plot.py', 'exec')
    for function in ('plot_site_projected_bands', 'plot_jdos', 'plot_rashba_edelstein',
                     'plot_weyl_points', 'plot_z2_wannier_centres', 'plot_mirror_chern',
                     'plot_doped_transport'):  # fmt: skip
        assert 'def %s()' % function in plot


def test_restart_wraps_the_build_only(workdir):
    cfg = _config(workdir, {'properties': ['dos', 'restart']})
    main = pd.build_run_script(cfg)
    compile(main, 'main.py', 'exec')
    branch = main[main.index("    if os.path.exists(RESTART + '_0.json'):") :]
    assert branch.index('p.restart_dump') < branch.index('    p.adaptive_smearing()')
    assert '        p.gradient_and_momenta()' in branch  # inside the else branch


def test_modifiers_alone_run_nothing_extra(workdir):
    cfg = _config(workdir, {'properties': ['hamiltonian_changes'], 'efield': '0 0 0.01'})
    main = pd.build_run_script(cfg)
    assert 'def run_properties' not in main and 'pass  # no properties selected' in main


def test_spin_hall_on_atoms_is_gui_only(workdir):
    question = next(q for q in WORKFLOWS['run'].questions() if q.key == 'shc_atoms')
    assert question.cli is False
    cfg = _config(workdir, {'properties': ['spin_Hall'], 'shc_atoms': ['1']})
    main = pd.build_run_script(cfg)
    assert 'SHC_ATOMS = [0]' in main and 'shc_proj=SHC_ATOMS' in main
    plain = pd.build_run_script(_config(workdir, {'properties': ['spin_Hall']}))
    assert 'SHC_ATOMS' not in plain


def test_atoms_from_the_pw_input_without_a_save(tmp_path):
    shutil.copy(SILICON / 'scf.in', tmp_path / 'scf.in')
    form = resolve_form(
        WORKFLOWS['run'], {'properties': ['site_projected_bands']}, Context(workdir=str(tmp_path))
    )
    item = next(i for i in form.items if i.get('key') == 'site_atoms')
    assert item['choices'] == [['1', '1: Si'], ['2', '2: Si']]
    assert form.errors['site_atoms'] == 'choose at least one atom'
