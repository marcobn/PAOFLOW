"""Parity of the declarative questions with the frozen hand-written prompts.

Every scenario answers the prompts through an oracle (prompt fragment -> answer, blank = default),
runs the frozen functions of ``legacy_collect`` and the new :mod:`PAOFLOW.gen.questions` path on the
same answers, and requires the same prompts, the same printed text, the same configuration (keys in
the same order) and byte-identical generated files.
"""

from __future__ import annotations

import copy
import sys
import types

import legacy_collect as legacy
import pytest

from PAOFLOW.gen import elphon_system as es
from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen import sscha_driver as sd
from PAOFLOW.gen.questions import collect

SI_SCF = """&control
  calculation = 'scf'
  prefix = 'si'
  pseudo_dir = './pseudo'
/
&system
  ibrav = 2, celldm(1) = 10.26, nat = 2, ntyp = 1, ecutwfc = 30
/
&electrons
/
ATOMIC_SPECIES
Si 28.0855 Si.upf
ATOMIC_POSITIONS crystal
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS automatic
6 6 6 0 0 0
"""


class Oracle:
    """``_input`` replacement: the first rule whose fragment is in the prompt answers it.

    Choice questions print their text and then read ``Choice [default]:``, so a rule that matches
    no prompt is tried on the text printed since the previous input.  A list value is consumed one
    answer per prompt (the last one repeats), for re-prompts and for the ACBN0 manifold rows.
    """

    def __init__(self, rules: dict) -> None:
        self.rules = [
            (fragment, list(v) if isinstance(v, list) else [v]) for fragment, v in rules.items()
        ]
        self.prompts: list[str] = []
        self.printed = ''
        self.used: set[str] = set()

    def write(self, text: str) -> None:
        self.printed += text

    def _answer(self, fragment: str, values: list) -> str:
        self.used.add(fragment)
        return values.pop(0) if len(values) > 1 else values[0]

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) > 500:
            raise RuntimeError('prompt loop: %r' % prompt)
        context, self.printed = self.printed, ''
        for text in (prompt, context):
            for fragment, values in self.rules:
                if fragment in text:
                    return self._answer(fragment, values)
        return ''


class Tee:
    """``sys.stdout`` wrapper that also feeds the printed text to the oracle."""

    def __init__(self, stream, oracle: Oracle) -> None:
        self.stream, self.oracle = stream, oracle

    def write(self, text: str) -> int:
        self.oracle.write(text)
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()


def _render(builder, cfg):
    try:
        return builder(cfg)
    except Exception as error:  # both paths must fail the same way
        return 'raised %s: %s' % (type(error).__name__, error)


#: Intended changes of the prompts after the refactor: (frozen legacy text, current text).  The
#: legacy output is updated with them before the comparison, so everything else stays exact.
LEGACY_UPDATES = (
    # 10 November 2026: the finite-difference coupling source of the electron-phonon system.
    (
        "Coupling source (dfpt = PAOFLOW's vertex from ph.x dvscf, norm-conserving; "
        'epw = EPW epbwrite, any pseudopotential)\n  1: dfpt (default)\n  2: epw\n',
        "Coupling source (dfpt = PAOFLOW's vertex from ph.x dvscf, norm-conserving; "
        'epw = EPW epbwrite, any pseudopotential; fd = finite differences of a supercell, '
        'pw.x only)\n  1: dfpt (default)\n  2: epw\n  3: fd\n',
    ),
)


def _updated(text: str) -> str:
    for old, new in LEGACY_UPDATES:
        text = text.replace(old, new)
    return text


def _run_both(monkeypatch, capsys, rules, legacy_call, new_call):
    results = []
    for call in (legacy_call, new_call):
        oracle = Oracle(copy.deepcopy(rules))
        monkeypatch.setattr(pd, '_input', oracle)
        monkeypatch.setattr(sys, 'stdout', Tee(sys.stdout, oracle))
        cfg = call()
        monkeypatch.undo()
        printed = capsys.readouterr().out
        unused = [fragment for fragment, _ in oracle.rules if fragment not in oracle.used]
        assert not unused, 'rules that matched no prompt: %s' % unused
        results.append((cfg, oracle.prompts, printed))
    (old_cfg, old_prompts, old_out), (new_cfg, new_prompts, new_out) = results
    assert new_prompts == [_updated(prompt) for prompt in old_prompts]
    assert new_out == _updated(old_out)
    if old_cfg is None:
        assert new_cfg is None
        return None
    assert list(new_cfg.items()) == list(old_cfg.items())
    assert repr(new_cfg) == repr(old_cfg)  # also 1 vs 1.0 and tuple vs list inside values
    return new_cfg


def _args(**kw):
    values = dict(savedir=None, prefix=None, upf=None, out=None)
    values.update(kw)
    return types.SimpleNamespace(**values)


@pytest.fixture()
def qe_dir(tmp_path):
    """A working directory with a save, a UPF in it, scf.in and a V-cutoff header."""
    save = tmp_path / 'si.save'
    save.mkdir()
    (save / 'Si.upf').write_text('<UPF version="2.0.1"></UPF>\n')
    (tmp_path / 'scf.in').write_text(SI_SCF)
    (tmp_path / 'Si.scf.in').write_text(
        '! Suggested eACBN0 intersite V cutoff: 3.1 Angstrom\n' + SI_SCF
    )
    return tmp_path


# ------------------------------------------------------------------------------------- run
RUN_SCENARIOS = {
    'all_properties': {'Enter a comma': '1 2 3 4 5 6 7 8 9 10 11 12'},
    'nonlinear_2d_projwfc': {
        'Enter a comma': '12',
        '2D system': 'y',
        'Projector source': 'projwfc',
        'tensor components': 'xyz,yyy',
        'responses': 'shg',
    },
    'molecule': {
        'Enter a comma': '1, 12',
        'isolated molecule': 'y',
        'coupled response of the molecule': 'hartree',
    },
    'coupled': {
        'Enter a comma': '2 12',
        'coupled optical response': 'y',
        'kernel levels': 'none,hartree',
        'SHG photon energies': '',
        'Projected DOS': 'n',
    },
    'metal_coupled': {  # 12 November 2026: Fermi-surface terms and the Drude channel
        'Enter a comma': '12',
        'metal or semimetal': 'y',
        'electronic temperature': '500',
        'intraband relaxation time': '15',
        'coherence damping in units': '2',
        'coupled optical response': 'y',
        'SHG photon energies': '',
    },
    'optical_only': {'Enter a comma': '10', 'Smearing': 'None', 'double-grid': 'n'},
    'emissivity_window': {
        'Enter a comma': '11 2 5',
        '  EMIN': '-3',
        '  EMAX': '2.5',
        '  NE (number': '501',
        'Spin-orbit': 'y',
        'npool': '4',
        'Basis configuration': 'extended',
        'plotting script': 'n',
    },
    'nothing_selected': {'Enter a comma': 'x 99'},
    'bands_ibrav': {'Enter a comma': '1', 'ibrav (0': '2', 'Band path k-points': '250'},
}


@pytest.mark.parametrize('name', sorted(RUN_SCENARIOS))
def test_run_workflow_parity(name, qe_dir, monkeypatch, capsys):
    rules = RUN_SCENARIOS[name]
    args = _args()
    workdir = str(qe_dir)

    def old():
        common = legacy.collect_common(args, workdir)
        return legacy.collect_run(common)

    cfg = _run_both(monkeypatch, capsys, rules, old, lambda: collect('run', args, workdir))
    if cfg is not None:
        cfg_wrapped = _run_both(
            monkeypatch,
            capsys,
            rules,
            old,
            lambda: pd.collect_run(pd.collect_common(args, workdir)),
        )
        assert cfg_wrapped == cfg
        assert _render(pd.build_run_script, cfg).startswith('#!/usr/bin/env python3')
        _render(pd.build_plot_script, cfg)


def test_run_workflow_overrides(qe_dir, monkeypatch, capsys):
    args = _args(savedir='x.save', prefix='x', upf='A.upf, B.upf')
    workdir = str(qe_dir)
    rules = {'Enter a comma': '1 2'}
    cfg = _run_both(
        monkeypatch,
        capsys,
        rules,
        lambda: legacy.collect_run(legacy.collect_common(args, workdir)),
        lambda: collect('run', args, workdir),
    )
    assert cfg['savedir'] == 'x.save' and cfg['upfs'] == ['A.upf', 'B.upf']


def test_nonlinear_block_parity(monkeypatch, capsys):
    rules = {
        'isolated molecule': 'n',
        'dense k-grid': '30',
        'coupled optical response': 'y',
        'SHG photon energies': '0.1,1.0',
    }
    seed = {'is_2d': False, 'projector': 'internal'}
    _run_both(
        monkeypatch,
        capsys,
        rules,
        lambda: legacy.collect_nonlinear(dict(seed)),
        lambda: pd.collect_nonlinear(dict(seed)),
    )


# ----------------------------------------------------------------------------------- acbn0
ACBN0_SCENARIOS = {
    'on_site': {'manifold': ['Si-3p 0.5', "'O-2p'", 'Ti-3d abc', '']},
    'intersite_internal': {
        'Include intersite V': 'y',
        'manifold': ['Si-3p 0.5', ''],
        'Initial intersite V': '0.7',
    },
    'intersite_projwfc': {
        'Projector source': 'projwfc',
        'Include intersite V': 'y',
        'manifold': ['Si-3p 0.5', ''],
        'neighbour cutoff': '2.9',
        'MPI command for python': 'mpirun -np 2',
    },
    'no_manifold': {'manifold': ''},
}


@pytest.mark.parametrize('name', sorted(ACBN0_SCENARIOS))
def test_acbn0_workflow_parity(name, qe_dir, monkeypatch, capsys):
    args = _args()
    workdir = str(qe_dir)
    cfg = _run_both(
        monkeypatch,
        capsys,
        ACBN0_SCENARIOS[name],
        lambda: legacy.collect_acbn0(legacy.collect_common(args, workdir)),
        lambda: collect('acbn0', args, workdir),
    )
    if cfg is not None:
        assert _render(pd.build_acbn0_script, cfg) == _render(pd.build_acbn0_script, dict(cfg))
        _render(pd.build_acbn0_plot_script, cfg)


# ---------------------------------------------------------------------------------- phonon
PHONON_SCENARIOS = {
    'defaults': {},
    'no_born': {'Compute Born charges': 'n'},
    'vib_no_emissivity': {'reststrahlen (phonon) emissivity': 'n', 'Frequency units': 'THz'},
    'qha_three_volumes': {
        'quasi-harmonic': 'y',
        'Number of sampled volumes': '3',
        'Maximum temperature': '800',
    },
    'qha_five_volumes': {'quasi-harmonic': 'y', 'Equation of state': 'murnaghan'},
    'raman_static_no_laser': {'Raman workflow': 'static'},
    'raman_static_laser': {'Raman workflow': 'static', 'Laser wavelength (nm': '532'},
    'raman_resonance': {'Raman workflow': 'resonance', 'Laser wavelength(s)': '488, 532'},
    'raman_all_expression': {
        'Raman workflow': 'all',
        'Laser wavelength(s)': '[n for n in range(450, 470, 5)]',
        'Born/epsilon method': 'field',
    },
}


@pytest.mark.parametrize('name', sorted(PHONON_SCENARIOS))
def test_phonon_workflow_parity(name, qe_dir, monkeypatch, capsys):
    args = _args()
    workdir = str(qe_dir)
    cfg = _run_both(
        monkeypatch,
        capsys,
        PHONON_SCENARIOS[name],
        lambda: legacy.collect_phonon(legacy.collect_common(args, workdir)),
        lambda: collect('phonon', args, workdir),
    )
    _render(pd.build_phonon_script, cfg)
    _render(pd.build_phonon_plot_script, cfg)
    if cfg['raman']:
        _render(pd.build_raman_script, cfg)
        _render(pd.build_raman_plot_script, cfg)


# ---------------------------------------------------------------------------------- elphon
ELPHON_SCENARIOS = {
    'metal_superconductivity': {
        'Is the system a metal': 'y',
        'Coarse k-grid': '6',
        'Coarse q-grid': ['4', '3'],
        'Bands in the nscf': '16',
        'Valence electrons': '8',
        'Workflows': 'superconductivity, arpes',
        'dense phonon q-grid per axis (NQ_DENSE)': '6',
        'NK_DENSE': ['13', '12'],
        'Migdal-Eliashberg temperatures': '1, 9, 3',
        'Fermi-surface pair coupling': 'y',
    },
    'semiconductor_defaults': {'Bands in the nscf': '12', 'Valence electrons': '8'},
    'semiconductor_emissivity': {
        'Valence electrons': '8',
        'thermal emissivity of a slab': 'y',
        'Target indirect gap for a scissor': '1.17',
        'Varshni': '4.73e-4, 636',
        'target static dielectric constant': '11.94',
        'Constant refractive index': '3.4',
        'Emission angles': '0, 45',
        'Optics nscf: Monkhorst-Pack': '10',
    },
    'semiconductor_mobility_eacbn0': {
        'Workflows': 'transport',
        'nscf in the same PAO basis (blank = PBE electrons)': 'eacbn0.save',
        'Varshni': '4.73e-4, 636',
    },
    'epw_source': {
        'Coupling source': 'epw',
        'exclude_bands': ['9-x', '1:5'],
        'Workflows': ['foo, optics', 'optics'],
    },
    'metal_resistivity_interband': {
        'Is the system a metal': 'y',
        'Workflows': 'transport, thermal, phonons, coupling',
        'Interband dielectric function': 'y',
        'ZG supercell per axis': '4',
    },
    'metal_everything': {
        'Is the system a metal': 'y',
        'Workflows': 'phonons coupling superconductivity optics transport arpes thermal',
        'Bands in the nscf': '30',
        'transport Eliashberg function': 'y',
        'thermal emissivity of a slab': 'y',
        'Direct dielectric function': 'n',
    },
    'missing_template': {'pw.x input of the system': 'missing.in'},
}


@pytest.mark.parametrize('name', sorted(ELPHON_SCENARIOS))
def test_elphon_workflow_parity(name, tmp_path, monkeypatch, capsys):
    (tmp_path / 'scf.in').write_text(SI_SCF)
    args = _args(out=str(tmp_path / 'system'))
    workdir = str(tmp_path)
    cfg = _run_both(
        monkeypatch,
        capsys,
        ELPHON_SCENARIOS[name],
        lambda: legacy.collect_elphon_system(args, workdir),
        lambda: collect('elphon', args, workdir),
    )
    assert _render(es.render_system_py, cfg)
    _render(es.qe_inputs, cfg)
    assert es.workflow_files(cfg)


def test_elphon_without_template_and_prefix_override(tmp_path, monkeypatch, capsys):
    args = _args(prefix='lead', out=None)
    cfg = _run_both(
        monkeypatch,
        capsys,
        {'Workflows': 'arpes'},
        lambda: legacy.collect_elphon_system(args, str(tmp_path)),
        lambda: collect('elphon', args, str(tmp_path)),
    )
    assert cfg['prefix'] == 'lead' and cfg['pw_template_text'] is None


# ----------------------------------------------------------------------------------- sscha
SSCHA_SCENARIOS = {
    'local_ionic': {
        'Temperatures (K': '1000, 2000',
        'q-grid of the harmonic': '6',
        'k-grid of the supercell': '6 6 6',
        'Electrons of the force runs': 'ionic',
        'Configurations per population': '100, 200',
        'pw.x command': 'pw.x -nk 4',
    },
    'slurm_fixed': {
        'Electrons of the force runs': 'fixed',
        '  Electronic temperature': '500',
        'Launcher of the pw.x runs': 'slurm',
        '  partition': 'genoa',
        '  MPI tasks per configuration': '64',
        '  account': 'proj',
        '  shell line': 'source ~/env.sh',
    },
    'manual_defaults': {'Launcher of the pw.x runs': 'manual'},
}


@pytest.mark.parametrize('name', sorted(SSCHA_SCENARIOS))
def test_sscha_workflow_parity(name, qe_dir, monkeypatch, capsys):
    harmonic = qe_dir / 'harmonic'
    harmonic.mkdir()
    (harmonic / 'si.dyn0').write_text('2 2 2\n1\n')
    args = _args()
    cfg = _run_both(
        monkeypatch,
        capsys,
        SSCHA_SCENARIOS[name],
        lambda: legacy.collect_sscha(args, str(qe_dir)),
        lambda: sd.collect_sscha(args, str(qe_dir)),
    )
    assert _render(sd.build_sscha_script, cfg).startswith('#!')


# ------------------------------------------------------------------------ builder coverage
#: Keys the builders read (with a default) that no question sets, CLI or GUI-only.  A new builder
#: key must get a question in PAOFLOW.gen.questions (``cli=False`` for a GUI-only refinement) or
#: be listed here on purpose.
BUILDER_ONLY_KEYS: dict[str, set[str]] = {
    'run': set(),
    'acbn0': set(),
    'phonon': set(),
    'elphon': set(),
    'sscha': set(),
}


def _builder_keys(functions) -> set[str]:
    import inspect
    import re

    keys: set[str] = set()
    for function in functions:
        source = inspect.getsource(function)
        keys |= set(re.findall(r"cfg\['([A-Za-z_0-9]+)'\]", source))
        keys |= set(re.findall(r"cfg\.get\('([A-Za-z_0-9]+)'", source))
    return keys


def _builders(name: str) -> list:
    common = [pd._projection_lines, pd._uses_projwfc]
    return {
        'run': [
            pd.build_run_script,
            pd.build_plot_script,
            pd._emit_standard_properties,
            pd._nonlinear_constant_lines,
            pd._nonlinear_run_lines,
            pd._band_path_constant_lines,
            pd._wants_explicit_band_path,
            pd._nonlinear_plot_entries,
            *common,
        ],
        'acbn0': [pd.build_acbn0_script, pd.build_acbn0_plot_script, *common],
        'phonon': [
            pd.build_phonon_script,
            pd.build_phonon_plot_script,
            pd.build_raman_script,
            pd.build_raman_plot_script,
            pd._pp_dir_line,
            pd._hubbard_file_line,
        ],
        'elphon': [es.render_system_py, es.render_run_qe, es.qe_inputs, es.workflow_files],
        'sscha': [sd.build_sscha_script],
    }[name]


#: Scenarios of questions added after the refactor: the frozen prompts do not have them, so they
#: take part in the builder-key coverage below, not in the parity runs.
ELPHON_NEW_SCENARIOS = {
    'fd_source': {
        'Coupling source': 'fd',
        'Is the system a metal': 'y',
        'Displacements': 'random',
        'Locality cutoff': '9',
    },
}

SCENARIOS = {
    'run': RUN_SCENARIOS,
    'acbn0': ACBN0_SCENARIOS,
    'phonon': PHONON_SCENARIOS,
    'elphon': {**ELPHON_SCENARIOS, **ELPHON_NEW_SCENARIOS},
    'sscha': SSCHA_SCENARIOS,
}


@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_every_builder_key_has_a_question(name, qe_dir, monkeypatch, capsys):
    harmonic = qe_dir / 'harmonic'
    harmonic.mkdir()
    (harmonic / 'si.dyn0').write_text('2 2 2\n1\n')
    produced: set[str] = set()
    for rules in SCENARIOS[name].values():
        oracle = Oracle(copy.deepcopy(rules))
        monkeypatch.setattr(pd, '_input', oracle)
        monkeypatch.setattr(sys, 'stdout', Tee(sys.stdout, oracle))
        cfg = collect(name, _args(out=str(qe_dir / 'system')), str(qe_dir))
        monkeypatch.undo()
        capsys.readouterr()
        produced |= set(cfg or {})
    from PAOFLOW.gen.questions import WORKFLOWS

    gui_only = {q.key for q in WORKFLOWS[name].questions() if not q.cli}
    assert not gui_only & produced, 'the CLI produced GUI-only keys'
    missing = _builder_keys(_builders(name)) - produced - gui_only - BUILDER_ONLY_KEYS[name]
    assert not missing, 'builder keys without a question: %s' % sorted(missing)
    stale = BUILDER_ONLY_KEYS[name] & produced
    assert not stale, 'listed as builder-only but produced: %s' % sorted(stale)
