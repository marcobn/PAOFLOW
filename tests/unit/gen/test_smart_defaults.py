"""Defaults read from the QE save and the pw.x template (decision D4: CLI and GUI alike)."""

from __future__ import annotations

import os
import shutil
import types

import pytest

from PAOFLOW.gen import paoflow_driver as pd
from PAOFLOW.gen.questions import collect
from PAOFLOW.gen.system_summary import is_two_dimensional, read_pw_input_summary, read_upf_header

HERE = os.path.dirname(os.path.abspath(__file__))
SNTE_XML = os.path.join(
    HERE,
    '..',
    '..',
    '..',
    'examples',
    'qe_examples',
    'example17',
    'SnTe.save',
    'data-file-schema.xml',
)

UPF_V2 = '<UPF version="2.0.1">\n<PP_HEADER element="{el}" pseudo_type="{kind}" relativistic="scalar"\n z_valence="{z}" functional="PBE"/>\n</UPF>\n'

TEMPLATE = """&control
  prefix = 'nb'
  pseudo_dir = './pp'
/
&system
  ibrav = 3, celldm(1) = 6.2, nat = 1, ntyp = 1, nbnd = 20,
  occupations = 'smearing', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
Nb 92.906 Nb.upf
ATOMIC_POSITIONS crystal
Nb 0.0 0.0 0.0
K_POINTS automatic
8 8 8 0 0 0
"""


def _args(**kw):
    values = dict(savedir=None, prefix=None, upf=None, out=None)
    values.update(kw)
    return types.SimpleNamespace(**values)


def _blank_answers(monkeypatch, menu='1'):
    prompts = []

    def answer(prompt):
        prompts.append(prompt)
        return menu if prompt.startswith('Enter a comma') else ''

    monkeypatch.setattr(pd, '_input', answer)
    return prompts


@pytest.mark.skipif(not os.path.isfile(SNTE_XML), reason='example17 save not present')
def test_run_defaults_follow_the_save(tmp_path, monkeypatch):
    save = tmp_path / 'SnTe.save'
    save.mkdir()
    shutil.copy(SNTE_XML, save / 'data-file-schema.xml')
    prompts = _blank_answers(monkeypatch)
    cfg = collect('run', _args(), str(tmp_path))
    assert cfg['is_2d'] is True and cfg['spin_orbit'] is True
    assert 'Is this a 2D system (slab with vacuum along c)? [Y/n]: ' in prompts


def test_run_defaults_without_a_save(tmp_path, monkeypatch):
    _blank_answers(monkeypatch)
    cfg = collect('run', _args(), str(tmp_path))
    assert cfg['is_2d'] is False and cfg['spin_orbit'] is False


def test_elphon_defaults_follow_the_template(tmp_path, monkeypatch):
    (tmp_path / 'pp').mkdir()
    (tmp_path / 'pp' / 'Nb.upf').write_text(UPF_V2.format(el='Nb', kind='PAW', z='13.0'))
    (tmp_path / 'scf.in').write_text(TEMPLATE)
    _blank_answers(monkeypatch)
    cfg = collect('elphon', _args(out=str(tmp_path / 'sys')), str(tmp_path))
    assert cfg['metal'] is True and cfg['nbnd'] == 20 and cfg['nelec'] == 13
    assert cfg['coupling_source'] == 'epw' and cfg['irreducible'] is False
    assert cfg['workflows'] == ['superconductivity', 'transport', 'arpes']


def test_upf_headers_v1_and_v2(tmp_path):
    v2 = tmp_path / 'a.upf'
    v2.write_text(UPF_V2.format(el='Si', kind='NC', z='4.0'))
    v1 = tmp_path / 'b.upf'
    v1.write_text(
        '<PP_INFO>\n</PP_INFO>\n<PP_HEADER>\n   0                   Version Number\n'
        '  Ti                   Element\n   US                  Ultrasoft pseudopotential\n'
        '    T                  Nonlinear Core Correction\n SLA PW PBX PBC    PBE  Exchange-Correlation\n'
        '   12.00000000000      Z valence\n</PP_HEADER>\n'
    )
    assert read_upf_header(str(v2))['z_valence'] == 4.0
    header = read_upf_header(str(v1))
    assert (
        header['element'] == 'Ti' and header['pseudo_type'] == 'US' and header['z_valence'] == 12.0
    )
    assert read_upf_header(str(tmp_path / 'missing.upf')) is None


def test_pw_summary_counts_valence_electrons(tmp_path):
    (tmp_path / 'pp').mkdir()
    (tmp_path / 'pp' / 'Nb.upf').write_text(UPF_V2.format(el='Nb', kind='NC', z='13.0'))
    (tmp_path / 'scf.in').write_text(TEMPLATE.replace('nat = 1', 'nat = 2') + '')
    summary = read_pw_input_summary(str(tmp_path / 'scf.in'))
    assert summary['prefix'] == 'nb' and summary['metal'] and summary['nbnd'] == 20
    assert summary['nelec'] == 13.0 and not summary['augmented']


def test_two_dimensional_detection():
    bulk = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]]
    slab = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 30.0]]
    atoms = [[0, 0, 10.0], [2.5, 2.5, 12.0]]
    assert is_two_dimensional(slab, atoms) and not is_two_dimensional(bulk, [[0, 0, 0]])
    # atoms spread over the whole height: a long bulk cell, not a slab
    spread = [[0, 0, z] for z in range(0, 30, 3)]
    assert not is_two_dimensional(slab, spread)
