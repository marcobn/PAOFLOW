"""Form resolution of the declarative questions (the GUI side of the CLI prompts)."""

from __future__ import annotations

import pytest

from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.acbn0 import manifold_row
from PAOFLOW.gen.questions.model import Context, Question
from PAOFLOW.gen.questions.resolve import interpret, label_and_hint, resolve_form


@pytest.mark.parametrize(
    'kind, raw, default, choices, expected',
    [
        ('int', '', 4, [], 4),
        ('int', ' 12 ', 4, [], 12),
        ('int', 3.0, 4, [], 3),
        ('float', '1e-3', 0.5, [], 1e-3),
        ('bool', '', True, [], True),
        ('bool', False, True, [], False),
        ('choice', 'epw', 'dfpt', ['dfpt', 'epw'], 'epw'),
        ('menu', ['dos', 'zzz', 'dos'], None, [('bands', 'B'), ('dos', 'D')], ['dos']),
        ('names', ['arpes', 'arpes', ' optics '], ['x'], ['arpes', 'optics'], ['arpes', 'optics']),
        ('text', '', 'BASIS_PS', [], 'BASIS_PS'),
    ],
)
def test_interpret_kinds(kind, raw, default, choices, expected):
    assert interpret(Question('k', 'k', kind=kind), raw, default, choices) == expected


@pytest.mark.parametrize(
    'kind, raw', [('int', '1.5'), ('float', 'x'), ('choice', 'no'), ('int', True)]
)
def test_interpret_errors(kind, raw):
    with pytest.raises(ValueError):
        interpret(Question('k', 'k', kind=kind), raw, 1, ['a'])


def test_rows_use_the_cli_parser():
    question = Question('hubbard', 'h', kind='rows', parse=manifold_row)
    rows = interpret(question, [['Si-3p', '0.5'], ['', ''], ["'O-2p'", '']], None, [])
    assert rows == [('Si-3p', 0.5), ('O-2p', 0.5)]


def test_labels_split_the_parenthesis():
    question = Question('q', 'Coarse q-grid per axis (ph.x; must divide the k-grid)')
    assert label_and_hint(question) == ('Coarse q-grid per axis', 'ph.x; must divide the k-grid')
    assert label_and_hint(Question('a', 'Is the system a metal?')) == ('Is the system a metal?', '')
    assert label_and_hint(Question('b', 'long prompt (x)', label='Short')) == ('Short', '')


def test_hidden_questions_keep_no_answer(tmp_path):
    context = Context(workdir=str(tmp_path))
    form = resolve_form(WORKFLOWS['run'], {'properties': ['dos'], 'nk': '50'}, context)
    assert 'nk' not in form.answers and form.config['nk'] == 400
    shown = {i['key']: i['visible'] for i in form.items if i['type'] == 'question'}
    assert shown['do_pdos'] and not shown['nk'] and not shown['nlo_eta']


def test_defaults_follow_earlier_answers(tmp_path):
    context = Context(workdir=str(tmp_path), overrides={'out': str(tmp_path)})
    answers = {'kgrid_axis': '4', 'workflows': ['superconductivity']}
    form = resolve_form(WORKFLOWS['elphon'], answers, context)
    items = {i['key']: i for i in form.items if i['type'] == 'question'}
    assert items['qgrid_axis']['value'] == '4' and items['sc_nq']['value'] == '16'
    assert items['sc_nk']['value'] == '32' and form.complete
    assert form.config['superconductivity']['nk'] == 32


def test_abort_message_is_a_form_error(tmp_path):
    form = resolve_form(WORKFLOWS['acbn0'], {}, Context(workdir=str(tmp_path)))
    assert form.errors['hubbard'] == 'No Hubbard manifolds entered; nothing to generate.'
    assert not form.complete
