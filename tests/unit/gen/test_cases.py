"""The example cases of ``examples/cases``: files, pseudopotentials, answers and the copy."""

from __future__ import annotations

import re

import pytest

from PAOFLOW.gen.cases import (
    case_files,
    case_plan,
    cases_root,
    copy_case,
    find_case,
    list_cases,
    pseudopotentials_root,
)
from PAOFLOW.gen.outputs import workflow_outputs
from PAOFLOW.gen.questions import WORKFLOWS
from PAOFLOW.gen.questions.model import Context
from PAOFLOW.gen.questions.resolve import resolve_form

pytestmark = pytest.mark.unit

CASES = list_cases()
SPECIES = re.compile(r'ATOMIC_SPECIES[^\n]*\n((?:[ \t]*\S+[ \t]+\S+[ \t]+\S+[ \t]*\n)+)')


def test_the_cases_are_there():
    assert cases_root() is not None and pseudopotentials_root() is not None
    names = {case.name for case in CASES}
    assert len(CASES) >= 24 and {'si_bands_transport', 'alh_junction_landauer', 'si_acbn0'} <= names


@pytest.mark.parametrize('case', CASES, ids=[case.name for case in CASES])
def test_case_is_consistent(case, tmp_path):
    assert case.workflow in WORKFLOWS and case.title and case.summary
    assert (case.folder / 'README.md').is_file()
    library = {upf.split('/')[-1] for upf in case.pseudopotentials}
    for upf in case.pseudopotentials:
        assert (pseudopotentials_root() / upf).is_file(), upf
    inputs = [path for path in case_files(case) if path.suffix == '.in']
    assert inputs, 'a case has its QE inputs'
    for step in case.qe:
        assert (case.folder / step['input']).is_file() and step['program'] in ('pw.x', 'projwfc.x')
    for path in inputs:
        text = (case.folder / path).read_text()
        match = SPECIES.search(text)
        if match:
            used = {line.split()[2] for line in match.group(1).splitlines() if line.strip()}
            assert used <= library, (path, used - library)
            assert re.search(r"pseudo_dir\s*=\s*'\./'", text), path
    written = copy_case(case, tmp_path)
    assert {p.name for p in written} >= library
    form = resolve_form(WORKFLOWS[case.workflow], case.answers, Context(workdir=str(tmp_path)))
    assert form.complete and not form.errors, form.errors
    unknown = set(case.answers) - {item['key'] for item in form.items if 'key' in item}
    assert not unknown, 'answers without a question: %s' % sorted(unknown)
    files = workflow_outputs(case.workflow, form.config)
    for generated in files:
        if generated.path.endswith('.py'):
            compile(generated.content(), generated.path, 'exec')


def test_plan_and_copy_conflicts(tmp_path):
    case = find_case('si_bands_transport')
    plan = case_plan(case)
    assert [step.argv for step in plan[:2]] == [
        ('pw.x', '-in', 'scf.in'),
        ('pw.x', '-in', 'nscf.in'),
    ]
    assert plan[0].stdout == 'scf.out' and plan[-1].action == 'setup:run' and plan[-1].manual
    assert len(plan) == 3  # internal projections: no projwfc.x
    written = copy_case(case, tmp_path)
    assert sorted(str(p) for p in written) == [
        'README.md', 'Si.pbe-tm-new-gipaw-v2.1.UPF', 'nscf.in', 'scf.in',
    ]  # fmt: skip
    assert not (tmp_path / 'silicon.save').exists() and not (tmp_path / 'case.json').exists()
    assert copy_case(case, tmp_path) == []  # identical files are left alone
    (tmp_path / 'scf.in').write_text('edited')
    with pytest.raises(FileExistsError) as error:
        copy_case(case, tmp_path)
    assert error.value.args[1] == ['scf.in']
    assert copy_case(case, tmp_path, overwrite=True) == [type(written[0])('scf.in')]
    with pytest.raises(KeyError, match='si_bands_transport'):
        find_case('no_such_case')


@pytest.mark.parametrize('case', CASES, ids=[case.name for case in CASES])
def test_cases_use_the_internal_projections(case):
    assert case.answers.get('projector', 'internal') == 'internal'
    assert all(step['program'] == 'pw.x' for step in case.qe)
    assert not (case.folder / 'proj.in').exists()


def test_projwfc_chosen_in_the_form_writes_proj_in(tmp_path):
    """projwfc stays an option: the files then hold proj.in and the plan runs projwfc.x first."""
    from PAOFLOW.gen.outputs import run_plan

    case = find_case('si_bands_transport')
    copy_case(case, tmp_path)
    answers = dict(case.answers, projector='projwfc')
    form = resolve_form(WORKFLOWS['run'], answers, Context(workdir=str(tmp_path)))
    files = workflow_outputs('run', form.config)
    proj = next(f for f in files if f.path == 'proj.in').content()
    assert "prefix = 'silicon'" in proj and 'lwrite_overlaps = .true.' in proj
    assert "outdir = './'" in proj
    plan = run_plan('run', form.config, files)
    assert plan[0].argv == ('projwfc.x', '-in', 'proj.in') and plan[1].argv[:2] == (
        'python',
        'main.py',
    )
    internal = workflow_outputs(
        'run', resolve_form(WORKFLOWS['run'], case.answers, Context(workdir=str(tmp_path))).config
    )
    assert 'proj.in' not in [f.path for f in internal]
