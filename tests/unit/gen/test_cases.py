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
    """case.json, the QE inputs and their pseudopotentials, the copy, and the drivers: generated
    from the answers of a generator workflow, or shipped with a ``scripts`` case."""
    assert (case.workflow in WORKFLOWS or case.scripts) and case.title and case.summary
    assert (case.folder / 'README.md').is_file()
    library = {upf.split('/')[-1] for upf in case.pseudopotentials}
    for upf in case.pseudopotentials:
        assert (pseudopotentials_root() / upf).is_file(), upf
    inputs = [path for path in case_files(case) if path.suffix == '.in']
    if not case.scripts:
        assert inputs, 'a case of a generator workflow has its QE inputs'
    for step in case.qe:
        assert (case.folder / step.get('cwd', '.') / step['input']).is_file(), step
        assert step['program'].endswith('.x'), step
    written = copy_case(case, tmp_path)
    assert {p.name for p in written} >= library
    for path in inputs:
        text = (case.folder / path).read_text()
        match = SPECIES.search(text)
        if match and 'pseudo_dir' in text:
            used = {line.split()[2] for line in match.group(1).splitlines() if line.strip()}
            assert used <= library, (path, used - library)
            pseudo_dir = re.search(r"pseudo_dir\s*=\s*['\"]([^'\"]*)['\"]", text).group(1)
            for upf in used:  # the copy puts the UPFs where the input looks for them
                assert (tmp_path / path.parent / pseudo_dir / upf).is_file(), (path, upf)
    for step in case.steps:
        command = step['command'].split() if isinstance(step['command'], str) else step['command']
        scripts = [arg for arg in command if arg.endswith('.py')]
        for script in scripts:
            target = tmp_path / step.get('cwd', '.') / script
            assert target.is_file(), (step['label'], script)
            compile(target.read_text(), str(target), 'exec')
    if case.scripts:
        assert case.steps, 'a scripts case lists its steps'
        return
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
    assert plan[0].stdout == 'scf.out' and plan[2].action == 'setup:run' and plan[2].manual
    assert len(plan) == 4  # internal projections: no projwfc.x; then the plotting how-to
    assert plan[3].argv == ('python', 'plotting.py')
    written = copy_case(case, tmp_path)
    assert sorted(str(p) for p in written) == [
        'README.md', 'Si.pbe-tm-new-gipaw-v2.1.UPF', 'nscf.in', 'plotting.py', 'scf.in',
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


GENERATED = [case for case in CASES if not case.scripts]


@pytest.mark.parametrize('case', GENERATED, ids=[case.name for case in GENERATED])
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


def test_scripts_case_plan_and_example(tmp_path, capsys):
    """A case whose drivers come with it: no Setup step, its steps in order (with their folders),
    and ``paoflow-gen --example`` copies it and prints the steps."""
    import argparse
    import json

    from PAOFLOW.gen.cases import load_case
    from PAOFLOW.gen.paoflow_driver import start_from_example

    folder = tmp_path / 'cases' / 'demo_scripts'
    (folder / 'qe').mkdir(parents=True)
    (folder / 'qe' / 'scf.in').write_text("&control\n pseudo_dir = '../'\n/\n")
    (folder / 'model.py').write_text('print(1)\n')
    (folder / 'README.md').write_text('demo\n')
    (folder / 'case.json').write_text(json.dumps({
        'title': 'Demo', 'summary': 'A scripts case.', 'workflow': 'scripts',
        'qe': [{'program': 'pw.x', 'input': 'scf.in', 'cwd': 'qe'}],
        'steps': [{'label': 'The model', 'command': ['python', 'model.py'], 'mpi': True},
                  {'label': 'Figures', 'command': 'python plot.py', 'interactive': True}],
    }))  # fmt: skip
    case = load_case(folder)
    assert case.scripts and case.answers == {}
    plan = case_plan(case)
    assert [step.label for step in plan] == ['scf.in (pw.x)', 'The model', 'Figures']
    assert (
        plan[0].cwd == 'qe'
        and plan[0].command(2) == '(cd qe && mpirun -np 2 pw.x -in scf.in > scf.out)'
    )
    assert plan[1].argv == ('python', 'model.py') and plan[1].mpi and plan[2].interactive
    assert not any(step.action for step in plan)
    with pytest.raises(ValueError, match='steps'):
        bad = tmp_path / 'cases' / 'bad'
        bad.mkdir()
        (bad / 'case.json').write_text(
            json.dumps({'title': 't', 'summary': 's', 'workflow': 'scripts'})
        )
        load_case(bad)

    import PAOFLOW.gen.cases as cases_module

    original = cases_module.cases_root
    cases_module.cases_root = lambda: tmp_path / 'cases'
    try:
        (tmp_path / 'cases' / 'bad' / 'case.json').unlink()
        out = tmp_path / 'copy'
        args = argparse.Namespace(example='demo_scripts', workdir=str(out), force=False)
        assert start_from_example(args) == 0
    finally:
        cases_module.cases_root = original
    printed = capsys.readouterr().out
    assert (out / 'model.py').is_file() and (out / 'qe' / 'scf.in').is_file()
    assert '1. scf.in (pw.x)' in printed and '2. The model: mpirun -np 4 python model.py' in printed


def _page_case(page):
    """The case named by the first tip of a Learn page (``paoflow-gen --example <case>``)."""
    import json

    text = page.read_text()
    if page.suffix == '.ipynb':
        cells = json.loads(text)['cells']
        text = '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')
    tip = re.search(r':::\{tip\}\n(.*?)\n:::', text, re.S)
    found = re.search(r'paoflow-gen --example (\w+)', tip.group(1)) if tip else None
    return found.group(1) if found else None


def test_case_docs_match_the_pages():
    """``docs`` of each case lists exactly the tutorials and how-tos whose tip names the case."""
    docs = cases_root().parents[1] / 'docs'
    pages = sorted((docs / 'learn').glob('tutorials/tutorial*.ipynb'))
    pages += [p for p in sorted((docs / 'learn').glob('howtos/*.md')) if p.name != 'index.md']
    named = {str(page.relative_to(docs)): _page_case(page) for page in pages}
    missing_tip = sorted(page for page, case in named.items() if case is None)
    assert not missing_tip, 'pages without a tip naming their case: %s' % missing_tip
    listed = {page: case.name for case in CASES for page in case.docs}
    unknown = sorted(page for page in listed if page not in named)
    assert not unknown, 'case.json docs entries that are no Learn page: %s' % unknown
    assert listed == named, sorted(set(listed.items()) ^ set(named.items()))
