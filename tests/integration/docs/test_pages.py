"""Re-run the tutorial notebooks and how-tos of the website on their base cases.

Every tutorial and how-to is built on one case of ``examples/cases``, which its opening tip names
(``paoflow-gen --example <case>``). The docs build does not execute the pages
(``nb_execution_mode = 'off'``), so this test is what notices that a page's code no longer runs,
or that a number the page quotes has changed: each page ends with asserts of those numbers (a
``remove-cell`` cell in a notebook, the end of a how-to's figures script).

- A tutorial makes the PAOFLOW calls of its case's driver: the ``main.py`` that ``paoflow-gen``
  and Studio write from the answers of a generator case, or the ``main.py`` of a scripts case.
  :func:`test_tutorial_follows_its_case` compares them without running anything.
- A how-to shows one workflow script of its case with ``literalinclude`` (``main.py``, or
  ``<name>.py`` when several pages share the case). It runs as ``python main.py > main.log`` in a
  copy of the case made by ``paoflow-gen --example <case> --with-data`` (on a generator case after
  the generated ``main.py``), followed there by the page's figures script,
  ``docs/learn/howtos/<page>/figures.py``, which draws the figures and asserts the numbers.

The data of a case (``case.json`` ``data``) come from the release ``case-data`` through the
download cache of :func:`PAOFLOW.gen.cases.case_data_archive`; an archive placed in that cache
(``~/.cache/paoflow/case-data``) is used without a download, for data not yet published.

Each page runs in a scratch directory, so the committed outputs and figures are never touched.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
PAGES = REPO / 'docs' / 'learn'
# notebooks that are not run: stubs, and pages whose case data are not published yet
UNWRITTEN = {'tutorial03'}
# how-tos whose workflow exceeds the run-time budget of the learn skill (5 min on 4 cores): the
# data of their case hold what the workflow writes, so the figures script runs on it on every pull request, and
# the <page>-workflow case that runs the workflow is marked slow
PRECOMPUTED = {'twisted-bilayer'}
TIMEOUT = 1800  # s; catches a hung page, far above the budget
HIDDEN = {'hide-input', 'remove-cell'}  # plotting and asserts: not part of the workflow
TIP = re.compile(r':::\{tip\}\n(.*?)\n:::', re.S)
EXAMPLE = re.compile(r'paoflow-gen --example (\w+)')
INCLUDE = re.compile(r'^```\{literalinclude\}\s+(\S+)', re.M)


def markdown(page: Path) -> str:
    """The Markdown of a page: the file, or the Markdown cells of a notebook."""
    if page.suffix != '.ipynb':
        return page.read_text()
    cells = json.loads(page.read_text())['cells']
    return '\n'.join(''.join(c['source']) for c in cells if c['cell_type'] == 'markdown')


def base_case(page: Path):
    """The case named by the first tip of the page."""
    from PAOFLOW.gen.cases import find_case

    tip = TIP.search(markdown(page))
    found = EXAMPLE.search(tip.group(1)) if tip else None
    if found is None:
        pytest.fail(f'{page}: the first tip names no case (paoflow-gen --example <case>)')
    return find_case(found.group(1))


def driver(case) -> str:
    """Source of the case's driver: generated from its answers, or its own ``main.py``."""
    if case.scripts:
        return (case.folder / 'main.py').read_text()
    from PAOFLOW.gen.outputs import workflow_outputs
    from PAOFLOW.gen.questions import WORKFLOWS
    from PAOFLOW.gen.questions.model import Context
    from PAOFLOW.gen.questions.resolve import resolve_form

    form = resolve_form(WORKFLOWS[case.workflow], case.answers, Context(workdir=str(case.folder)))
    return next(
        f for f in workflow_outputs(case.workflow, form.config) if f.path == 'main.py'
    ).content()


def _constants(tree: ast.Module) -> dict[str, object]:
    """Module-level ``NAME = <literal>`` and ``A, B = <literals>`` assignments."""
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            found[target.id] = value
        elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
            found.update({t.id: v for t, v in zip(target.elts, value) if isinstance(t, ast.Name)})
    return found


def _value(node: ast.expr, constants: dict[str, object]) -> object:
    """The literal value of an argument, with module constants resolved; None if not literal."""
    try:
        return ast.literal_eval(node)
    except ValueError:
        pass
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        try:
            return constants[node.value.id][ast.literal_eval(node.slice)]
        except (KeyError, ValueError, IndexError, TypeError):
            return None
    return None


def paoflow_calls(source: str) -> list[tuple[str, dict[str, object]]]:
    """The PAOFLOW calls of some code in source order: the constructor and every method, with
    their literal arguments (positional ones keyed by position). Arguments that are not literal
    (paths built at run time, variables) are kept as None."""
    from PAOFLOW.PAOFLOW import PAOFLOW

    names = {m for m in dir(PAOFLOW) if not m.startswith('_')} | {'PAOFLOW'}
    tree = ast.parse(source)
    constants = _constants(tree)
    calls = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in names
        ),
        key=lambda c: (c.lineno, c.col_offset),
    )
    return [
        (
            call.func.attr,
            {
                **{i: _value(a, constants) for i, a in enumerate(call.args)},
                **{k.arg: _value(k.value, constants) for k in call.keywords},
            },
        )
        for call in calls
    ]


def compare_calls(expected: str, found: str) -> list[str]:
    """Differences between the PAOFLOW calls of ``found`` and those of the driver ``expected``.

    Methods must come in the same order with the same literal arguments; an argument that the
    driver does not give as a literal (``savedir``, ``workpath``: paths of the run) is not
    compared, so the page may give it as it needs or leave it out.
    """
    driver_calls, page_calls = paoflow_calls(expected), paoflow_calls(found)
    problems = []
    if [m for m, _ in driver_calls] != [m for m, _ in page_calls]:
        problems.append(
            'methods differ:\n  driver: %s\n  page:   %s'
            % (' '.join(m for m, _ in driver_calls), ' '.join(m for m, _ in page_calls))
        )
        return problems
    for (method, want), (_, got) in zip(driver_calls, page_calls):
        free = {key for key, value in want.items() if value is None}
        want = {k: v for k, v in want.items() if k not in free}
        got = {k: v for k, v in got.items() if k not in free}
        if want != got:
            problems.append(f'{method}: driver {want}, page {got}')
    return problems


def _notebooks() -> list[Path]:
    return sorted(PAGES.glob('tutorials/tutorial*.ipynb'))


@pytest.mark.parametrize('notebook', _notebooks(), ids=lambda p: p.stem)
def test_tutorial_follows_its_case(notebook: Path) -> None:
    """The visible cells of a tutorial make the PAOFLOW calls of its case's driver."""
    if notebook.stem == 'tutorial03':
        pytest.skip('stub')
    cells = json.loads(notebook.read_text())['cells']
    visible = '\n'.join(
        ''.join(c['source'])
        for c in cells
        if c['cell_type'] == 'code' and not HIDDEN & set(c['metadata'].get('tags', []))
    )
    case = base_case(notebook)
    problems = compare_calls(driver(case), visible)
    if case.data and f'case-data/{case.data["file"]}' not in visible:
        problems.append(f'the setup cell does not download case-data/{case.data["file"]}')
    if problems:
        pytest.fail(
            f'{notebook.name} does not make the PAOFLOW calls of the driver of {case.name}:\n'
            + '\n'.join(problems),
            pytrace=False,
        )


def _howtos() -> list[Path]:
    return sorted(PAGES.glob('howtos/*.md'))


def workflow_script(page: Path) -> Path:
    """The one script of the base case that the how-to shows with ``literalinclude``."""
    targets = {(page.parent / t).resolve() for t in INCLUDE.findall(page.read_text())}
    if len(targets) != 1:
        pytest.fail(f'{page.name} includes {len(targets)} scripts, not 1', pytrace=False)
    return targets.pop()


def figures_script(page: Path) -> Path:
    """The figures and asserts of a how-to: ``docs/learn/howtos/<page>/figures.py``."""
    return page.with_suffix('') / 'figures.py'


def _cases() -> list:
    cases = [
        pytest.param(p, 'notebook', id=p.stem) for p in _notebooks() if p.stem not in UNWRITTEN
    ]
    for page in _howtos():
        if INCLUDE.search(page.read_text()) is None:
            continue  # index
        if page.stem in PRECOMPUTED:
            cases.append(pytest.param(page, 'figures', id=page.stem))
            slow = pytest.mark.slow
            cases.append(pytest.param(page, 'scripts', id=f'{page.stem}-workflow', marks=slow))
        else:
            cases.append(pytest.param(page, 'scripts', id=page.stem))
    return cases


def _run_notebook(path: Path, workdir: Path) -> None:
    nbclient = pytest.importorskip('nbclient')
    nbformat = pytest.importorskip('nbformat')
    from PAOFLOW.gen.cases import case_data_archive

    case = base_case(path)
    if case.data:  # the setup cell extracts this archive instead of downloading it again
        shutil.copy(case_data_archive(case), workdir / case.data['file'])
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name='python3',
        resources={'metadata': {'path': str(workdir)}},
    )
    client.execute()


def _run(script: str, cwd: Path, log: Path | None = None) -> None:
    """Run one script in ``cwd``; with ``log``, save its printed output there."""
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    run = subprocess.run(
        [sys.executable, script],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if log is not None:
        log.write_text(run.stdout)
    if run.returncode != 0:
        pytest.fail(f'{script} failed:\n{run.stdout[-3000:]}\n{run.stderr[-5000:]}', pytrace=False)


def _run_howto(page: Path, mode: str, workdir: Path) -> None:
    from PAOFLOW.gen.paoflow_driver import main as paoflow_gen

    case = base_case(page)
    script = workflow_script(page)
    if script.parent != case.folder:
        pytest.fail(f'{page.name} shows {script}, not a script of its case {case.name}')
    argv = ['--example', case.name, '--workdir', str(workdir)]
    if paoflow_gen(argv + (['--with-data'] if case.data else [])) != 0:
        pytest.fail(f'paoflow-gen --example {case.name} failed', pytrace=False)
    if mode == 'scripts':
        if not case.scripts and script.name != 'main.py':  # it reads the output of the driver
            _run('main.py', workdir, log=workdir / 'main.log')
        _run(script.name, workdir, log=workdir / f'{script.stem}.log')
    _run(str(figures_script(page)), workdir)


@pytest.mark.parametrize(('page', 'mode'), _cases())
def test_page_runs(page: Path, mode: str, tmp_path: Path) -> None:
    """Run the page's code from scratch; its own asserts check the numbers it quotes."""
    if mode == 'notebook':
        _run_notebook(page, tmp_path)
    else:
        _run_howto(page, mode, tmp_path)
