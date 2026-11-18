"""Re-run the tutorial notebooks and how-to scripts of the website.

The docs build does not execute the pages (``nb_execution_mode = 'off'``), so this test is what
notices that a page's code no longer runs, or that a number the page quotes has changed: each
page ends with asserts of those numbers (a ``remove-cell`` cell in a notebook, the end of a
how-to's ``figures.py``). A how-to runs as ``python main.py > main.log`` followed by
``python figures.py``. Each case runs in a scratch directory with the release asset of the page,
so the committed outputs and figures are never touched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tests.integration.qe.assets import ensure_assets_available, resolve_asset_source

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
PAGES = REPO / 'docs' / 'learn'
URL = 'https://github.com/marcobn/PAOFLOW/releases/download/'
ASSETS = {  # archive -> (release tag, sha256)
    'tutorial_assets.tar.gz': (
        'tutorial-assets-v2',
        '39940d3e8b02dd77ba914cf05843e4e09a18539c20c665b2f0b5691ed1a9b6e3',
    ),
    'howto_assets.tar.gz': (
        'tutorial-assets-v2',
        'a3515fa9ab519216b153c93971405bd02ddfdf9976f62257ccae4e9f0ce48426',
    ),
}
UNWRITTEN = {'tutorial03'}  # stub without stored outputs; add it to the test once it is written
# how-tos whose main.py exceeds the run-time budget of the learn skill (5 min on 4
# cores): their asset folder holds what main.py writes, so figures.py runs on it on every pull
# request, and the <task>-workflow case that runs main.py is marked slow
PRECOMPUTED = {'twisted-bilayer'}
TIMEOUT = 1800  # s; catches a hung page, far above the budget


def _cases() -> list:
    cases = [
        pytest.param(p, 'notebook', id=p.stem)
        for p in sorted(PAGES.glob('tutorials/tutorial*.ipynb'))
        if p.stem not in UNWRITTEN
    ]
    for folder in sorted(p.parent for p in PAGES.glob('howtos/*/main.py')):
        if folder.name in PRECOMPUTED:
            cases.append(pytest.param(folder, 'figures', id=folder.name))
            slow = pytest.mark.slow
            cases.append(pytest.param(folder, 'scripts', id=f'{folder.name}-workflow', marks=slow))
        else:
            cases.append(pytest.param(folder, 'scripts', id=folder.name))
    return cases


def _archive(name: str) -> Path:
    """Local copy of a release archive, downloaded once into the test-asset cache."""
    release, sha256 = ASSETS[name]
    source = resolve_asset_source(
        url=f'{URL}{release}/{name}',
        sha256=sha256,
        version=release,
        env_archive_var='PAOFLOW_DOCS_ASSET_ARCHIVE',
        env_url_var='PAOFLOW_DOCS_ASSET_URL',
        env_sha256_var='PAOFLOW_DOCS_ASSET_SHA256',
        env_version_var='PAOFLOW_DOCS_ASSET_VERSION',
        cache_key=name.removesuffix('.tar.gz'),
    )
    return ensure_assets_available(source)


def _run_notebook(path: Path, workdir: Path) -> None:
    nbclient = pytest.importorskip('nbclient')
    nbformat = pytest.importorskip('nbformat')
    # the setup cell extracts its input from this archive instead of downloading it again
    shutil.copy(_archive('tutorial_assets.tar.gz'), workdir / 'tutorial_assets.tar.gz')
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook,
        timeout=TIMEOUT,
        kernel_name='python3',
        resources={'metadata': {'path': str(workdir)}},
    )
    client.execute()


def _run(script: Path, cwd: Path, log: Path | None = None) -> None:
    """Run one script in ``cwd``; with ``log``, save its printed output there."""
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    run = subprocess.run(
        [sys.executable, script.name],
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


def _run_howto(folder: Path, mode: str, workdir: Path) -> None:
    # a how-to reads its own folder of the how-to archive, or the DFT data of the tutorials
    with tarfile.open(_archive('howto_assets.tar.gz'), 'r:gz') as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(f'{folder.name}/')]
        tar.extractall(workdir, members=members, filter='data')
    if mode == 'figures':  # the asset folder holds the outputs and main.log of main.py
        precomputed = workdir / folder.name
        shutil.copy(folder / 'figures.py', precomputed)
        _run(precomputed / 'figures.py', precomputed)
        return
    with tarfile.open(_archive('tutorial_assets.tar.gz'), 'r:gz') as tar:
        tar.extractall(workdir, filter='data')
    shutil.copytree(folder, workdir, dirs_exist_ok=True)
    _run(workdir / 'main.py', workdir, log=workdir / 'main.log')
    _run(workdir / 'figures.py', workdir)


@pytest.mark.parametrize(('page', 'mode'), _cases())
def test_page_runs(page: Path, mode: str, tmp_path: Path) -> None:
    """Run the page's code from scratch; its own asserts check the numbers it quotes."""
    if mode == 'notebook':
        _run_notebook(page, tmp_path)
    else:
        _run_howto(page, mode, tmp_path)
