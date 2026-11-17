"""Re-run the tutorial notebooks and how-to scripts of the website.

The docs build does not execute the pages (``nb_execution_mode = 'off'``), so this test is what
notices that a page's code no longer runs, or that a number the page quotes has changed: each
page ends with asserts of those numbers (a ``remove-cell`` cell in a notebook, the end of a
how-to's ``main.py``). Each case runs in a scratch directory with the release asset of the page,
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
PAGES = REPO / 'docs' / 'tutorials'
RELEASE = 'tutorial-assets-v1'
URL = f'https://github.com/marcobn/PAOFLOW/releases/download/{RELEASE}/'
SHA256 = {
    'tutorial_assets.tar.gz': 'df21d5d53d57082952ec5488526a938dc47c0ffbbc2d4ea31fcf1883ee40bb21',
    'howto_assets.tar.gz': '033fe224a2c1e086feabfd667a02b62e21c4ecf7efceac4708ef911d7eed666f',
}
UNWRITTEN = {'tutorial03'}  # stub without stored outputs; add it to the test once it is written
SLOW = set()  # pages above the run-time budget of the tutorials-howtos skill (5 min on 4 cores)
TIMEOUT = 1800  # s; catches a hung page, far above the budget


def _name(path: Path) -> str:
    """Page name: the notebook stem, or the how-to directory that holds ``main.py``."""
    return path.stem if path.suffix == '.ipynb' else path.parent.name


def _cases() -> list:
    notebooks = [p for p in sorted(PAGES.glob('tutorial*.ipynb')) if p.stem not in UNWRITTEN]
    scripts = sorted(PAGES.glob('howtos/*/main.py'))
    return [
        pytest.param(p, id=_name(p), marks=[pytest.mark.slow] if _name(p) in SLOW else [])
        for p in notebooks + scripts
    ]


def _archive(name: str) -> Path:
    """Local copy of a release archive, downloaded once into the test-asset cache."""
    source = resolve_asset_source(
        url=URL + name,
        sha256=SHA256[name],
        version=RELEASE,
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


def _run_script(path: Path, workdir: Path) -> None:
    shutil.copytree(path.parent, workdir, dirs_exist_ok=True)
    with tarfile.open(_archive('howto_assets.tar.gz'), 'r:gz') as tar:
        prefix = f'{path.parent.name}/'
        members = [m for m in tar.getmembers() if m.name.startswith(prefix)]
        tar.extractall(workdir, members=members, filter='data')
    env = {**os.environ, 'MPLBACKEND': 'Agg'}
    run = subprocess.run(
        [sys.executable, path.name],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT,
    )
    if run.returncode != 0:
        pytest.fail(f'{path} failed:\n{run.stdout[-3000:]}\n{run.stderr[-5000:]}', pytrace=False)


@pytest.mark.parametrize('page', _cases())
def test_page_runs(page: Path, tmp_path: Path) -> None:
    """Run the page's code from scratch; its own asserts check the numbers it quotes."""
    if page.suffix == '.ipynb':
        _run_notebook(page, tmp_path)
    else:
        _run_script(page, tmp_path)
