from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from .assets import ensure_assets_extracted, resolve_asset_source


@dataclass(frozen=True)
class _AssetOptions:
    archive: Optional[str]
    url: Optional[str]
    sha256: Optional[str]
    version: Optional[str]
    link_mode: str


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('qe-assets')

    group.addoption(
        '--qe-assets-archive',
        action='store',
        default=None,
        help='Local path to qe_test_assets tar.gz (overrides env PAOFLOW_QE_ASSET_ARCHIVE).',
    )
    group.addoption(
        '--qe-assets-url',
        action='store',
        default=None,
        help='URL to qe_test_assets tar.gz (overrides env PAOFLOW_QE_ASSET_URL).',
    )
    group.addoption(
        '--qe-assets-sha256',
        action='store',
        default=None,
        help='Expected sha256 for the asset tarball (overrides env PAOFLOW_QE_ASSET_SHA256).',
    )
    group.addoption(
        '--qe-assets-version',
        action='store',
        default=None,
        help='Asset version label used for cache naming (overrides env PAOFLOW_QE_ASSET_VERSION).',
    )
    group.addoption(
        '--qe-assets-link',
        action='store',
        default='symlink',
        choices=['symlink', 'copy'],
        help='How to overlay assets into the sandbox (symlink or copy).',
    )


@pytest.fixture(scope='session')
def qe_asset_options(pytestconfig: pytest.Config) -> _AssetOptions:
    return _AssetOptions(
        archive=pytestconfig.getoption('--qe-assets-archive'),
        url=pytestconfig.getoption('--qe-assets-url'),
        sha256=pytestconfig.getoption('--qe-assets-sha256'),
        version=pytestconfig.getoption('--qe-assets-version'),
        link_mode=str(pytestconfig.getoption('--qe-assets-link') or 'symlink'),
    )


@pytest.fixture(scope='session')
def qe_assets_root(qe_asset_options: _AssetOptions) -> Optional[Path]:
    source = resolve_asset_source(
        archive_path=qe_asset_options.archive,
        url=qe_asset_options.url,
        sha256=qe_asset_options.sha256,
        version=qe_asset_options.version,
    )

    if source.archive_path is None and not source.url:
        raise pytest.UsageError(
            'QE integration assets are required. '
            'Set --qe-assets-archive/--qe-assets-url or '
            'PAOFLOW_QE_ASSET_ARCHIVE/PAOFLOW_QE_ASSET_URL.'
        )

    return ensure_assets_extracted(source)


@pytest.fixture(scope='session')
def qe_assets_link_mode(qe_asset_options: _AssetOptions) -> str:
    return qe_asset_options.link_mode
