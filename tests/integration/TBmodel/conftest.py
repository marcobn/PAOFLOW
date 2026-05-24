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
    group = parser.getgroup('tbmodel-assets')

    group.addoption(
        '--tbmodel-assets-archive',
        action='store',
        default=None,
        help='Local path to tbmodel_test_assets tar.gz (overrides env PAOFLOW_TBMODEL_ASSET_ARCHIVE).',
    )
    group.addoption(
        '--tbmodel-assets-url',
        action='store',
        default=None,
        help='URL to tbmodel_test_assets tar.gz (overrides env PAOFLOW_TBMODEL_ASSET_URL).',
    )
    group.addoption(
        '--tbmodel-assets-sha256',
        action='store',
        default=None,
        help='Expected sha256 for the asset tarball (overrides env PAOFLOW_TBMODEL_ASSET_SHA256).',
    )
    group.addoption(
        '--tbmodel-assets-version',
        action='store',
        default=None,
        help='Asset version label used for cache naming (overrides env PAOFLOW_TBMODEL_ASSET_VERSION).',
    )
    group.addoption(
        '--tbmodel-assets-link',
        action='store',
        default='symlink',
        choices=['symlink', 'copy'],
        help='How to overlay assets into the sandbox (symlink or copy).',
    )


@pytest.fixture(scope='session')
def tbmodel_asset_options(pytestconfig: pytest.Config) -> _AssetOptions:
    return _AssetOptions(
        archive=pytestconfig.getoption('--tbmodel-assets-archive'),
        url=pytestconfig.getoption('--tbmodel-assets-url'),
        sha256=pytestconfig.getoption('--tbmodel-assets-sha256'),
        version=pytestconfig.getoption('--tbmodel-assets-version'),
        link_mode=str(pytestconfig.getoption('--tbmodel-assets-link') or 'symlink'),
    )


@pytest.fixture(scope='session')
def tbmodel_assets_root(tbmodel_asset_options: _AssetOptions) -> Optional[Path]:
    source = resolve_asset_source(
        archive_path=tbmodel_asset_options.archive,
        url=tbmodel_asset_options.url,
        sha256=tbmodel_asset_options.sha256,
        version=tbmodel_asset_options.version,
    )

    if source.archive_path is None and not source.url:
        return None

    return ensure_assets_extracted(source)


@pytest.fixture(scope='session')
def tbmodel_assets_link_mode(tbmodel_asset_options: _AssetOptions) -> str:
    return tbmodel_asset_options.link_mode
