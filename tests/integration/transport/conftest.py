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
    group = parser.getgroup('transport-assets')

    group.addoption(
        '--transport-assets-archive',
        action='store',
        default=None,
        help='Local path to transport_test_assets tar.gz (overrides env PAOFLOW_TRANSPORT_ASSET_ARCHIVE).',
    )
    group.addoption(
        '--transport-assets-url',
        action='store',
        default=None,
        help='URL to transport_test_assets tar.gz (overrides env PAOFLOW_TRANSPORT_ASSET_URL).',
    )
    group.addoption(
        '--transport-assets-sha256',
        action='store',
        default=None,
        help='Expected sha256 for the asset tarball (overrides env PAOFLOW_TRANSPORT_ASSET_SHA256).',
    )
    group.addoption(
        '--transport-assets-version',
        action='store',
        default=None,
        help='Asset version label used for cache naming (overrides env PAOFLOW_TRANSPORT_ASSET_VERSION).',
    )
    group.addoption(
        '--transport-assets-link',
        action='store',
        default='symlink',
        choices=['symlink', 'copy'],
        help='How to overlay assets into the sandbox (symlink or copy).',
    )


@pytest.fixture(scope='session')
def transport_asset_options(pytestconfig: pytest.Config) -> _AssetOptions:
    return _AssetOptions(
        archive=pytestconfig.getoption('--transport-assets-archive'),
        url=pytestconfig.getoption('--transport-assets-url'),
        sha256=pytestconfig.getoption('--transport-assets-sha256'),
        version=pytestconfig.getoption('--transport-assets-version'),
        link_mode=str(pytestconfig.getoption('--transport-assets-link') or 'symlink'),
    )


@pytest.fixture(scope='session')
def transport_assets_root(transport_asset_options: _AssetOptions) -> Optional[Path]:
    source = resolve_asset_source(
        archive_path=transport_asset_options.archive,
        url=transport_asset_options.url,
        sha256=transport_asset_options.sha256,
        version=transport_asset_options.version,
    )

    if source.archive_path is None and not source.url:
        return None

    return ensure_assets_extracted(source)


@pytest.fixture(scope='session')
def transport_assets_link_mode(transport_asset_options: _AssetOptions) -> str:
    return transport_asset_options.link_mode
