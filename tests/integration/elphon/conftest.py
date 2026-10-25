"""Fixture that provides the electron-phonon integration asset (Si DFPT data + EPW reference)."""

from __future__ import annotations

import os

import pytest

from tests.integration.qe.assets import ensure_assets_extracted, resolve_asset_source


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('elphon-test-assets')
    group.addoption(
        '--elphon-test-assets-archive',
        action='store',
        default=None,
        help='Local elphon_test_assets.tar.gz',
    )
    group.addoption(
        '--elphon-test-assets-url',
        action='store',
        default=None,
        help='URL of elphon_test_assets.tar.gz',
    )
    group.addoption(
        '--elphon-test-assets-sha256',
        action='store',
        default=None,
        help='Expected sha256 of the tarball',
    )
    group.addoption(
        '--elphon-test-assets-version',
        action='store',
        default=None,
        help='Version label for the cache',
    )


@pytest.fixture(scope='session')
def elphon_assets(request: pytest.FixtureRequest) -> str:
    """Directory holding ``si/`` (nscf save, ph.x ``save/`` and the EPW reference subset)."""
    opt = request.config.getoption
    source = resolve_asset_source(
        archive_path=opt('--elphon-test-assets-archive'),
        url=opt('--elphon-test-assets-url'),
        sha256=opt('--elphon-test-assets-sha256'),
        version=opt('--elphon-test-assets-version'),
        env_archive_var='PAOFLOW_ELPHON_ASSET_ARCHIVE',
        env_url_var='PAOFLOW_ELPHON_ASSET_URL',
        env_sha256_var='PAOFLOW_ELPHON_ASSET_SHA256',
        env_version_var='PAOFLOW_ELPHON_ASSET_VERSION',
        cache_key='elphon',
    )
    if source.archive_path is None and not source.url:
        pytest.skip(
            'electron-phonon assets not configured: set PAOFLOW_ELPHON_ASSET_ARCHIVE (or _URL/_SHA256/_VERSION)'
        )
    root = ensure_assets_extracted(source)
    path = os.path.join(root, 'si')
    if not os.path.isdir(path):
        pytest.skip('electron-phonon asset tarball has no si/ directory')
    return path
