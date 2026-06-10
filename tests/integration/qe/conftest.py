from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from .assets import AssetSource, ensure_assets_extracted, resolve_asset_source


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup('qe-test-assets')

    group.addoption(
        '--qe-test-assets-archive',
        action='store',
        default=None,
        help='Local path to qe_test_assets.tar.gz (overrides env PAOFLOW_QE_TEST_ASSET_ARCHIVE).',
    )
    group.addoption(
        '--qe-test-assets-url',
        action='store',
        default=None,
        help='URL to qe_test_assets.tar.gz (overrides env PAOFLOW_QE_TEST_ASSET_URL).',
    )
    group.addoption(
        '--qe-test-assets-sha256',
        action='store',
        default=None,
        help='Expected sha256 for the QE test asset tarball (overrides env PAOFLOW_QE_TEST_ASSET_SHA256).',
    )
    group.addoption(
        '--qe-test-assets-version',
        action='store',
        default=None,
        help='QE test asset version label used for cache naming (overrides env PAOFLOW_QE_TEST_ASSET_VERSION).',
    )
    group.addoption(
        '--qe-test-assets-link',
        action='store',
        default='symlink',
        choices=['symlink', 'copy'],
        help='How to overlay assets into the sandbox (symlink or copy).',
    )


def _resolve_required_asset_source(
    *,
    archive: Optional[str],
    url: Optional[str],
    sha256: Optional[str],
    version: Optional[str],
    env_archive_var: str,
    env_url_var: str,
    env_sha256_var: str,
    env_version_var: str,
    cache_key: str,
    description: str,
) -> AssetSource:
    source = resolve_asset_source(
        archive_path=archive,
        url=url,
        sha256=sha256,
        version=version,
        env_archive_var=env_archive_var,
        env_url_var=env_url_var,
        env_sha256_var=env_sha256_var,
        env_version_var=env_version_var,
        cache_key=cache_key,
    )

    if source.archive_path is None and not source.url:
        raise pytest.UsageError(
            f'{description} are required. '
            f'Set the corresponding CLI options or '
            f'{env_archive_var}/{env_url_var}.'
        )

    return source


@pytest.fixture(scope='session')
def qe_test_assets_root(pytestconfig: pytest.Config) -> Optional[Path]:
    source = _resolve_required_asset_source(
        archive=pytestconfig.getoption('--qe-test-assets-archive'),
        url=pytestconfig.getoption('--qe-test-assets-url'),
        sha256=pytestconfig.getoption('--qe-test-assets-sha256'),
        version=pytestconfig.getoption('--qe-test-assets-version'),
        env_archive_var='PAOFLOW_QE_TEST_ASSET_ARCHIVE',
        env_url_var='PAOFLOW_QE_TEST_ASSET_URL',
        env_sha256_var='PAOFLOW_QE_TEST_ASSET_SHA256',
        env_version_var='PAOFLOW_QE_TEST_ASSET_VERSION',
        cache_key='qe-test',
        description='QE test assets',
    )

    return ensure_assets_extracted(source)


@pytest.fixture(scope='session')
def qe_test_assets_link_mode(pytestconfig: pytest.Config) -> str:
    return str(pytestconfig.getoption('--qe-test-assets-link') or 'symlink')
