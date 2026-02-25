from __future__ import annotations

import hashlib
import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class AssetSource:
    archive_path: Optional[Path]
    url: Optional[str]
    sha256: Optional[str]
    version: str


def _default_cache_dir() -> Path:
    xdg = os.environ.get('XDG_CACHE_HOME')
    if xdg:
        return Path(xdg).expanduser().resolve() / 'paoflow' / 'qe-assets'
    return Path.home().expanduser().resolve() / '.cache' / 'paoflow' / 'qe-assets'


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir.resolve()

    for member in tar.getmembers():
        member_path = (dest_dir / member.name).resolve()
        if not str(member_path).startswith(str(dest) + os.sep) and member_path != dest:
            raise RuntimeError(f'Unsafe path in tar archive: {member.name}')

    tar.extractall(dest_dir)


def resolve_asset_source(
    *,
    archive_path: Optional[str] = None,
    url: Optional[str] = None,
    sha256: Optional[str] = None,
    version: Optional[str] = None,
) -> AssetSource:
    """Resolve asset configuration.

    Priority: explicit args -> env vars.

    Env vars:
      - PAOFLOW_QE_ASSET_ARCHIVE: local path to a tar.gz
      - PAOFLOW_QE_ASSET_URL: URL to download the tar.gz
      - PAOFLOW_QE_ASSET_SHA256: expected sha256 for the archive
      - PAOFLOW_QE_ASSET_VERSION: label used for caching (default: "dev")
    """

    env_archive = os.environ.get('PAOFLOW_QE_ASSET_ARCHIVE')
    env_url = os.environ.get('PAOFLOW_QE_ASSET_URL')
    env_sha = os.environ.get('PAOFLOW_QE_ASSET_SHA256')
    env_ver = os.environ.get('PAOFLOW_QE_ASSET_VERSION')

    archive = archive_path if archive_path is not None else env_archive
    url_val = url if url is not None else env_url
    sha_val = sha256 if sha256 is not None else env_sha
    ver_val = (version if version is not None else env_ver) or 'dev'

    archive_p = Path(archive).expanduser().resolve() if archive else None
    if archive_p is None and not url_val:
        return AssetSource(archive_path=None, url=None, sha256=sha_val, version=ver_val)

    return AssetSource(archive_path=archive_p, url=url_val, sha256=sha_val, version=ver_val)


def ensure_assets_available(
    source: AssetSource,
    *,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Ensure the tarball exists locally (download if needed).

    Returns the local archive path.
    """

    cache = (cache_dir or _default_cache_dir()).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    if source.archive_path is not None:
        if not source.archive_path.is_file():
            raise FileNotFoundError(f'Asset archive not found: {source.archive_path}')
        if source.sha256:
            got = _sha256_file(source.archive_path)
            if got.lower() != source.sha256.lower():
                raise RuntimeError(
                    f'Asset archive sha256 mismatch: got {got}, expected {source.sha256}'
                )
        return source.archive_path

    if not source.url:
        raise RuntimeError('No asset source configured (archive path or URL).')

    # Download mode.
    archive_path = cache / f'qe_test_assets_{source.version}.tar.gz'
    if archive_path.is_file() and source.sha256:
        got = _sha256_file(archive_path)
        if got.lower() == source.sha256.lower():
            return archive_path

    tmp = archive_path.with_suffix(archive_path.suffix + '.tmp')
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(source.url) as r, tmp.open('wb') as f:
        f.write(r.read())

    if source.sha256:
        got = _sha256_file(tmp)
        if got.lower() != source.sha256.lower():
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f'Downloaded asset sha256 mismatch: got {got}, expected {source.sha256}'
            )

    tmp.replace(archive_path)
    return archive_path


def ensure_assets_extracted(
    source: AssetSource,
    *,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Extract the asset tarball into a cache directory and return its root."""

    cache = (cache_dir or _default_cache_dir()).resolve()
    cache.mkdir(parents=True, exist_ok=True)

    archive_path = ensure_assets_available(source, cache_dir=cache)
    digest = _sha256_file(archive_path)
    extract_root = cache / 'extracted' / digest
    marker = extract_root / '.extracted'

    if marker.is_file():
        return extract_root

    if extract_root.exists():
        # Partial extraction from a previous interrupted run.
        for child in extract_root.iterdir():
            if child.name != '.extracted':
                if child.is_dir():
                    # Best-effort cleanup; if it fails, extract will overwrite.
                    pass

    with tarfile.open(archive_path, 'r:*') as tf:
        _safe_extract_tar(tf, extract_root)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f'sha256={digest}\n', encoding='utf-8')
    return extract_root
