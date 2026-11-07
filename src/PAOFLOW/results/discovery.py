"""Discovery: the datasets of a folder tree (one per kind, folder and group of files).

Every file is matched against the catalogue (by priority, first match wins); files of the same
kind and folder whose named groups differ only in the kind's ``series`` groups (tensor component,
spin channel, temperature, ...) form one dataset with one series per file.  The scan skips QE
saves, basis tables and caches, and stops after :data:`MAX_FILES` files.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from .catalogue import CATEGORY_ORDER, KINDS, SKIPPED_FILES, SKIPPED_FOLDERS
from .model import DatasetRef, Kind

MAX_FILES = 20000
MAX_DEPTH = 6

_ORDERED_KINDS = sorted(KINDS, key=lambda kind: kind.priority)
_COMPILED = [
    (kind, re.compile(kind.pattern), re.compile(kind.exclude) if kind.exclude else None)
    for kind in _ORDERED_KINDS
]
_SKIP_DIR = re.compile(SKIPPED_FOLDERS)
_SKIP_FILE = re.compile(SKIPPED_FILES)


class _Default(dict):
    def __missing__(self, key: str) -> str:
        return ''


@dataclass
class _Match:
    kind: Kind
    groups: dict[str, str]
    path: str
    folder: str
    mtime: float
    size: int


def match_file(name: str, folder: str) -> tuple[Kind, dict[str, str]] | None:
    """The kind of a file and its named groups (``None`` when nothing matches).

    Parameters
    ----------
    name : str
        File name.
    folder : str
        Its folder, relative to the scanned root (``.`` for the root).

    Returns
    -------
    tuple of (Kind, dict) or None
    """
    if _SKIP_FILE.fullmatch(name):
        return None
    for kind, pattern, exclude in _COMPILED:
        match = pattern.fullmatch(name)
        if match is None or (exclude is not None and exclude.fullmatch(name)):
            continue
        groups = {k: v for k, v in match.groupdict().items() if v is not None}
        if kind.folder is not None:
            folder_match = re.search(kind.folder, folder.replace(os.sep, '/'))
            if folder_match is None:
                continue
            groups.update({k: v for k, v in folder_match.groupdict().items() if v is not None})
        return kind, groups
    return None


def _walk(root: str):
    root_depth = root.rstrip(os.sep).count(os.sep)
    count = 0
    for folder, folders, files in os.walk(root):
        depth = folder.count(os.sep) - root_depth
        folders[:] = sorted(
            d
            for d in folders
            if not d.startswith('.') and not _SKIP_DIR.search(d) and depth < MAX_DEPTH
        )
        for name in sorted(files):
            if name.startswith('.'):
                continue
            count += 1
            if count > MAX_FILES:
                return
            yield folder, name


def pretty_label(key: str, value: str) -> str:
    """Display text of one series group value.

    Parameters
    ----------
    key : str
        Group name (``spin``, ``tag``, ``q``, ``temperature``, ...).
    value : str
        Its value in the file name.

    Returns
    -------
    str
    """
    if key == 'spin':
        return {'0': 'spin ↑', '1': 'spin ↓'}.get(value, 'spin %s' % value)
    if key == 'tag':
        return value or 'DFT'
    if key == 'q':
        return {'dos': 'DOS', 'dosdk': 'DOS', 'pdos_sum': 'Σ PDOS', 'pdosdk_sum': 'Σ PDOS'}.get(
            value, value
        )
    if key == 'temperature':
        return '%s K' % value
    if key == 'angle':
        return '%s°' % value
    if key == 'spol':
        return 's=%s' % value
    if key == 'tr':
        return 'transport' if value else 'α²F'
    if key == 'level':
        return {
            'ipa': 'IPA',
            'lf': 'local fields',
            'alda': 'ALDA',
            'lrc': 'LRC',
            'boot': 'bootstrap',
        }.get(value, value)
    return value


def _title(kind: Kind, groups: dict[str, str]) -> str:
    values = _Default({key: value.strip('_') for key, value in groups.items()})
    if kind.describe is not None:
        values['title'] = kind.describe(groups)[0]
    title = kind.title.format_map(values).strip()
    title = re.sub(r'\s+', ' ', title).replace('()', '').strip()
    return title or kind.name


def discover(root: str) -> list[DatasetRef]:
    """Datasets of every PAOFLOW output under ``root``.

    Parameters
    ----------
    root : str
        Folder to scan (an ``output`` folder, a run folder, an electron-phonon system folder).

    Returns
    -------
    list of DatasetRef
        Sorted by category, folder and title.
    """
    root = os.path.abspath(root)
    matches: list[_Match] = []
    for folder, name in _walk(root):
        relative_folder = os.path.relpath(folder, root)
        found = match_file(name, relative_folder)
        if found is None:
            continue
        kind, groups = found
        path = os.path.join(folder, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        matches.append(
            _Match(
                kind,
                groups,
                os.path.relpath(path, root),
                relative_folder,
                stat.st_mtime,
                stat.st_size,
            )
        )
    grouped: dict[tuple, list[_Match]] = {}
    for item in matches:
        shared = tuple(sorted((k, v) for k, v in item.groups.items() if k not in item.kind.series))
        grouped.setdefault((item.kind.name, item.folder, shared), []).append(item)
    references = []
    for (kind_name, folder, shared), items in grouped.items():
        kind = items[0].kind
        varying = [key for key in kind.series if len({i.groups.get(key, '') for i in items}) > 1]
        items.sort(
            key=lambda i: tuple(_sort_value(i.groups.get(k, '')) for k in kind.series) + (i.path,)
        )
        labels = [
            ' · '.join(
                pretty_label(k, i.groups.get(k, ''))
                for k in varying
                if (i.groups.get(k, '') or k == 'tag')
            )
            for i in items
        ]
        identity = '%s|%s|%s' % (kind_name, folder, shared)
        references.append(
            DatasetRef(
                id=hashlib.sha1(identity.encode()).hexdigest()[:12],
                kind=kind_name,
                title=_title(kind, dict(shared)),
                category=kind.category,
                family=kind.family,
                folder=folder,
                files=[i.path for i in items],
                labels=labels,
                group=dict(shared),
                mtime=max(i.mtime for i in items),
                size=sum(i.size for i in items),
            )
        )
    order = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    references.sort(key=lambda r: (order.get(r.category, 99), r.folder, r.title))
    return references


def _sort_value(value: str):
    try:
        return (0, float(value), '')
    except ValueError:
        components = ('xx', 'yy', 'zz', 'xy', 'xz', 'yz')
        return (1, components.index(value) if value in components else 99, value)
