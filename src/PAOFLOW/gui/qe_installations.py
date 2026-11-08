"""Quantum ESPRESSO installations the Runs tab can run (``pw.x``, ``ph.x``, ... of one ``bin/``).

A machine often holds several QE builds (``~/Programs/qe-7.4.1``, ``qe-7.5``, a conda
environment, ...), and the PATH of the server finds only one of them.  :func:`discover_installations`
lists them all, and the Runs tab chooses one by its ``bin`` folder.  The server accepts only folders
of this list, so no request can run an executable that was not found here.

Where the search looks:

- the folders of the PATH that hold ``pw.x``;
- the siblings of those installations (``<parent>/*/bin/pw.x``: ``qe-7.5`` next to ``qe-7.4.1``);
- the folders given to ``paoflow-studio --qe-bin`` (an installation or its ``bin/``);
- common places, unless ``search=False``: ``~/Programs``, ``~/Local/Programs``, ``~/software``,
  ``~/opt``, ``~/apps``, ``/opt``, ``/usr/local``, Homebrew, and the conda environments.

The version comes from the build tree (``include/qe_version.h``), conda metadata or the folder name;
``make.inc`` (or ``CMakeCache.txt``) tells whether the build has MPI.  Nothing is executed.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

#: Install roots searched for ``*/bin/pw.x`` (relative to the home folder, or absolute).
HOME_ROOTS = ('Programs', 'Local/Programs', 'software', 'opt', 'apps')
SYSTEM_ROOTS = ('/opt', '/usr/local')
CONDA_BASES = ('anaconda3', 'miniconda3', 'miniforge3', 'mambaforge', 'micromamba')
HOMEBREW_BINS = ('/opt/homebrew/bin', '/usr/local/bin')

VERSION_LINE = re.compile(r"version_number\s*=\s*'([^']+)'")
VERSION_IN_NAME = re.compile(r'(\d+\.\d+(?:\.\d+)?)')


@dataclass(frozen=True)
class QeInstallation:
    """One Quantum ESPRESSO build.

    Parameters
    ----------
    bin : Path
        Folder with ``pw.x`` (put first on the PATH of a run).
    root : Path
        Installation folder (``bin/..`` for a build tree, the package prefix for a package).
    version : str
        ``'7.4.1'``; empty when unknown.
    mpi : bool or None
        Built with MPI (``make.inc`` or ``CMakeCache.txt``); ``None`` when unknown.
    on_path : bool
        The ``pw.x`` the PATH of the server finds.
    programs : tuple of str
        The QE programs of :data:`PAOFLOW.gui.jobs.QE_PROGRAMS` present in ``bin``.
    """

    bin: Path
    root: Path
    version: str
    mpi: bool | None
    on_path: bool
    programs: tuple[str, ...]

    @property
    def mpirun(self) -> Path | None:
        """The ``mpirun`` shipped with the build (conda environments), if any."""
        for name in ('mpirun', 'mpiexec'):
            candidate = self.bin / name
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        return None

    def label(self) -> str:
        """``'7.4.1 · ~/Local/Programs/qe-7.4.1'`` (with the PATH and serial marks)."""
        text = '%s · %s' % (self.version or 'version ?', _home_short(self.root))
        if self.on_path:
            text += ' (PATH)'
        if self.mpi is False:
            text += ' · serial'
        return text

    def to_json(self) -> dict:
        """Fields the GUI shows (paths as strings)."""
        return {
            'bin': str(self.bin),
            'root': str(self.root),
            'version': self.version,
            'mpi': self.mpi,
            'on_path': self.on_path,
            'programs': list(self.programs),
            'label': self.label(),
        }


def _home_short(path: Path) -> str:
    home = str(Path.home())
    text = str(path)
    return '~' + text[len(home) :] if text == home or text.startswith(home + os.sep) else text


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _bin_of(folder: Path) -> Path | None:
    """``folder`` itself or ``folder/bin`` when it holds an executable ``pw.x``."""
    for candidate in (folder, folder / 'bin'):
        if _executable(candidate / 'pw.x'):
            return candidate
    return None


def _install_roots(folder: Path, real: Path) -> list[Path]:
    """Folders that may describe the build, the installation folder first.

    A QE build tree links ``bin/pw.x`` to ``PW/src/pw.x``: its installation is ``bin/..``.  A
    package (Homebrew, a shared ``/usr/local/bin``) links ``bin/pw.x`` to ``<prefix>/bin/pw.x``:
    its installation is that prefix.
    """
    first = real.parent.parent if real.parent.name == 'bin' else folder.parent
    roots = []
    for root in (first, folder.parent, real.parent.parent, real.parent.parent.parent):
        if root not in roots:
            roots.append(root)
    return roots


def _version(roots: Sequence[Path]) -> str:
    for root in roots:
        try:
            match = VERSION_LINE.search(
                (root / 'include' / 'qe_version.h').read_text('utf-8', 'replace')
            )
        except OSError:
            continue
        if match:
            return match.group(1).strip()
    for root in roots:
        meta = root / 'conda-meta'
        if meta.is_dir():
            for record in sorted(meta.glob('qe-*.json')):
                match = re.match(r'qe-(\d[^-]*)-', record.name)
                if match:
                    return match.group(1)
    for root in roots:
        match = VERSION_IN_NAME.search(root.name)
        if match:
            return match.group(1)
    return ''


def _mpi(roots: Sequence[Path]) -> bool | None:
    for root in roots:
        try:
            text = (root / 'make.inc').read_text('utf-8', 'replace')
        except OSError:
            continue
        flags = ' '.join(line for line in text.splitlines() if line.strip().startswith('DFLAGS'))
        return '__MPI' in flags
    for root in roots:
        try:
            text = (root / 'CMakeCache.txt').read_text('utf-8', 'replace')
        except OSError:
            continue
        match = re.search(r'^QE_ENABLE_MPI:BOOL=(\w+)', text, re.MULTILINE)
        if match:
            return match.group(1).upper() in ('ON', 'TRUE', '1', 'YES')
    return None


def _version_key(installation: QeInstallation) -> tuple:
    numbers = tuple(int(n) for n in re.findall(r'\d+', installation.version)[:3])
    return (bool(numbers), numbers, str(installation.bin))


def _search_folders(path_bins: Sequence[Path], search: bool) -> Iterable[Path]:
    for folder in path_bins:  # siblings of the installations on the PATH
        installation = _install_roots(folder, (folder / 'pw.x').resolve())[0]
        yield from sorted(installation.parent.glob('*/bin'))
    if not search:
        return
    home = Path.home()
    roots = [home / name for name in HOME_ROOTS] + [Path(name) for name in SYSTEM_ROOTS]
    for root in roots:
        yield root / 'bin'
        yield from sorted(root.glob('*/bin'))
        yield from sorted(root.glob('*/*/bin'))  # ~/Local/Programs/quantum-espresso/qe-7.5/bin
    for base in CONDA_BASES:
        yield home / base / 'bin'
        yield from sorted((home / base / 'envs').glob('*/bin'))
    prefix = os.environ.get('CONDA_PREFIX')
    if prefix:
        yield Path(prefix) / 'bin'
    for folder in HOMEBREW_BINS:
        yield Path(folder)


def discover_installations(
    extra: Sequence[str | Path] = (), path: str | None = None, search: bool = True
) -> list[QeInstallation]:
    """The Quantum ESPRESSO builds of this machine, newest first.

    Parameters
    ----------
    extra : sequence of str or Path, optional
        Folders given by the user (``--qe-bin``): an installation or its ``bin/``.
    path : str, optional
        The PATH to read (default: the PATH of this process).
    search : bool, optional
        Also look in the common places (default). The PATH, its siblings and ``extra`` are always
        searched.

    Returns
    -------
    list of QeInstallation
        One entry per ``pw.x`` (symbolic links to the same file count once).
    """
    from .jobs import QE_PROGRAMS

    path = os.environ.get('PATH', '') if path is None else path
    path_bins = []
    for entry in path.split(os.pathsep):
        if entry and _executable(Path(entry) / 'pw.x'):
            path_bins.append(Path(entry))
    first_on_path = (path_bins[0] / 'pw.x').resolve() if path_bins else None
    candidates = [_bin_of(Path(os.path.expanduser(str(folder)))) for folder in extra]
    candidates += path_bins
    candidates += [_bin_of(folder) for folder in _search_folders(path_bins, search)]
    found: dict[Path, QeInstallation] = {}
    for folder in candidates:
        if folder is None:
            continue
        folder = Path(os.path.abspath(folder))
        real = (folder / 'pw.x').resolve()
        if real in found:
            continue
        roots = _install_roots(folder, real)
        found[real] = QeInstallation(
            bin=folder,
            root=roots[0],
            version=_version(roots),
            mpi=_mpi(roots),
            on_path=real == first_on_path,
            programs=tuple(name for name in QE_PROGRAMS if _executable(folder / name)),
        )
    return sorted(found.values(), key=_version_key, reverse=True)


def default_installation(installations: Sequence[QeInstallation]) -> QeInstallation | None:
    """The build a run uses when none is chosen: the one on the PATH, else the newest.

    Parameters
    ----------
    installations : sequence of QeInstallation
        From :func:`discover_installations`.

    Returns
    -------
    QeInstallation or None
    """
    for installation in installations:
        if installation.on_path:
            return installation
    return installations[0] if installations else None
