"""Discovery of the Quantum ESPRESSO builds the Runs tab offers (no QE needed: stand-in trees)."""

from __future__ import annotations

import os

import pytest

from PAOFLOW.gui.qe_installations import default_installation, discover_installations

pytestmark = pytest.mark.unit


def _build_tree(root, version, mpi=True, programs=('pw.x', 'ph.x')):
    """A QE build tree: bin/<program> links to <PKG>/src/<program>, like `make` leaves it."""
    (root / 'bin').mkdir(parents=True)
    (root / 'PW' / 'src').mkdir(parents=True)
    for program in programs:
        target = root / 'PW' / 'src' / program
        target.write_text('#!/bin/sh\necho "Program %s v.%s"\n' % (program, version))
        target.chmod(0o755)
        (root / 'bin' / program).symlink_to(os.path.join('..', 'PW', 'src', program))
    if version:
        (root / 'include').mkdir()
        (root / 'include' / 'qe_version.h').write_text(
            "! global version number\nCHARACTER (LEN=6) :: version_number = '%s'\n" % version
        )
    (root / 'make.inc').write_text('DFLAGS = -D__FFTW%s\n' % (' -D__MPI' if mpi else ''))
    return root / 'bin'


def test_path_build_its_siblings_and_their_versions(tmp_path):
    programs = tmp_path / 'Programs'
    on_path = _build_tree(programs / 'qe-7.4.1', '7.4.1')
    _build_tree(programs / 'qe-7.5', '7.5', programs=('pw.x', 'ph.x', 'epw.x'))
    _build_tree(programs / 'qe-6.8', '6.8', mpi=False)
    (programs / 'notes').mkdir()  # a sibling that is no QE build
    found = discover_installations(path='/usr/bin:%s' % on_path, search=False)
    assert [q.version for q in found] == ['7.5', '7.4.1', '6.8']
    newest, current, old = found
    assert current.on_path and not newest.on_path and default_installation(found) is current
    assert newest.root == programs / 'qe-7.5' and newest.bin == programs / 'qe-7.5' / 'bin'
    assert newest.programs == ('pw.x', 'ph.x', 'epw.x') and newest.mpi is True
    assert old.mpi is False and old.label().endswith(' · serial')
    assert current.label().endswith('qe-7.4.1 (PATH)')


def test_extra_folders_conda_metadata_and_duplicates(tmp_path):
    programs = tmp_path / 'Programs'
    bin_75 = _build_tree(programs / 'qe-7.5', '7.5')
    unversioned = _build_tree(tmp_path / 'codes' / 'espresso-git', '')
    env = tmp_path / 'envs' / 'qe'
    (env / 'bin').mkdir(parents=True)
    (env / 'conda-meta').mkdir()
    (env / 'conda-meta' / 'qe-7.3.1-mpi_openmpi_h1234_0.json').write_text('{}')
    for name in ('pw.x', 'mpirun'):
        (env / 'bin' / name).write_text('#!/bin/sh\n')
        (env / 'bin' / name).chmod(0o755)
    links = tmp_path / 'links'
    links.mkdir()
    (links / 'pw.x').symlink_to(bin_75 / 'pw.x')  # the same build seen twice
    found = discover_installations(
        extra=[unversioned.parent, env, links, bin_75], path='', search=False
    )
    by_version = {q.version: q for q in found}
    assert set(by_version) == {'7.5', '7.3.1', ''}
    assert by_version['7.3.1'].mpirun == env / 'bin' / 'mpirun'
    assert by_version['7.5'].mpirun is None
    assert by_version[''].bin == unversioned and by_version[''].label().startswith('version ?')
    assert found[-1].version == ''  # unknown versions last
    assert default_installation(found) is found[0]  # nothing on the PATH: the newest


def test_nothing_found(tmp_path):
    assert discover_installations(path=str(tmp_path), search=False) == []
    assert default_installation([]) is None
