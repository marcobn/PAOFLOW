"""The launch contract of the drivers that run Quantum ESPRESSO (PAOFLOW_* variables first)."""

from __future__ import annotations

import pytest

from PAOFLOW.utils import qe_launch as ql

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ql.VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_defaults_without_variables():
    assert ql.mpi_prefix('mpirun -np 4') == 'mpirun -np 4'
    assert ql.qe_command('ph.x', 'ph.x -npool 4') == 'ph.x -npool 4'
    assert ql.qe_bin('/opt/qe/bin') == '/opt/qe/bin' and ql.qe_options() == ''
    assert ql.mpi_prefix('mpirun -np 2', ql.MPI_HARTREE) == 'mpirun -np 2'


def test_variables_take_precedence(monkeypatch):
    monkeypatch.setenv(ql.MPI_QE, '')  # one rank: serial, not the default
    assert ql.mpi_prefix('mpirun -np 4') == ''
    monkeypatch.setenv(ql.MPI_QE, 'mpirun -np 8')
    monkeypatch.setenv(ql.QE_BIN, '/q/bin')
    assert ql.qe_command('pw.x', 'pw.x -npool 4') == '/q/bin/pw.x'  # the run decides the options
    monkeypatch.setenv(ql.QE_OPTIONS, '-nk 4')
    assert ql.qe_command('ph.x', 'ph.x') == '/q/bin/ph.x -nk 4'
    assert ql.mpi_prefix('mpirun -np 2', ql.MPI_HARTREE) == 'mpirun -np 8'  # follows pw.x
    assert ql.mpi_prefix('mpirun -np 1', ql.MPI_PY) == 'mpirun -np 1'  # python does not
    monkeypatch.setenv(ql.MPI_HARTREE, 'mpirun -np 16')
    assert ql.mpi_prefix('mpirun -np 2', ql.MPI_HARTREE) == 'mpirun -np 16'
    monkeypatch.delenv(ql.QE_BIN)
    assert ql.qe_command('pw.x', 'pw.x') == 'pw.x -nk 4'
