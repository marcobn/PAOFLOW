"""Unit tests for header printing helpers."""

import types
import pytest

from PAOFLOW.transport.io import write_header


@pytest.mark.unit
def test_write_header_emits_three_lines(monkeypatch):
    """Header writer should output the separator, centered line, and separator."""
    calls = []

    monkeypatch.setattr(write_header, 'log_rank0', lambda msg: calls.append(msg))

    # mpi4py communicator methods are C-level and read-only; patch the module symbol instead.
    fake_mpi = types.SimpleNamespace(COMM_WORLD=types.SimpleNamespace(Get_rank=lambda: 0))
    monkeypatch.setattr(write_header, 'MPI', fake_mpi)

    write_header.write_header('UNIT TEST')

    assert len(calls) == 3
    assert calls[1].strip().startswith('=')


@pytest.mark.unit
def test_write_header_rejects_long_message():
    """Messages longer than 66 chars should raise a ValueError."""
    with pytest.raises(ValueError):
        write_header.write_header('x' * 66)
