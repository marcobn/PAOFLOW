from __future__ import annotations

import pytest

from PAOFLOW.Transport import Transport
from PAOFLOW.transport.Transport import Transport as CompatTransport


def test_build_hamiltonian_blocks_delegates_to_build_blocks(monkeypatch):
    transport = Transport(data_controller=object())
    expected = {'blocks': object()}
    monkeypatch.setattr(transport, 'build_blocks', lambda: expected)

    assert transport.build_hamiltonian_blocks() is expected


def test_build_hamiltonian_blocks_requires_prepare():
    transport = Transport(data_controller=object())

    with pytest.raises(RuntimeError, match=r'Call prepare\(\.\.\.\) before build_blocks\(\)\.'):
        transport.build_hamiltonian_blocks()


def test_compat_import_path_exposes_build_hamiltonian_blocks():
    transport = CompatTransport(data_controller=object())

    assert callable(transport.build_hamiltonian_blocks)
