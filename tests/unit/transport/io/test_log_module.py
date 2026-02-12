"""Unit tests for logging helpers in transport IO."""

import numpy as np
import pytest

from PAOFLOW.transport.io import log_module
from PAOFLOW.transport.io.input_parameters import AtomicProjData


@pytest.mark.unit
def test_log_rank0_emits_only_on_rank_zero(monkeypatch):
    """log_rank0 should only emit when rank==0 and logger is initialized."""
    calls = []

    class DummyLogger:
        def info(self, message):
            calls.append(message)

    monkeypatch.setattr(log_module, '_logger', DummyLogger())
    monkeypatch.setattr(log_module, 'rank', 0)

    log_module.log_rank0('hello')

    monkeypatch.setattr(log_module, 'rank', 1)
    log_module.log_rank0('ignored')

    assert calls == ['hello']


@pytest.mark.unit
def test_log_proj_data_adds_overlap_message():
    """log_proj_data should include orthogonal basis message when overlap is off."""
    proj_data = AtomicProjData(
        nbnds=1,
        nkpnts=1,
        nspin=1,
        nawf=1,
        nelec=1.0,
        efermi=0.0,
        energy_units='eV',
        kpts=np.zeros((3, 1)),
        wk=np.ones(1),
        eigvals=np.zeros((1, 1, 1)),
        proj=np.zeros((1, 1, 1, 1), dtype=complex),
    )

    class DummyAtomic:
        do_overlap_transformation = False

    class DummyData:
        atomic_proj = DummyAtomic()

    lines = log_module.log_proj_data(proj_data, DummyData())

    assert any('orthogonal basis' in line for line in lines)
