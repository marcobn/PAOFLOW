"""Unit tests for logging helpers in transport IO."""

import pytest

from PAOFLOW.transport.io import log_module


class DummyDataController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


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
    attr = {
        'nbnds': 1,
        'nkpnts': 1,
        'nspin': 1,
        'nawf': 1,
        'nelec': 1.0,
        'Efermi': 0.0,
        'energy_units': 'eV',
    }
    data_controller = DummyDataController({}, attr)

    class DummyAtomic:
        do_overlap_transformation = False

    class DummyData:
        atomic_proj = DummyAtomic()

    lines = log_module.log_proj_data(data_controller, DummyData())

    assert any('orthogonal basis' in line for line in lines)
