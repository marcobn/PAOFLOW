import pytest

from PAOFLOW.transport.utils import divide_et_impera


@pytest.mark.unit
def test_divide_work_balances_ranges(monkeypatch):
    calls = []

    def fake_log(chunk, item):
        calls.append((chunk, item))

    monkeypatch.setattr(divide_et_impera, 'log_parallelization_info', fake_log)

    ranges = [divide_et_impera.divide_work(0, 9, rank, 3, 'items') for rank in range(3)]

    assert ranges == [(0, 3), (4, 6), (7, 9)]
    assert calls == [(3, 'items')] * 3
