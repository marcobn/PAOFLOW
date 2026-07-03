import pytest

from PAOFLOW.transport.utils import timing as timing_module


@pytest.mark.unit
def test_clock_start_stop_updates_totals(monkeypatch):
    values = iter([1.0, 3.5])
    monkeypatch.setattr(timing_module, 'perf_counter', lambda: next(values))

    clock = timing_module.Clock('unit')
    clock.start()
    clock.stop()

    assert clock.call_count == 1
    assert clock.total_time == pytest.approx(2.5)
    assert clock.avg_time() == pytest.approx(2.5)


@pytest.mark.unit
def test_clock_time_upto_now(monkeypatch):
    values = iter([2.0, 4.0])
    monkeypatch.setattr(timing_module, 'perf_counter', lambda: next(values))

    clock = timing_module.Clock('unit')
    clock.start()

    assert clock.time_upto_now() == pytest.approx(2.0)


@pytest.mark.unit
def test_clock_errors():
    clock = timing_module.Clock('unit')

    with pytest.raises(RuntimeError):
        clock.stop()

    clock.start()

    with pytest.raises(RuntimeError):
        clock.start()


@pytest.mark.unit
def test_timed_function_stops_on_exception(monkeypatch):
    class DummyTiming:
        def __init__(self):
            self.calls = []

        def start(self, name):
            self.calls.append(('start', name))

        def stop(self, name):
            self.calls.append(('stop', name))

    dummy = DummyTiming()
    monkeypatch.setattr(timing_module, 'global_timing', dummy)

    @timing_module.timed_function()
    def explode():
        raise ValueError('boom')

    with pytest.raises(ValueError):
        explode()

    assert dummy.calls == [('start', 'explode'), ('stop', 'explode')]
