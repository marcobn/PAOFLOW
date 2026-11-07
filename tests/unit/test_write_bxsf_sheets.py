"""``DataController.write_bxsf``: the per-band SKEAF sheets are written only for band energies."""

from __future__ import annotations

import numpy as np

from PAOFLOW.DataController import DataController


class _Recorder:
    """A stand-in carrying the attributes ``write_bxsf`` reads."""

    def __init__(self) -> None:
        self.rank = 0
        self.data_attributes = {'fermi_up': 1.0, 'fermi_dw': -1.0}


def _calls(monkeypatch, **kwargs) -> list[str]:
    import PAOFLOW.writers.write2bxsf as full_writer
    import PAOFLOW.writers.write2bxsf4skeaf as sheet_writer

    called: list[str] = []
    monkeypatch.setattr(full_writer, 'write2bxsf', lambda *a, **k: called.append('bxsf'))
    monkeypatch.setattr(sheet_writer, 'write2bxsf4skeaf', lambda *a, **k: called.append('sheets'))
    values = np.zeros((2, 2, 2, 2))
    DataController.write_bxsf(_Recorder(), 'field.bxsf', values, 2, **kwargs)
    return called


def test_fermi_surface_writes_the_skeaf_sheets(monkeypatch):
    assert _calls(monkeypatch) == ['bxsf', 'sheets']


def test_berry_curvature_does_not_overwrite_the_sheets(monkeypatch):
    assert _calls(monkeypatch, skeaf_sheets=False) == ['bxsf']
