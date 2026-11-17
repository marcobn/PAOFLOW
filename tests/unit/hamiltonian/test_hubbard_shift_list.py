"""A HubbardU list shorter than the basis shifts the first orbitals (``add_external_fields``)."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _model_run(tmp_path, monkeypatch):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import kane_mele_model

    monkeypatch.chdir(tmp_path)
    return PAOFLOW.PAOFLOW(model=kane_mele_model(), outputdir='output', verbose=False)


def test_short_list_covers_the_first_orbitals(tmp_path, monkeypatch):
    p = _model_run(tmp_path, monkeypatch)
    hrs = p.data_controller.data_arrays['HRs']
    nawf = hrs.shape[0]
    before = np.array(hrs).reshape(nawf, nawf, -1, hrs.shape[-1])[:, :, 0, 0].copy()
    p.add_external_fields(HubbardU=[0.0, 0.4])
    hrs = p.data_controller.data_arrays['HRs']
    after = np.array(hrs).reshape(nawf, nawf, -1, hrs.shape[-1])[:, :, 0, 0]
    shift = np.zeros(nawf)
    shift[1] = -0.2  # -U/2 on orbital 2 (1-based)
    np.testing.assert_allclose(np.diag(after - before).real, shift, atol=1e-12)
    np.testing.assert_allclose(p.data_controller.data_arrays['HubbardU'][:2], [0.0, 0.4])


def test_too_long_a_list_raises(tmp_path, monkeypatch):
    p = _model_run(tmp_path, monkeypatch)
    p.data_controller.data_attributes['abort_on_exception'] = True
    nawf = p.data_controller.data_arrays['HRs'].shape[0]
    with pytest.raises(ValueError, match='orbitals'):
        p.add_external_fields(HubbardU=[0.1] * (nawf + 1))
