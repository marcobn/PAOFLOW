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


def test_shifts_by_manifold_label():
    """A dict of manifolds resolves on the internal basis records (any case), not on indices."""
    from PAOFLOW.hamiltonian.add_ext_field import hubbard_shifts_from_labels

    records = [{'atom': atom, 'label': label} for atom, label in
               [('Al', '3S'), ('Al', '3P'), ('Al', '3P'), ('Al', '3P'), ('P', '3S'), ('P', '3P')]]  # fmt: skip
    values = hubbard_shifts_from_labels({'atomic_basis': records}, {'al-3p': 0.1, 'P-3P': 2.31})
    np.testing.assert_allclose(values, [0.0, 0.1, 0.1, 0.1, 0.0, 2.31])
    with pytest.raises(ValueError, match='Al-3S, Al-3P, P-3S, P-3P'):
        hubbard_shifts_from_labels({'atomic_basis': records}, {'Al-3d': 1.0})
    with pytest.raises(ValueError, match='internal projections'):
        hubbard_shifts_from_labels({}, {'Al-3p': 1.0})


def test_label_dict_on_a_model_without_basis_is_refused(tmp_path, monkeypatch):
    p = _model_run(tmp_path, monkeypatch)
    p.data_controller.data_attributes['abort_on_exception'] = True
    with pytest.raises(ValueError, match='internal projections'):
        p.add_external_fields(HubbardU={'C-2p': 0.1})
