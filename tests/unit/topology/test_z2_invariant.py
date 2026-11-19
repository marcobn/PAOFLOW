"""Z2 index from the Wannier charge centres (``PAOFLOW.z2_invariant``) on the Kane-Mele model."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('delta, expected', [(0.2, 1), (2.0, 0)])
def test_kane_mele_phases(tmp_path, monkeypatch, delta, expected):
    """Quantum spin Hall for a small staggered potential, trivial beyond 3 sqrt(3) lambda_SO."""
    pytest.importorskip('z2pack')
    pytest.importorskip('tbmodels')
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import kane_mele_model

    monkeypatch.chdir(tmp_path)
    p = PAOFLOW.PAOFLOW(model=kane_mele_model(delta=delta), outputdir='output', verbose=False)
    result = p.z2_invariant()
    assert result == {'nocc': 2, 'planes': [0.0], 'z2': [expected]}
    wcc = np.loadtxt(tmp_path / 'output' / 'z2_wcc_k3_0.dat')
    assert wcc.shape[1] == 3 and wcc[0, 0] == 0.0 and wcc[-1, 0] == 1.0
    assert 'Hks' not in p.data_controller.data_arrays  # rebuilt for the file only


def test_needs_spin_orbit(tmp_path, monkeypatch):
    from PAOFLOW.topology.z2_invariant import do_z2_invariant

    class Controller:
        def data_dicts(self):
            return {}, {'dftSO': False, 'nelec': 2.0}

    with pytest.raises(ValueError, match='spin-orbit'):
        do_z2_invariant(Controller())
