"""Unit tests for atomic projection conversion helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.parsers.atmproj_tools import reshape_pao_hamiltonian


class DummyDataController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr

    def full_hamiltonian_k(self):
        return self._arry['Hks']


@pytest.mark.unit
def test_get_pao_hamiltonian_shapes():
    """Hamiltonian reshaping should produce (nspin, nkpnts, nawf, nawf) arrays."""
    nawf = 2
    nkpnts = 2
    nspin = 1
    hks = np.arange(nawf * nawf * nkpnts * nspin).reshape(nawf, nawf, 1, 1, nkpnts, nspin)
    hrs = np.arange(nawf * nawf * nkpnts * nspin).reshape(nawf, nawf, 1, 1, nkpnts, nspin)

    arry = {'Hks': hks, 'HRs': hrs}
    attr = {'nspin': nspin, 'nkpnts': nkpnts, 'nawf': nawf}

    data = reshape_pao_hamiltonian(DummyDataController(arry, attr))

    assert data['Hk'].shape == (nspin, nkpnts, nawf, nawf)
    assert data['HR'].shape == (nspin, nkpnts, nawf, nawf)
