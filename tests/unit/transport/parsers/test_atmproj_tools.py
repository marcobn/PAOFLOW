"""Unit tests for atomic projection conversion helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.parsers.atmproj_tools import parse_atomic_proj_data, reshape_pao_hamiltonian


class DummyDataController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


class DummyConductorData:
    class DummyAtomic:
        do_overlap_transformation = False

    atomic_proj = DummyAtomic()


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


@pytest.mark.unit
def test_parse_atomic_proj_data_builds_model():
    """Atomic projection parser should combine header, k-point, and matrix data."""
    arry = {
        'kpnts': np.zeros((1, 3)),
        'kpnts_wght': np.array([1.0]),
        'b_vectors': np.eye(3),
        'my_eigsmat': np.zeros((1, 1, 1)),
        'U': np.zeros((1, 1, 1, 1), dtype=complex),
    }
    attr = {
        'nbnds': 1,
        'nkpnts': 1,
        'nspin': 1,
        'nawf': 1,
        'nelec': 1.0,
        'Efermi': 0.0,
        'energy_units': 'eV',
        'alat': 1.0,
    }

    data = parse_atomic_proj_data(DummyConductorData(), DummyDataController(arry, attr))

    assert data.nbnds == 1
    assert data.kpts.shape == (3, 1)
