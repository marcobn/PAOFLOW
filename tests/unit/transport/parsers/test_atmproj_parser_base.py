"""Unit tests for atomic projection parser helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.parsers.atmproj_parser_base import (
    parse_eigenvalues,
    parse_header,
    parse_kpoints,
    parse_overlaps,
    parse_projections,
)


class DummyDataController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


@pytest.mark.unit
def test_parse_header_extracts_attributes():
    """Header parsing should map required attributes to output keys."""
    _, attr = (
        {},
        {
            'nbnds': 2,
            'nkpnts': 3,
            'nspin': 1,
            'nawf': 4,
            'nelec': 8.0,
            'Efermi': 5.5,
            'energy_units': 'eV',
        },
    )
    controller = DummyDataController({}, attr)

    header = parse_header(controller)

    assert header['nbnds'] == 2
    assert header['efermi'] == 5.5
    assert header['energy_units'] == 'eV'


@pytest.mark.unit
def test_parse_kpoints_normalizes_weights():
    """K-point parser should normalize weights and return crystal coords."""
    kpnts = np.array([[0.0, 0.5, 0.0]])
    arry = {
        'kpnts': kpnts,
        'kpnts_wght': np.array([2.0]),
        'b_vectors': np.eye(3),
    }
    attr = {'alat': 2.0}
    controller = DummyDataController(arry, attr)

    data = parse_kpoints(controller)

    np.testing.assert_allclose(data['wk'], [1.0])
    np.testing.assert_allclose(data['vkpts_crystal'], kpnts.T)


@pytest.mark.unit
def test_parse_eigenvalues_and_projections():
    """Eigenvalues and projection matrices should be returned with expected shapes."""
    arry = {
        'my_eigsmat': np.zeros((2, 3, 1)),
        'U': np.zeros((2, 1, 3, 1), dtype=complex),
    }
    controller = DummyDataController(arry, {})

    eigvals = parse_eigenvalues(controller)
    proj = parse_projections(controller)

    assert eigvals.shape == (2, 3, 1)
    assert proj.shape == (1, 2, 3, 1)


@pytest.mark.unit
def test_parse_overlaps_respects_flag():
    """Overlaps should be returned only when overlap transformation is enabled."""
    arry = {'Sks': np.ones((2, 2, 1), dtype=complex)}
    controller = DummyDataController(arry, {})

    class DummyAtomic:
        do_overlap_transformation = True

    class DummyData:
        atomic_proj = DummyAtomic()

    overlaps = parse_overlaps(DummyData(), controller)

    assert overlaps is not None
