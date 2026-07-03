"""Unit tests for read_matrix block population."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock
from PAOFLOW.transport.parsers.read_matrix import read_matrix


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
def test_read_matrix_populates_h_and_s():
    """read_matrix should build k-space H/S for a simple onsite block."""
    nawf = 2
    hr = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=complex)

    arry = {
        'ivr': np.array([[0, 0, 0]]),
        'HRs': hr,
    }
    attr = {'nawf': nawf, 'nspin': 1}

    controller = DummyDataController(arry, attr)

    block = OperatorBlock('block_00C')
    block.allocate(dim1=2, dim2=2, nkpnts=1)
    block.ivr_par = np.zeros((2, 1), dtype=int)
    block.wr_par = np.ones(1)
    block.table_par = np.ones((1, 1), dtype=complex)

    read_matrix(
        yaml_data=DummyConductorData(),
        data_controller=controller,
        ispin=0,
        transport_direction=3,
        opr=block,
    )

    np.testing.assert_allclose(block.H[:, :, 0], hr[0].T)
    np.testing.assert_allclose(block.S[:, :, 0], np.eye(2))


@pytest.mark.unit
def test_read_matrix_requires_spin_for_polarized():
    """Spin-polarized runs should reject unspecified ispin values."""
    arry = {'ivr': np.array([[0, 0, 0]]), 'HRs': np.zeros((1, 1, 1), dtype=complex)}
    attr = {'nawf': 1, 'nspin': 2}
    controller = DummyDataController(arry, attr)

    block = OperatorBlock('block_00C')
    block.allocate(dim1=1, dim2=1, nkpnts=1)
    block.ivr_par = np.zeros((2, 1), dtype=int)
    block.wr_par = np.ones(1)
    block.table_par = np.ones((1, 1), dtype=complex)

    with pytest.raises(ValueError):
        read_matrix(
            yaml_data=DummyConductorData(),
            data_controller=controller,
            ispin=-1,
            transport_direction=3,
            opr=block,
        )
