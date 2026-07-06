"""Unit tests for Hamiltonian initialization helpers."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.hamiltonian import HamiltonianSystem
from PAOFLOW.transport.hamiltonian.hamiltonian_init import (
    check_leads_are_identical,
    initialize_hamiltonian_blocks,
)


class DummyConductorData:
    def __init__(self):
        self.hamiltonian_tags = {}


@pytest.mark.unit
def test_initialize_hamiltonian_blocks_bulk_symmetrizes(monkeypatch):
    """Bulk initialization should copy blocks and symmetrize H and S."""
    ham = HamiltonianSystem(dimL=2, dimC=2, dimR=2, nkpts_par=1)
    ivr_par3d = np.zeros((3, 2), dtype=int)
    wr_par = np.ones(2)
    table_par = np.ones((2, 1), dtype=complex)

    def fake_read_matrix(conductor_data, data_controller, ispin, transport_direction, block):
        block.H[:, :, 0] = np.array([[1.0, 2.0], [0.0, 3.0]])
        block.S[:, :, 0] = np.array([[1.0, 0.5], [0.0, 1.0]])

    monkeypatch.setattr(
        'PAOFLOW.transport.hamiltonian.hamiltonian_init.read_matrix',
        fake_read_matrix,
    )

    initialize_hamiltonian_blocks(
        output_dir='.',
        ham_system=ham,
        ivr_par3D=ivr_par3d,
        wr_par=wr_par,
        table_par=table_par,
        ispin=0,
        transport_direction='z',
        calculation_type='bulk',
        data_controller=object(),
        conductor_data=DummyConductorData(),
    )

    expected_H = 0.5 * (np.array([[1.0, 2.0], [0.0, 3.0]]) + np.array([[1.0, 0.0], [2.0, 3.0]]))
    expected_S = 0.5 * (np.array([[1.0, 0.5], [0.0, 1.0]]) + np.array([[1.0, 0.0], [0.5, 1.0]]))

    np.testing.assert_allclose(ham.blc_00C.H[:, :, 0], expected_H)
    np.testing.assert_allclose(ham.blc_00L.H[:, :, 0], expected_H)
    np.testing.assert_allclose(ham.blc_00R.H[:, :, 0], expected_H)
    np.testing.assert_allclose(ham.blc_00C.S[:, :, 0], expected_S)


@pytest.mark.unit
def test_check_leads_are_identical_uses_sgm_files_and_arrays():
    """Lead identity check should compare self-energy filenames and index arrays."""
    ham = HamiltonianSystem(dimL=1, dimC=1, dimR=1, nkpts_par=1)
    ham.allocate(np.zeros((2, 1), dtype=int), {})

    # Freshly allocated leads share identical (zero) index arrays.
    assert check_leads_are_identical(ham) is True

    # Differing self-energy datafiles make the leads non-identical.
    assert check_leads_are_identical(ham, datafile_L_sgm='L', datafile_R_sgm='R') is False

    # Differing index arrays make the leads non-identical.
    ham.blc_00R.irows = ham.blc_00R.irows + 1
    assert check_leads_are_identical(ham) is False
