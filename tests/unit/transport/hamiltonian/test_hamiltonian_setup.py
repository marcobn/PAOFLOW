"""Unit tests for auxiliary Hamiltonian setup at each (E, k)."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock


def _make_block(name: str, with_corr: bool = True) -> OperatorBlock:
    block = OperatorBlock(name)
    block.allocate(dim1=1, dim2=1, nkpnts=1, ne_sgm=2, lhave_corr=with_corr)
    block.H[:, :, 0] = 2.0
    block.S[:, :, 0] = 1.0
    if with_corr:
        block.sgm[:, :, 0, 0] = 0.5
        block.sgm[:, :, 0, 1] = 0.25
    return block


@pytest.mark.unit
def test_hamiltonian_setup_applies_shifts_and_sgm():
    """Auxiliary matrices should include shifts and correlation terms."""
    blc_00C = _make_block('blc_00C')
    blc_01L = _make_block('blc_01L')

    egrid = np.array([1.0])

    hamiltonian_setup(
        ik=0,
        ie_g=0,
        egrid=egrid,
        shift_L=0.2,
        shift_C=0.1,
        shift_R=0.3,
        shift_C_corr=0.05,
        blc_blocks={'blc_00C': blc_00C, 'blc_01L': blc_01L},
        ie_buff=0,
    )

    expected_c = (1.0 - 0.1) * 1.0 - 2.0 - 0.5 - 0.05 * 1.0
    expected_l = (1.0 - 0.2) * 1.0 - 2.0 - 0.5

    np.testing.assert_allclose(blc_00C.aux[:, :, 0], np.conj([[expected_c]]))
    np.testing.assert_allclose(blc_01L.aux[:, :, 0], [[expected_l]])

    assert blc_00C.ie == 0
    assert blc_00C.ik == 0
    assert blc_00C.ie_buff == 0


@pytest.mark.unit
def test_hamiltonian_setup_preserves_complex_onsite_block() -> None:
    """On-site spin-orbit entries retain their sign in E*S - H."""
    block = OperatorBlock('blc_00C')
    block.allocate(dim1=2, dim2=2, nkpnts=1, ne_sgm=0, lhave_corr=False)
    hamiltonian = np.array([[0.2, 0.3j], [-0.3j, -0.4]])
    overlap = np.array([[1.0, 0.05j], [-0.05j, 1.0]])
    block.H[:, :, 0], block.S[:, :, 0] = hamiltonian, overlap
    hamiltonian_setup(
        ik=0,
        ie_g=0,
        egrid=np.array([0.7]),
        shift_L=0.0,
        shift_C=0.1,
        shift_R=0.0,
        blc_blocks={'blc_00C': block},
    )
    np.testing.assert_allclose(block.aux[:, :, 0], 0.6 * overlap - hamiltonian)
