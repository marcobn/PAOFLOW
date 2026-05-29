"""Unit tests for HamiltonianSystem blocks view and memory usage."""

import pytest

from PAOFLOW.transport.hamiltonian.hamiltonian import HamiltonianSystem


@pytest.mark.unit
def test_hamiltonian_system_blocks_property():
    """Blocks property should expose all expected block keys."""
    ham = HamiltonianSystem(dimL=1, dimC=1, dimR=1, nkpts_par=1)

    keys = set(ham.blocks.keys())
    assert keys == {'blc_00L', 'blc_01L', 'blc_00R', 'blc_01R', 'blc_00C', 'blc_LC', 'blc_CR'}
