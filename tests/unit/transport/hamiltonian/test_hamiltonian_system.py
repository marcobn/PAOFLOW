"""Unit tests for HamiltonianSystem container behavior."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.hamiltonian import HamiltonianSystem


@pytest.mark.unit
def test_hamiltonian_system_allocate_sets_blocks():
    """Allocation should size all blocks and set tags from the provided dictionary."""
    ham = HamiltonianSystem(dimL=1, dimC=2, dimR=1, nkpts_par=1)
    ivr_par = np.zeros((2, 1), dtype=int)
    tags = {'block_00C': {'rows': 'all', 'cols': 'all'}}

    ham.allocate(ivr_par, tags)

    assert ham.blc_00C.allocated
    assert ham.blc_00C.tag['rows'] == 'all'


@pytest.mark.unit
def test_hamiltonian_system_memusage_positive():
    """memusage should report a positive value after allocation."""
    ham = HamiltonianSystem(dimL=1, dimC=1, dimR=1, nkpts_par=1)
    ham.allocate(np.zeros((2, 1), dtype=int), {})

    assert ham.memusage() > 0.0


@pytest.mark.unit
def test_hamiltonian_system_allocate_rejects_invalid_dims():
    """Allocation should reject non-positive sizes."""
    ham = HamiltonianSystem(dimL=0, dimC=1, dimR=1, nkpts_par=1)

    with pytest.raises(ValueError):
        ham.allocate(np.zeros((2, 1), dtype=int), {})
