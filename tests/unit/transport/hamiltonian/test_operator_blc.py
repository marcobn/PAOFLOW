"""Unit tests for OperatorBlock allocation and utilities."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlock


@pytest.mark.unit
def test_operator_block_allocate_and_at_k():
    """Allocation should create arrays, and at_k should expose per-k views."""
    block = OperatorBlock('blk')
    block.allocate(dim1=2, dim2=2, nkpnts=1)
    block.H[:, :, 0] = np.array([[1.0, 2.0], [3.0, 4.0]])
    block.S[:, :, 0] = np.eye(2)

    view = block.at_k(0)

    np.testing.assert_allclose(view.H, block.H[:, :, 0])
    np.testing.assert_allclose(view.S, block.S[:, :, 0])


@pytest.mark.unit
def test_operator_block_update_requires_allocation():
    """Updating metadata on unallocated blocks should fail."""
    block = OperatorBlock('blk')

    with pytest.raises(RuntimeError):
        block.update(ie=1)


@pytest.mark.unit
def test_operator_block_copy_from_copies_arrays():
    """copy_from should clone allocated data and metadata."""
    source = OperatorBlock('src')
    source.allocate(dim1=1, dim2=1, nkpnts=1)
    source.H[:, :, 0] = 2.0
    source.S[:, :, 0] = 3.0

    target = OperatorBlock('dst')
    target.copy_from(source)

    np.testing.assert_allclose(target.H, source.H)
    np.testing.assert_allclose(target.S, source.S)
    assert target is not source


@pytest.mark.unit
def test_operator_block_memusage_invalid_type():
    """Unknown memory type strings should raise errors."""
    block = OperatorBlock('blk')
    block.allocate(dim1=1, dim2=1, nkpnts=1)

    with pytest.raises(ValueError):
        block.memusage(memtype='invalid')
