"""Memory-bounded k-blocks of the dense vertex kernel."""

from PAOFLOW.elphon.elph_bloch import DENSE_BLOCK_BYTES, dense_block_size


def test_primitive_cells_keep_the_historical_block():
    # Every primitive-cell case of the examples (<= 48 orbitals, <= 6 modes) fits the budget at 4096.
    assert dense_block_size(18, 3, 3) == 4096
    assert dense_block_size(48, 6, 6) == 4096


def test_large_supercells_shrink_the_block_to_the_budget():
    nawf, nmode = 144, 24
    kb = dense_block_size(nawf, nmode, nmode)
    assert 8 <= kb < 4096
    assert 3 * 16 * nawf * nawf * nmode * kb <= DENSE_BLOCK_BYTES
    assert dense_block_size(nawf, nmode, nmode, budget_bytes=1.0e6) == 8
    assert dense_block_size(nawf, nmode, nmode, budget_bytes=2 * DENSE_BLOCK_BYTES) >= 2 * kb
