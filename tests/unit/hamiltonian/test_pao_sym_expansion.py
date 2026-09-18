import numpy as np
import pytest

from PAOFLOW.hamiltonian.pao_sym import get_full_grid, open_grid

NK = 4
SHELLS = np.array([0, 1])  # s + p
NAWF = int(np.sum(2 * SHELLS + 1))


def _cubic_inversion_system(seed=1234):
    """One atom at the origin in a simple cubic cell, symmetry {E, I}."""
    full_grid = get_full_grid(NK, NK, NK, 0, 0, 0)

    def fold(k):
        return ((k % 1.0) + 0.5) % 1.0 - 0.5

    seen, keep = set(), []
    for i, k in enumerate(full_grid):
        if tuple(np.round(fold(-k), 6) + 0.0) in seen:
            continue
        seen.add(tuple(np.round(fold(k), 6) + 0.0))
        keep.append(i)

    rng = np.random.default_rng(seed)
    shape = (len(keep), NAWF, NAWF)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return {
        'Hksp': a + np.conj(np.transpose(a, (0, 2, 1))),
        'full_grid': full_grid,
        'kp': np.ascontiguousarray(full_grid[keep]),
        'symop': np.array([np.eye(3), -np.eye(3)]),
        'sym_TR': np.array([False, False]),
        'equiv_atom': np.zeros((2, 1), dtype=int),
        'atom_pos': np.zeros((1, 3)),
    }


def _expand(sysd, symm_grid, npool=1):
    return open_grid(
        sysd['Hksp'].copy(),
        sysd['full_grid'],
        sysd['kp'].copy(),
        sysd['symop'],
        sysd['symop'].copy(),
        sysd['atom_pos'],
        SHELLS,
        np.zeros(NAWF, dtype=int),
        sysd['equiv_atom'],
        NK,
        NK,
        NK,
        0,
        0,
        0,
        False,
        sysd['sym_TR'],
        None,
        False,
        symm_grid,
        1.0e-6,
        4,
        False,
        npool,
    )


@pytest.mark.parametrize('symm_grid', [False, True])
def test_open_grid_fills_the_full_bz(symm_grid):
    sysd = _cubic_inversion_system()
    assert sysd['kp'].shape[0] < sysd['full_grid'].shape[0]  # wedge is genuinely reduced

    Hks_full = _expand(sysd, symm_grid)

    assert Hks_full.shape == (NK**3, NAWF, NAWF)
    assert np.isfinite(Hks_full).all()


@pytest.mark.parametrize('symm_grid', [False, True])
def test_open_grid_output_is_hermitian(symm_grid):
    Hks_full = _expand(_cubic_inversion_system(), symm_grid)

    for ik in range(Hks_full.shape[0]):
        block = Hks_full[ik]
        assert np.allclose(block, np.conj(block.T), atol=1e-12)


def test_open_grid_is_independent_of_npool():
    """npool only chunks the scatter/gather messages; results must not move."""
    sysd = _cubic_inversion_system()
    reference = _expand(sysd, symm_grid=False)

    for npool in (2, 3, 4):
        assert np.array_equal(_expand(sysd, symm_grid=False, npool=npool), reference)
