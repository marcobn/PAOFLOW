"""Tests for the per-shell change-of-basis matrices used by the
relativistic NL-velocity rotation (``tesseral_to_ylm_matrix`` and
``clebsch_jm_matrix`` in ``PAOFLOW.projection.do_atwfc_proj``)."""

import numpy as np
import pytest

from PAOFLOW.projection.do_atwfc_proj import (
    calc_ylmg_complex_0,
    calc_ylmg_so,
    clebsch_jm_matrix,
    tesseral_to_ylm_matrix,
)

ALL_L = [0, 1, 2, 3]
_BLOCK_YLM = {0: 0, 1: 1, 2: 4, 3: 9}
_BLOCK_JM = {0: 0, 1: 2, 2: 8, 3: 18}


@pytest.mark.parametrize('l', ALL_L)
def test_tesseral_to_ylm_unitary(l):
    n = 2 * l + 1
    U = tesseral_to_ylm_matrix(l)
    assert U.shape == (n, n)
    assert np.allclose(U @ U.conj().T, np.eye(n), atol=1e-14)
    assert np.allclose(U.conj().T @ U, np.eye(n), atol=1e-14)


@pytest.mark.parametrize('l', ALL_L)
def test_tesseral_to_ylm_matches_calc_ylmg_complex_0(l):
    """U applied to an arbitrary tesseral input reproduces calc_ylmg_complex_0."""
    rng = np.random.default_rng(42 + l)
    n = 2 * l + 1
    s = _BLOCK_YLM[l]
    ylmg = np.zeros((1, 16))
    ylmg[0, s : s + n] = rng.standard_normal(n)
    ylmgc_ref = calc_ylmg_complex_0(ylmg)[0, s : s + n]
    U = tesseral_to_ylm_matrix(l)
    ylmgc_check = U @ ylmg[0, s : s + n]
    assert np.allclose(ylmgc_check, ylmgc_ref, atol=1e-14)


@pytest.mark.parametrize('l', ALL_L)
def test_clebsch_jm_unitary(l):
    n = 2 * l + 1
    C = clebsch_jm_matrix(l)
    assert C.shape == (2 * n, 2 * n)
    assert np.allclose(C @ C.conj().T, np.eye(2 * n), atol=1e-14)
    assert np.allclose(C.conj().T @ C, np.eye(2 * n), atol=1e-14)


@pytest.mark.parametrize('l', ALL_L)
def test_clebsch_jm_matches_calc_ylmg_so(l):
    """C applied to a one-hot upper/lower spinor input reproduces calc_ylmg_so."""
    n = 2 * l + 1
    s = _BLOCK_YLM[l]
    jm0 = _BLOCK_JM[l]
    C = clebsch_jm_matrix(l)
    for sigma in (0, 1):  # 0 = upper, 1 = lower
        for m in range(n):
            ylmgc = np.zeros((1, 16), dtype=complex)
            ylmgc[0, s + m] = 1.0
            ref = calc_ylmg_so(ylmgc)[sigma, jm0 : jm0 + 2 * n]
            check = C[:, sigma * n + m]
            assert np.allclose(check, ref, atol=1e-14), f'l={l}, sigma={sigma}, m={m}'


@pytest.mark.parametrize('l', [1, 2, 3])
def test_clebsch_jm_row_sums_squared(l):
    """Each |chi_{j,m_j}> is a normalised CG superposition: row norm = 1."""
    C = clebsch_jm_matrix(l)
    norms = np.sum(np.abs(C) ** 2, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-14)


@pytest.mark.parametrize('l', [1, 2, 3])
def test_clebsch_jm_subshell_sizes(l):
    """First 2l columns are j=l-1/2 (size 2l), next 2(l+1) are j=l+1/2."""
    C = clebsch_jm_matrix(l)
    n_lower = 2 * l  # j = l - 1/2
    n_upper = 2 * l + 2  # j = l + 1/2
    assert C.shape[0] == n_lower + n_upper
