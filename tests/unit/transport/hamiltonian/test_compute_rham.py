"""Unit tests for real-space Hamiltonian construction."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham


@pytest.mark.unit
def test_compute_rham_zero_vector_is_weighted_sum():
    """For R=0, the Fourier phase is 1 and the result is a weighted sum of H(k)."""
    rvec = np.zeros(3)
    kpts = np.zeros((3, 2))
    wk = np.array([0.25, 0.75])
    hk = np.array(
        [
            [[1.0, 0.0], [0.0, 2.0]],
            [[2.0, 1.0], [1.0, 3.0]],
        ],
        dtype=complex,
    )

    hr = compute_rham(rvec, hk, kpts, wk)

    expected = wk[0] * hk[0] + wk[1] * hk[1]
    np.testing.assert_allclose(hr, expected)
