"""Unit tests for real-to-kspace Fourier transforms."""

import numpy as np
import pytest

from PAOFLOW.transport.hamiltonian.fourier_par import fourier_transform_real_to_kspace


@pytest.mark.unit
def test_fourier_transform_real_to_kspace_simple():
    """Fourier transform should apply weights and phase factors per R-vector."""
    rh = np.array([[[1.0, 2.0]]], dtype=complex)
    wr = np.array([1.0, 0.5])
    table = np.array([[1.0, 1.0j], [1.0, -1.0j]])

    kh = fourier_transform_real_to_kspace(rh, wr, table)

    expected_k0 = wr[0] * table[0, 0] * rh[:, :, 0] + wr[1] * table[1, 0] * rh[:, :, 1]
    expected_k1 = wr[0] * table[0, 1] * rh[:, :, 0] + wr[1] * table[1, 1] * rh[:, :, 1]

    np.testing.assert_allclose(kh[:, :, 0], expected_k0)
    np.testing.assert_allclose(kh[:, :, 1], expected_k1)
