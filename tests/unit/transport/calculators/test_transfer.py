"""Unit tests for transfer-matrix calculations."""

import numpy as np
import pytest

from PAOFLOW.transport.calculators.transfer import compute_surface_transfer_matrices


@pytest.mark.unit
def test_compute_surface_transfer_matrices_zero_coupling_converges():
    """Zero coupling should converge immediately to zero transfer matrices."""
    h_eff = np.zeros((2, 2), dtype=complex)
    s_eff = np.eye(2, dtype=complex)
    t_coupling = np.zeros((2, 2), dtype=complex)

    tot, tott, niter = compute_surface_transfer_matrices(
        h_eff=h_eff,
        s_eff=s_eff,
        t_coupling=t_coupling,
        delta=1e-2,
        transfer_thr=1e-12,
    )

    np.testing.assert_allclose(tot, 0.0)
    np.testing.assert_allclose(tott, 0.0)
    assert niter == 1


@pytest.mark.unit
def test_compute_surface_transfer_matrices_singular_increments_fail_counter():
    """Singular initial matrices should increment the fail counter and return zeros."""
    h_eff = np.zeros((1, 1), dtype=complex)
    s_eff = np.zeros((1, 1), dtype=complex)
    t_coupling = np.zeros((1, 1), dtype=complex)
    fail_counter = {}

    tot, tott, niter = compute_surface_transfer_matrices(
        h_eff=h_eff,
        s_eff=s_eff,
        t_coupling=t_coupling,
        delta=0.0,
        fail_counter=fail_counter,
        fail_limit=5,
    )

    np.testing.assert_allclose(tot, 0.0)
    np.testing.assert_allclose(tott, 0.0)
    assert niter == 0
    assert fail_counter['nfail'] == 1
