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


@pytest.mark.unit
def test_complex_surface_green_matches_finite_chain() -> None:
    """Complex hopping and on-site SOC must agree with direct lead inversion."""
    onsite = np.array([[0.2, 0.04j], [-0.04j, -0.1]])
    hopping = np.array([[-0.3, 0.07 + 0.02j], [-0.01j, -0.2]])
    energy, broadening, cells = 1.3, 0.03, 40
    effective = energy * np.eye(2) - onsite
    transfer, _, iterations = compute_surface_transfer_matrices(
        effective, np.eye(2), hopping, broadening, transfer_thr=1e-26
    )
    assert iterations < 100
    surface = np.linalg.inv(effective + 1j * broadening * np.eye(2) - hopping @ transfer)
    finite = (
        np.kron(np.eye(cells), onsite)
        + np.kron(np.diag(np.ones(cells - 1), 1), hopping)
        + np.kron(np.diag(np.ones(cells - 1), -1), hopping.conj().T)
    )
    direct = np.linalg.inv((energy + 1j * broadening) * np.eye(2 * cells) - finite)[:2, :2]
    np.testing.assert_allclose(surface, direct, rtol=1e-12, atol=1e-13)
