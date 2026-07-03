"""Unit tests for Green's function calculations."""

import numpy as np
import pytest

from PAOFLOW.transport.calculators import green as green_module


@pytest.mark.unit
def test_compute_surface_green_function_right_surface():
    """Right surface Green's function matches explicit matrix inversion."""
    h_eff = np.diag([2.0, 3.0]).astype(complex)
    s_eff = np.eye(2, dtype=complex)
    t_coupling = np.zeros((2, 2), dtype=complex)
    transfer = np.zeros((2, 2), dtype=complex)

    g = green_module.compute_surface_green_function(
        h_eff=h_eff,
        s_eff=s_eff,
        t_coupling=t_coupling,
        transfer_matrix=transfer,
        transfer_matrix_conj=transfer,
        igreen=1,
        delta=1e-3,
    )

    expected = np.linalg.inv(h_eff + 1j * 1e-3 * s_eff)
    np.testing.assert_allclose(g, expected)


@pytest.mark.unit
def test_compute_surface_green_function_invalid_igreen():
    """Invalid igreen values should raise a ValueError."""
    h_eff = np.eye(1, dtype=complex)
    s_eff = np.eye(1, dtype=complex)
    t_coupling = np.zeros((1, 1), dtype=complex)
    transfer = np.zeros((1, 1), dtype=complex)

    with pytest.raises(ValueError):
        green_module.compute_surface_green_function(
            h_eff=h_eff,
            s_eff=s_eff,
            t_coupling=t_coupling,
            transfer_matrix=transfer,
            transfer_matrix_conj=transfer,
            igreen=2,
            delta=1e-3,
        )


@pytest.mark.unit
def test_compute_surface_green_function_singular_raises():
    """Singular matrices should raise a RuntimeError during inversion."""
    h_eff = np.zeros((1, 1), dtype=complex)
    s_eff = np.zeros((1, 1), dtype=complex)
    t_coupling = np.zeros((1, 1), dtype=complex)
    transfer = np.zeros((1, 1), dtype=complex)

    with pytest.raises(RuntimeError):
        green_module.compute_surface_green_function(
            h_eff=h_eff,
            s_eff=s_eff,
            t_coupling=t_coupling,
            transfer_matrix=transfer,
            transfer_matrix_conj=transfer,
            igreen=1,
            delta=0.0,
        )


@pytest.mark.unit
def test_compute_conductor_green_function_surface_uses_only_left(monkeypatch):
    """Surface mode should only subtract the left self-energy."""
    monkeypatch.setattr(
        green_module,
        'compute_non_interacting_gf',
        lambda **kwargs: np.eye(2, dtype=complex),
    )
    sigma_l = np.diag([0.2, 0.1]).astype(complex)

    g_c = green_module.compute_conductor_green_function(
        blc_00C=None,
        sigma_l=sigma_l,
        surface=True,
    )

    expected = np.linalg.inv(np.eye(2) - sigma_l)
    np.testing.assert_allclose(g_c, expected)


@pytest.mark.unit
def test_compute_conductor_green_function_requires_sigma_r(monkeypatch):
    """Non-surface mode requires both left and right self-energies."""
    monkeypatch.setattr(
        green_module,
        'compute_non_interacting_gf',
        lambda **kwargs: np.eye(1, dtype=complex),
    )

    with pytest.raises(ValueError):
        green_module.compute_conductor_green_function(
            blc_00C=None,
            sigma_l=np.array([[0.1 + 0.0j]]),
            sigma_r=None,
            surface=False,
        )


@pytest.mark.unit
def test_compute_conductor_green_function_with_both_leads(monkeypatch):
    """With both leads, the conductor Green's function inverts (G0^-1 - Sigma)."""
    monkeypatch.setattr(
        green_module,
        'compute_non_interacting_gf',
        lambda **kwargs: np.eye(2, dtype=complex),
    )
    sigma_l = np.diag([0.2, 0.1]).astype(complex)
    sigma_r = np.diag([0.1, 0.3]).astype(complex)

    g_c = green_module.compute_conductor_green_function(
        blc_00C=None,
        sigma_l=sigma_l,
        sigma_r=sigma_r,
        surface=False,
    )

    expected = np.linalg.inv(np.eye(2) - sigma_l - sigma_r)
    np.testing.assert_allclose(g_c, expected)
