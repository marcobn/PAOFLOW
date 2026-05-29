"""Unit tests for transmittance evaluation and interpolation."""

import numpy as np
import pytest

from PAOFLOW.transport.calculators.transmittance import (
    evaluate_transmittance,
    interpolate_transmittance,
)


@pytest.mark.unit
def test_interpolate_transmittance_returns_slice_when_no_division():
    """ndiv=1 should return the original window without interpolation."""
    egrid = np.array([0.0, 1.0, 2.0])
    transm = np.array([1.0, 2.0, 3.0])

    egrid_new, transm_new = interpolate_transmittance(egrid, transm, 0, 2, 1)

    np.testing.assert_allclose(egrid_new, egrid)
    np.testing.assert_allclose(transm_new, transm)


@pytest.mark.unit
def test_evaluate_transmittance_landauer_no_eigenchannels():
    """Landauer formula without eigenchannels should return diag of T-matrix."""
    gamma_L = np.eye(2)
    gamma_R = np.eye(2)
    g_ret = np.eye(2, dtype=complex)

    conduct, vecs = evaluate_transmittance(
        gamma_L,
        gamma_R,
        g_ret,
        formula='landauer',
        do_eigenchannels=False,
        do_eigplot=False,
    )

    np.testing.assert_allclose(conduct, [1.0, 1.0])
    assert vecs is None


@pytest.mark.unit
def test_evaluate_transmittance_generalized_requires_correlation():
    """The generalized formula requires a correlation self-energy."""
    gamma = np.eye(1)
    g_ret = np.eye(1, dtype=complex)

    with pytest.raises(AssertionError):
        evaluate_transmittance(
            gamma,
            gamma,
            g_ret,
            formula='generalized',
            do_eigenchannels=False,
            do_eigplot=False,
        )


@pytest.mark.unit
def test_evaluate_transmittance_eigenchannels():
    """Eigenchannel path should return channel eigenvalues for a simple case."""
    gamma_L = np.eye(2)
    gamma_R = np.eye(2)
    g_ret = np.eye(2, dtype=complex)

    conduct, vecs = evaluate_transmittance(
        gamma_L,
        gamma_R,
        g_ret,
        formula='landauer',
        do_eigenchannels=True,
        do_eigplot=False,
    )

    np.testing.assert_allclose(conduct, [1.0, 1.0])
    assert vecs is None


@pytest.mark.unit
def test_evaluate_transmittance_eigplot_returns_vectors():
    """Eigplot mode should return eigenvectors along with conductance values."""
    gamma_L = np.eye(2)
    gamma_R = np.eye(2)
    g_ret = np.eye(2, dtype=complex)

    conduct, vecs = evaluate_transmittance(
        gamma_L,
        gamma_R,
        g_ret,
        formula='landauer',
        do_eigenchannels=True,
        do_eigplot=True,
    )

    np.testing.assert_allclose(conduct, [1.0, 1.0])
    assert vecs is not None
    assert vecs.shape == (2, 2)
