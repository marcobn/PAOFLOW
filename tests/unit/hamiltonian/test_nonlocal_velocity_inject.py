"""Phase 4 step 1: pin the in-place injection of Delta_pksp into dHksp.

The sign and overall prefactor of
:func:`PAOFLOW.hamiltonian.nonlocal_velocity.inject_into_dHksp` are
*provisional* — they reflect PAOFLOW's internal convention
``pksp = -<n|p|m>`` (verified by the cubium Hellmann-Feynman test) and
the Rydberg→eV conversion needed because ``dHksp`` is in eV·Bohr while
the operator returned by :func:`build_nonlocal_velocity_kspace` is in
Ry/Bohr (or Ha/Bohr).  These tests freeze that contract so the
calibration against ``epsilon.x`` in later Phase 4 work can change one
constant in one place and have it propagate consistently.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    RYDBERG_IN_EV,
    inject_into_dHksp,
)


def _random_arrays(rng, nktot=4, nawf=3, nspin=2):
    dH = rng.standard_normal((nktot, 3, nawf, nawf, nspin)) + 1j * rng.standard_normal(
        (nktot, 3, nawf, nawf, nspin)
    )
    dP = rng.standard_normal((nktot, 3, nawf, nawf)) + 1j * rng.standard_normal(
        (nktot, 3, nawf, nawf)
    )
    return dH, dP


def test_inject_rydberg_default_sign_subtracts_lambda_dP():
    rng = np.random.default_rng(0)
    dH, dP = _random_arrays(rng)
    dH_before = dH.copy()
    inject_into_dHksp(dH, dP)  # defaults: units='rydberg', sign=-1
    expected = dH_before - RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(dH, expected, rtol=0, atol=1e-12)


def test_inject_hartree_uses_factor_two():
    rng = np.random.default_rng(1)
    dH, dP = _random_arrays(rng)
    dH_before = dH.copy()
    inject_into_dHksp(dH, dP, units='hartree')
    expected = dH_before - 2.0 * RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(dH, expected, rtol=0, atol=1e-12)


def test_inject_positive_sign_adds():
    rng = np.random.default_rng(2)
    dH, dP = _random_arrays(rng)
    dH_before = dH.copy()
    inject_into_dHksp(dH, dP, sign=+1)
    expected = dH_before + RYDBERG_IN_EV * dP[..., None]
    np.testing.assert_allclose(dH, expected, rtol=0, atol=1e-12)


def test_inject_zero_dP_is_noop():
    rng = np.random.default_rng(3)
    dH, _ = _random_arrays(rng)
    dP = np.zeros((dH.shape[0], 3, dH.shape[2], dH.shape[3]), dtype=complex)
    dH_before = dH.copy()
    inject_into_dHksp(dH, dP)
    np.testing.assert_array_equal(dH, dH_before)


def test_inject_broadcasts_uniformly_across_spin():
    rng = np.random.default_rng(4)
    dH, dP = _random_arrays(rng, nspin=2)
    inject_into_dHksp(dH, dP)
    # difference between spin channels must be unchanged by the inject
    # since the correction is spin-diagonal at the projector level.
    diff = dH[..., 0] - dH[..., 1]
    dH_recomputed = dH.copy()
    inject_into_dHksp(dH_recomputed, -dP)  # undo + inject again
    inject_into_dHksp(dH_recomputed, dP)
    np.testing.assert_allclose(diff, dH_recomputed[..., 0] - dH_recomputed[..., 1])


def test_inject_is_in_place_and_returns_none():
    rng = np.random.default_rng(5)
    dH, dP = _random_arrays(rng)
    result = inject_into_dHksp(dH, dP)
    assert result is None


def test_inject_linearity_in_dP():
    rng = np.random.default_rng(6)
    dH0, dP = _random_arrays(rng)
    # f(a) := inject(dH0, a*dP) - dH0 should be linear in a.
    dH_a = dH0.copy()
    inject_into_dHksp(dH_a, 2.5 * dP)
    dH_b = dH0.copy()
    inject_into_dHksp(dH_b, dP)
    delta_a = dH_a - dH0
    delta_b = dH_b - dH0
    np.testing.assert_allclose(delta_a, 2.5 * delta_b, rtol=1e-12, atol=1e-12)


def test_inject_rejects_bad_units():
    dH = np.zeros((1, 3, 1, 1, 1), dtype=complex)
    dP = np.zeros((1, 3, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match='units'):
        inject_into_dHksp(dH, dP, units='eV')


def test_inject_rejects_bad_sign():
    dH = np.zeros((1, 3, 1, 1, 1), dtype=complex)
    dP = np.zeros((1, 3, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match='sign'):
        inject_into_dHksp(dH, dP, sign=0)


def test_inject_rejects_shape_mismatch():
    dH = np.zeros((2, 3, 4, 4, 1), dtype=complex)
    dP = np.zeros((2, 3, 3, 3), dtype=complex)  # wrong nawf
    with pytest.raises(ValueError, match='mismatch'):
        inject_into_dHksp(dH, dP)


def test_inject_rejects_bad_dHksp_rank():
    dH = np.zeros((1, 3, 1, 1), dtype=complex)  # missing spin axis
    dP = np.zeros((1, 3, 1, 1), dtype=complex)
    with pytest.raises(ValueError, match='dHksp'):
        inject_into_dHksp(dH, dP)


def test_inject_rejects_bad_delta_pksp_rank():
    dH = np.zeros((1, 3, 1, 1, 1), dtype=complex)
    dP = np.zeros((1, 3, 1, 1, 1), dtype=complex)  # 5-D, wrong
    with pytest.raises(ValueError, match='delta_pksp'):
        inject_into_dHksp(dH, dP)


def test_rydberg_in_ev_constant_matches_qe_value():
    # read_QE_xml uses Hart2eV = 2 * 13.60569193; our constant must round-
    # trip to the same eV/Ry within 1e-7.
    assert abs(RYDBERG_IN_EV - 13.605693122994) < 1e-12
    assert abs(2 * RYDBERG_IN_EV - 2 * 13.60569193) < 5e-6
