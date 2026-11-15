"""Transport Eliashberg coupling, band velocities and plasma frequency on the dense grid."""

import numpy as np
import pytest

from PAOFLOW.elphon.elph_bloch import (
    RY_TO_EV,
    _diagonal_velocities,
    fermi_surface_plasma_frequency,
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
)
from PAOFLOW.response.drude import plasma_frequency_tensor


def _toy_cache(velocity_x):
    """Two-point (Nk = 2) one-band cache with a constant vertex and flat deltas."""
    nk = 2
    nkd = nk**3
    velocity = np.zeros((nkd, 1, 3))
    velocity[:, 0, 0] = velocity_x
    return {
        'Nk': nk,
        'nawf': 1,
        'V': np.ones((nkd, 1, 1), dtype=complex),
        'dk': np.ones((1, nkd, 1)),
        'dos': np.ones(1),
        'phkg': np.ones((nkd, 1), dtype=complex),
        'Midxg': np.zeros((1, 3), dtype=int),
        'Wg': np.ones(1),
        'ng_vertex': (1, 1, 1),
        'velocity_diag': velocity,
    }


def _run(cache, q):
    gR = np.ones((1, 1, 1, 1, 1, 1), dtype=complex)
    return lambda_q_dense_ws_fast(
        gR, cache, np.array(q), np.ones((1, 1)), np.array([2.0]), transport=True
    )


def test_transport_weight_limits():
    grid_x = np.repeat([0, 1], 4)  # first crystal index of the 2^3 grid (C order)
    # v(k + q) = v(k): no momentum relaxation -> lambda_tr = 0
    same = _run(_toy_cache(np.ones(8)), [0.5, 0.0, 0.0])
    assert same['lambda_qnu'][0, 0] > 0.0
    assert same['lambda_tr_qnu'][0, 0] == pytest.approx(0.0, abs=1e-12)
    # v(k + q) = -v(k): backscattering -> lambda_tr = 2 lambda
    flip = _run(_toy_cache(np.where(grid_x == 0, 1.0, -1.0)), [0.5, 0.0, 0.0])
    assert flip['lambda_tr_qnu'][0, 0] == pytest.approx(2.0 * flip['lambda_qnu'][0, 0])
    # without transport no extra key
    gR = np.ones((1, 1, 1, 1, 1, 1), dtype=complex)
    plain = lambda_q_dense_ws_fast(
        gR, _toy_cache(np.ones(8)), np.zeros(3), np.ones((1, 1)), np.array([2.0])
    )
    assert 'lambda_tr_qnu' not in plain
    with pytest.raises(ValueError):
        cache = _toy_cache(np.ones(8))
        del cache['velocity_diag']
        _run(cache, [0.5, 0.0, 0.0])


def _degenerate_hamiltonian(ng=4):
    """Two identical, decoupled s orbitals on a simple cubic lattice (degenerate bands)."""
    HRs = np.zeros((2, 2, ng, ng, ng, 1), dtype=complex)
    for orbital in range(2):
        HRs[orbital, orbital, 0, 0, 0, 0] = 0.5
        for axis in range(3):
            for sign in (1, -1):
                n = [0, 0, 0]
                n[axis] = sign % ng
                HRs[orbital, orbital, n[0], n[1], n[2], 0] = -1.0
    return HRs


def test_diagonal_velocities_gauge_invariant_and_plasma_frequency():
    electrons = precompute_dense_electrons(
        _degenerate_hamiltonian(), np.eye(3), 6, [0.05], None, (4, 4, 4),
        velocities=True, alat=1.0, diagonal_velocities=True,
    )  # fmt: skip
    reference = electrons['velocity_diag'].copy()
    # both bands are exactly degenerate: free-electron-like cos band velocity
    np.testing.assert_allclose(reference[:, 0], reference[:, 1], atol=1e-12)
    rotated = dict(electrons)
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
    rotated['V'] = electrons['V'] @ q  # another basis of the degenerate pair
    np.testing.assert_allclose(_diagonal_velocities(rotated, 64), reference, atol=1e-10)
    # dispersion E = 0.5 - 2 sum cos(2 pi k): dE/dk_x = 4 pi sin(2 pi k_x) (eV, alat = 1 bohr)
    K = electrons['K']
    np.testing.assert_allclose(
        reference[:, 0, 0] * RY_TO_EV, 2.0 * np.sin(2 * np.pi * K[:, 0]), atol=1e-10
    )
    # Rydberg (e-ph) and eV (generic) plasma frequencies agree
    volume = 1.0
    ry_version = fermi_surface_plasma_frequency(electrons, volume)
    ev_version = plasma_frequency_tensor(
        reference * RY_TO_EV, electrons['dk'][0] / RY_TO_EV, volume
    )
    np.testing.assert_allclose(ry_version, ev_version, rtol=1e-12, atol=1e-10)
