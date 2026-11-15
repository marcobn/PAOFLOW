"""The Bloch-derivative engine against the exact atomic-gauge operators of hand-built models."""

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import (
    PAIRS_AB,
    bloch_terms_from_HRs,
    hamiltonian_derivatives_block,
    velocity_and_curvature_operators,
)

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    '_nonlinear_models', Path(__file__).resolve().parents[1] / 'response' / '_nonlinear_models.py'
)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
honeycomb_model, random_hopping_model = _models.honeycomb_model, _models.random_hopping_model


@pytest.mark.parametrize(
    'model',
    [
        random_hopping_model(seed=1),
        random_hopping_model(seed=2, time_reversal=False),
        honeycomb_model(),
    ],
)
def test_velocity_and_curvature_match_exact_atomic_gauge(model):
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    rng = np.random.default_rng(7)
    kpts = rng.uniform(-0.5, 0.5, (5, 3))
    if model.ng[2] == 1:
        kpts[:, 2] = 0.0
    H, dH, d2H = hamiltonian_derivatives_block(terms, kpts)
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    for ik, k in enumerate(kpts):
        HA, dHA, d2HA = model.atomic_gauge(k)
        D = model.phases(k)
        # lattice -> atomic gauge: H^A = D^+ H D, and the same for v and w
        assert np.allclose(D.conj().T @ H[ik] @ D, HA, atol=1e-10)
        for a in range(3):
            assert np.allclose(D.conj().T @ v[ik, a] @ D, dHA[a], atol=1e-10)
            for b in range(3):
                assert np.allclose(D.conj().T @ w[ik, a, b] @ D, d2HA[a, b], atol=1e-10)
        assert np.allclose(w[ik], np.swapaxes(w[ik], 0, 1))


def test_derivatives_match_finite_differences_of_the_atomic_hamiltonian():
    model = random_hopping_model(seed=3, time_reversal=False)
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    k0 = np.array([0.13, -0.27, 0.31])
    H, dH, d2H = hamiltonian_derivatives_block(terms, k0[None])
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    D = model.phases(k0)
    step = 1.0e-4  # 1/bohr
    bg_cart = np.linalg.inv(model.at).T
    for a in range(3):
        dk = np.zeros(3)
        dk[a] = step
        dk_cryst = (dk * model.alat / (2 * np.pi)) @ np.linalg.inv(
            bg_cart
        )  # cartesian step -> crystal
        hp = model.atomic_gauge(k0 + dk_cryst)[0]
        hm = model.atomic_gauge(k0 - dk_cryst)[0]
        fd = (hp - hm) / (2 * step)
        assert np.allclose(D.conj().T @ v[0, a] @ D, fd, atol=1e-6)
        for b in range(3):
            dkb = np.zeros(3)
            dkb[b] = step
            dkb_cryst = (dkb * model.alat / (2 * np.pi)) @ np.linalg.inv(bg_cart)
            hpp = model.atomic_gauge(k0 + dk_cryst + dkb_cryst)[0]
            hpm = model.atomic_gauge(k0 + dk_cryst - dkb_cryst)[0]
            hmp = model.atomic_gauge(k0 - dk_cryst + dkb_cryst)[0]
            hmm = model.atomic_gauge(k0 - dk_cryst - dkb_cryst)[0]
            fd2 = (hpp - hpm - hmp + hmm) / (4 * step * step)
            assert np.allclose(D.conj().T @ w[0, a, b] @ D, fd2, atol=1e-5)


def test_pairs_and_cutoff_and_origin_positions():
    model = random_hopping_model(seed=4)
    assert len(PAIRS_AB) == 6
    full = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    cut = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau, hopping_cutoff=1e-3)
    assert cut.cells.shape[0] <= full.cells.shape[0]
    k = np.array([[0.2, 0.1, -0.3]])
    H_full = hamiltonian_derivatives_block(full, k)[0]
    H_cut = hamiltonian_derivatives_block(cut, k)[0]
    assert np.allclose(H_full, H_cut, atol=5e-3)
    origin = bloch_terms_from_HRs(model.HRs, model.at, model.alat, None)
    assert np.allclose(origin.separation_bohr, 0.0)
    e_full = np.linalg.eigvalsh(H_full)
    e_origin = np.linalg.eigvalsh(hamiltonian_derivatives_block(origin, k)[0])
    assert np.allclose(e_full, e_origin, atol=1e-10)  # the spectrum does not depend on the gauge
