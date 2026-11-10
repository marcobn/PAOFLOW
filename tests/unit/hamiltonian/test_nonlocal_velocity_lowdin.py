"""The Löwdin transformation and the orbital signs of the non-local velocity correction.

``build_nonlocal_velocity_kspace`` assembles ⟨φ_a|m/(iħ)[V_NL, r]|φ_b⟩ between the raw atomic
orbitals; PAOFLOW works in the Löwdin basis ω = φ S^{-1/2}, so the drivers pass the Bloch
overlap S(k) of the orbitals and the operator is transformed as S^{-1/2} Δp S^{-1/2}
(``lowdin_transform``), then carries PAOFLOW's orbital signs (``pao_orbital_signs``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    build_nl_real_space_tables,
    build_nonlocal_velocity_kspace,
    compute_nonlocal_velocity_on_grid,
    enumerate_nl_pairs,
    load_beta_projectors,
    load_pao_orbitals,
    lowdin_transform,
    pao_orbital_signs,
    scalar_to_jm_matrix,
)
from PAOFLOW.hamiltonian.position_dipoles import (
    assemble_overlap_kspace,
    build_position_tables,
    enumerate_pao_pairs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'


def _random_hermitian(rng, nk, n, cartesian=True):
    shape = (nk, 3, n, n) if cartesian else (nk, n, n)
    m = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return m + np.conj(np.swapaxes(m, -1, -2))


def _random_overlap(rng, nk, n):
    a = rng.normal(size=(nk, n, n)) + 1j * rng.normal(size=(nk, n, n))
    return a @ np.conj(np.swapaxes(a, -1, -2)) + 0.5 * np.eye(n)[None]


def test_lowdin_transform_identity_overlap_is_identity():
    rng = np.random.default_rng(0)
    m = _random_hermitian(rng, 2, 5)
    np.testing.assert_allclose(lowdin_transform(m, np.repeat(np.eye(5)[None], 2, 0)), m)


def test_lowdin_transform_matches_explicit_inverse_sqrt_and_keeps_hermiticity():
    rng = np.random.default_rng(1)
    s = _random_overlap(rng, 3, 6)
    m = _random_hermitian(rng, 3, 6)
    out = lowdin_transform(m, s)
    for k in range(3):
        w, u = np.linalg.eigh(s[k])
        inv_sqrt = (u / np.sqrt(w)) @ u.conj().T
        np.testing.assert_allclose(out[k], inv_sqrt[None] @ m[k] @ inv_sqrt[None], atol=1e-10)
    np.testing.assert_allclose(out, np.conj(np.swapaxes(out, -1, -2)), atol=1e-10)
    # a (nk, n, n) operator works the same way
    np.testing.assert_allclose(lowdin_transform(m[:, 0], s), out[:, 0], atol=1e-12)


def test_lowdin_transform_rejects_singular_overlap():
    with pytest.raises(ValueError, match='positive definite'):
        lowdin_transform(np.zeros((1, 3, 2, 2), complex), np.zeros((1, 2, 2), complex))


@pytest.fixture(scope='module')
def si_catalogs():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    alat = 10.2
    a_vectors = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3])
    arrays = {'species': [('Si', SI_ONCV_UPF)], 'atoms': ['Si', 'Si'], 'tau': tau}
    attributes = {'fpath': str(SI_ONCV_DIR)}

    class DC:
        def data_dicts(self):
            return arrays, attributes

    dc = DC()
    beta, pao = load_beta_projectors(dc), load_pao_orbitals(dc)
    a_cart = a_vectors * alat
    nl_tables = build_nl_real_space_tables(
        beta, pao, enumerate_nl_pairs(beta, pao, a_cart, pao_tol=1e-2), q_max=15.0, n_q=300
    )
    overlap_tables = build_position_tables(
        pao,
        enumerate_pao_pairs(pao, a_cart, pao_tol=1e-2),
        q_max=15.0,
        n_q=300,
        include_dipole=False,
    )
    return beta, pao, nl_tables, overlap_tables, alat


def test_pao_orbital_signs_pattern(si_catalogs):
    _, pao, _, _, _ = si_catalogs
    signs = pao_orbital_signs(pao)
    # per site: s, p_z (m=0), p_x (m=+1), p_y (m=-1): (-1)^l times the odd-|m| sign
    assert signs.tolist() == [1.0, -1.0, 1.0, 1.0] * 2
    assert all(abs(x) == 1.0 for x in signs)


def test_overlap_only_tables_have_zero_dipoles(si_catalogs):
    _, _, _, overlap_tables, _ = si_catalogs
    assert all(np.all(block == 0.0) for block in overlap_tables.dipole)
    assert any(np.abs(block).max() > 0.1 for block in overlap_tables.overlap)


def test_transformed_operator_is_lowdin_of_raw_and_hermitian(si_catalogs):
    beta, pao, nl_tables, overlap_tables, alat = si_catalogs
    k_cart = np.array([[0.11, -0.07, 0.19], [0.0, 0.0, 0.0]])
    raw = build_nonlocal_velocity_kspace(beta, pao, nl_tables, k_cart, units='rydberg')
    overlap_k = assemble_overlap_kspace(overlap_tables, k_cart)
    full = build_nonlocal_velocity_kspace(
        beta, pao, nl_tables, k_cart, units='rydberg', overlap_k=overlap_k
    )
    # ``raw`` already carries PAOFLOW's orbital signs; the overlap of the tables must be
    # brought to the same convention before the two routes can be compared
    signs = pao_orbital_signs(pao)
    overlap_pf = overlap_k * signs[None, :, None] * signs[None, None, :]
    np.testing.assert_allclose(full, lowdin_transform(raw, overlap_pf), atol=1e-12)
    np.testing.assert_allclose(full, np.conj(np.swapaxes(full, -1, -2)), atol=1e-12)
    assert np.abs(full - raw).max() > 1e-3  # the transformation is not a no-op for this basis
    # the grid wrapper passes the overlap through
    kgrid = (k_cart * alat / (2.0 * np.pi)).T  # (3, nk) in 2 pi / alat
    via_grid = compute_nonlocal_velocity_on_grid(
        beta, pao, nl_tables, kgrid, alat, units='rydberg', overlap_tables=overlap_tables
    )
    np.testing.assert_allclose(via_grid, full, atol=1e-12)
    with pytest.raises(ValueError, match='overlap_k must have shape'):
        build_nonlocal_velocity_kspace(
            beta, pao, nl_tables, k_cart, units='rydberg', overlap_k=overlap_k[:1]
        )


def test_scalar_to_jm_overlap_is_hermitian_unitary_rotation(si_catalogs):
    _, pao, _, overlap_tables, _ = si_catalogs
    k_cart = np.array([[0.11, -0.07, 0.19]])
    s = assemble_overlap_kspace(overlap_tables, k_cart)
    s_jm = scalar_to_jm_matrix(s, pao)
    assert s_jm.shape == (1, 2 * pao.total_nlm, 2 * pao.total_nlm)
    np.testing.assert_allclose(s_jm, np.conj(np.swapaxes(s_jm, -1, -2)), atol=1e-12)
    # a unitary rotation of S (+) S keeps the eigenvalues (each doubled)
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(s_jm[0])),
        np.sort(np.repeat(np.linalg.eigvalsh(s[0]), 2)),
        atol=1e-10,
    )
