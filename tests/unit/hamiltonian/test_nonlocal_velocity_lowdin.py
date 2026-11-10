"""The Löwdin transformation, the orbital signs and the (j, m_j) rotation of the PAO catalog.

PAOFLOW works in the Löwdin basis ω = φ S^{-1/2} of the atomic orbitals: an operator assembled
between the raw orbitals is transformed as S^{-1/2} O S^{-1/2} (``lowdin_transform``) and
carries PAOFLOW's orbital signs (``pao_orbital_signs``); fully relativistic runs rotate the
scalar matrices to the (j, m_j) basis (``scalar_to_jm_matrix``).  These helpers serve the
position dipoles of :mod:`PAOFLOW.hamiltonian.position_dipoles`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
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

    pao = load_pao_orbitals(DC())
    a_cart = a_vectors * alat
    overlap_tables = build_position_tables(
        pao,
        enumerate_pao_pairs(pao, a_cart, pao_tol=1e-2),
        q_max=15.0,
        n_q=300,
        include_dipole=False,
    )
    return pao, overlap_tables, alat


def test_pao_orbital_signs_pattern(si_catalogs):
    pao, _, _ = si_catalogs
    signs = pao_orbital_signs(pao)
    # per site: s, p_z (m=0), p_x (m=+1), p_y (m=-1): (-1)^l times the odd-|m| sign
    assert signs.tolist() == [1.0, -1.0, 1.0, 1.0] * 2
    assert all(abs(x) == 1.0 for x in signs)


def test_overlap_only_tables_have_zero_dipoles(si_catalogs):
    _, overlap_tables, _ = si_catalogs
    assert all(np.all(block == 0.0) for block in overlap_tables.dipole)
    assert any(np.abs(block).max() > 0.1 for block in overlap_tables.overlap)


def test_lowdin_transform_of_signed_overlap_is_identity(si_catalogs):
    """S^{-1/2} S S^{-1/2} = 1: the Bloch overlap of the tables, in PAOFLOW's sign convention."""
    pao, overlap_tables, _ = si_catalogs
    k_cart = np.array([[0.11, -0.07, 0.19], [0.0, 0.0, 0.0]])
    overlap_k = assemble_overlap_kspace(overlap_tables, k_cart)
    signs = pao_orbital_signs(pao)
    overlap_pf = overlap_k * signs[None, :, None] * signs[None, None, :]
    out = lowdin_transform(overlap_pf, overlap_pf)
    np.testing.assert_allclose(out, np.repeat(np.eye(pao.total_nlm)[None], 2, 0), atol=1e-10)
    assert (
        np.abs(overlap_pf - np.eye(pao.total_nlm)[None]).max() > 1e-2
    )  # the basis is not orthonormal


def test_scalar_to_jm_overlap_is_hermitian_unitary_rotation(si_catalogs):
    pao, overlap_tables, _ = si_catalogs
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
