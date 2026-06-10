"""Phase 3c tests: k-space non-local velocity correction.

Verifies the assembly of P_I(k), P^α_I(k), and the operator

    Δp_α(k) = (m/iℏ) Σ_I [P_I†·D^I·P^α_I − (P^α_I)†·D^I·P_I]

from real-space ⟨β|φ⟩ / ⟨β|r_α|φ⟩ tables.  Uses the Si ONCV setup
that is already exercised by the table-builder tests.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    _dion_lm_expanded,
    assemble_beta_projections_k,
    build_nl_real_space_tables,
    build_nonlocal_velocity_kspace,
    enumerate_nl_pairs,
    iter_beta_lm,
    load_beta_projectors,
    load_pao_orbitals,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _DC:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


def _si_diamond_dc():
    """Build a minimal Si diamond DataController-like object."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    fpath = os.path.join(repo_root, 'examples/qe_examples/example16/Si_ONCV')
    alat = 10.2
    a_cart = 0.5 * alat * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3])
    arry = {
        'species': [('Si', 'Si_ONCV_PBE_sr.UPF')],
        'atoms': ['Si', 'Si'],
        'tau': tau,
        'a_vectors': a_cart / alat,
    }
    attr = {'fpath': fpath, 'alat': alat}
    return _DC(arry, attr), a_cart


@pytest.fixture(scope='module')
def si_tables():
    dc, a_cart = _si_diamond_dc()
    b = load_beta_projectors(dc)
    p = load_pao_orbitals(dc)
    pairs = enumerate_nl_pairs(b, p, a_cart, pao_tol=1e-2)
    t = build_nl_real_space_tables(b, p, pairs, q_max=15.0, n_q=300)
    return b, p, pairs, t, a_cart


# ---------------------------------------------------------------------------
# _dion_lm_expanded
# ---------------------------------------------------------------------------


def test_dion_lm_expanded_si_block_diagonal(si_tables):
    b, _p, _pairs, _t, _a = si_tables
    site0 = b.sites[0]
    D_lm = _dion_lm_expanded(site0)
    # Si ONCV: l-channels = [0,0,1,1,2,2] → n_lm = 1+1+3+3+5+5 = 18
    assert D_lm.shape == (18, 18)
    # Symmetric
    np.testing.assert_allclose(D_lm, D_lm.T, atol=1e-14)
    # Within same (l, m) the value must match D_rad[i,j].
    D_rad = site0.species.dion
    lchan = site0.species.lchannels
    # Pick the first s-s pair → diag of LM-block at index (0,0)
    assert D_lm[0, 0] == pytest.approx(D_rad[0, 0])
    # And the off-diagonal s1-s2 coupling (lchan[0]=lchan[1]=0)
    assert D_lm[0, 1] == pytest.approx(D_rad[0, 1])
    # m mixing must vanish: l=1, m=0 ↔ m=1 channel
    # Find offsets for first p-channel
    offs = []
    cur = 0
    for l in lchan:
        offs.append(cur)
        cur += 2 * l + 1
    p1_off = offs[2]  # first p channel
    assert D_lm[p1_off, p1_off + 1] == 0.0
    # cross-l coupling vanishes: s-channel ↔ p-channel
    assert D_lm[0, p1_off] == 0.0


def test_dion_lm_expanded_matches_iter_beta_lm_order(si_tables):
    """The LM expansion order must match :func:`iter_beta_lm`."""
    b, _p, _pairs, _t, _a = si_tables
    site = b.sites[0]
    D_lm = _dion_lm_expanded(site)
    lm_tuples = list(iter_beta_lm(site))
    # For each (local, ch_idx, l, m_std, qe_m): D_lm diagonal value equals D_rad[ch,ch]
    D_rad = site.species.dion
    for local, ch_idx, _l, _m, _qe_m in lm_tuples:
        assert D_lm[local, local] == pytest.approx(D_rad[ch_idx, ch_idx])


# ---------------------------------------------------------------------------
# assemble_beta_projections_k
# ---------------------------------------------------------------------------


def test_P_at_gamma_is_sum_of_real_blocks(si_tables):
    """At k = 0 the phase factors are all 1, so P_I(0) = Σ_pairs(I→·) S_bp."""
    b, p, pairs, t, _a = si_tables
    k = np.zeros((1, 3))
    P_list, Pa_list = assemble_beta_projections_k(b, p, t, k)

    nsites = len(b.sites)
    nawf = p.total_nlm
    P_expected = [np.zeros((t.beta_lm_per_site[I], nawf), dtype=complex) for I in range(nsites)]
    Pa_expected = [np.zeros((3, t.beta_lm_per_site[I], nawf), dtype=complex) for I in range(nsites)]
    pao_offsets = [s.basis_offset for s in p.sites]
    for pair, S, Sr in zip(pairs, t.S_bp, t.S_rbp):
        I = pair.beta_site
        J = pair.pao_site
        off = pao_offsets[J]
        sz = S.shape[1]
        P_expected[I][:, off : off + sz] += S
        Pa_expected[I][:, :, off : off + sz] += Sr

    for I in range(nsites):
        np.testing.assert_allclose(P_list[I][0], P_expected[I], atol=1e-14)
        np.testing.assert_allclose(Pa_list[I][0], Pa_expected[I], atol=1e-14)
        # All-real at Γ
        np.testing.assert_allclose(P_list[I][0].imag, 0.0, atol=1e-14)
        np.testing.assert_allclose(Pa_list[I][0].imag, 0.0, atol=1e-14)


def test_P_minus_k_conj_of_P_k(si_tables):
    """Real S-blocks → P(-k) = conj(P(k)) and same for Pa."""
    b, p, _pairs, t, _a = si_tables
    rng = np.random.default_rng(1)
    k = rng.normal(size=(4, 3)) * 0.3
    P_pos, Pa_pos = assemble_beta_projections_k(b, p, t, k)
    P_neg, Pa_neg = assemble_beta_projections_k(b, p, t, -k)
    for I in range(len(P_pos)):
        np.testing.assert_allclose(P_neg[I], np.conj(P_pos[I]), atol=1e-12)
        np.testing.assert_allclose(Pa_neg[I], np.conj(Pa_pos[I]), atol=1e-12)


def test_P_shapes(si_tables):
    b, p, _pairs, t, _a = si_tables
    k = np.zeros((3, 3))
    P, Pa = assemble_beta_projections_k(b, p, t, k)
    assert len(P) == len(b.sites)
    for I in range(len(P)):
        assert P[I].shape == (3, t.beta_lm_per_site[I], p.total_nlm)
        assert Pa[I].shape == (3, 3, t.beta_lm_per_site[I], p.total_nlm)


def test_assemble_rejects_no_dipole_tables(si_tables):
    b, p, pairs, _t, _a = si_tables
    t_nod = build_nl_real_space_tables(b, p, pairs, q_max=15.0, n_q=300, include_dipole=False)
    with pytest.raises(ValueError, match='include_dipole'):
        assemble_beta_projections_k(b, p, t_nod, np.zeros((1, 3)))


def test_assemble_rejects_bad_k_shape(si_tables):
    b, p, _pairs, t, _a = si_tables
    with pytest.raises(ValueError, match='shape'):
        assemble_beta_projections_k(b, p, t, np.zeros(3))


# ---------------------------------------------------------------------------
# build_nonlocal_velocity_kspace
# ---------------------------------------------------------------------------


def test_dP_is_hermitian(si_tables):
    b, p, _pairs, t, _a = si_tables
    rng = np.random.default_rng(0)
    k = rng.normal(size=(5, 3)) * 0.4
    dP = build_nonlocal_velocity_kspace(b, p, t, k)
    assert dP.shape == (5, 3, p.total_nlm, p.total_nlm)
    for ik in range(5):
        for alpha in range(3):
            M = dP[ik, alpha]
            np.testing.assert_allclose(M, np.conj(M.T), atol=1e-12)


def test_dP_TRS_negate_and_conj(si_tables):
    """For a TRS-symmetric system in a real basis: dP(-k) = -conj(dP(k))."""
    b, p, _pairs, t, _a = si_tables
    rng = np.random.default_rng(2)
    k = rng.normal(size=(3, 3)) * 0.5
    dP_pos = build_nonlocal_velocity_kspace(b, p, t, k)
    dP_neg = build_nonlocal_velocity_kspace(b, p, t, -k)
    np.testing.assert_allclose(dP_neg, -np.conj(dP_pos), atol=1e-12)


def test_dP_at_gamma_is_imag(si_tables):
    """At Γ, P and Pa are real → bracket is purely imaginary → −i·bracket is real.
    But the bracket itself [A − A†] for real A is `(A − A.T)` (anti-symmetric real),
    so −i·(real anti-sym) is purely **imaginary** and Hermitian."""
    b, p, _pairs, t, _a = si_tables
    dP = build_nonlocal_velocity_kspace(b, p, t, np.zeros((1, 3)))
    # Real part must vanish (the bracket is real anti-symmetric).
    np.testing.assert_allclose(dP.real, 0.0, atol=1e-12)
    # Hermitian + purely imaginary → imag part is anti-symmetric.
    M = dP[0, 0]
    np.testing.assert_allclose(M.imag, -M.imag.T, atol=1e-12)


def test_dP_rydberg_factor_two(si_tables):
    b, p, _pairs, t, _a = si_tables
    k = np.array([[0.1, 0.2, -0.05]])
    dP_h = build_nonlocal_velocity_kspace(b, p, t, k, units='hartree')
    dP_r = build_nonlocal_velocity_kspace(b, p, t, k, units='rydberg')
    np.testing.assert_allclose(dP_r, 2.0 * dP_h, atol=1e-14)


def test_dP_rejects_bad_units(si_tables):
    b, p, _pairs, t, _a = si_tables
    with pytest.raises(ValueError, match='units'):
        build_nonlocal_velocity_kspace(b, p, t, np.zeros((1, 3)), units='SI')


def test_dP_zero_when_dion_zero(si_tables):
    """Setting D=0 must zero out the entire correction."""
    b, p, _pairs, t, _a = si_tables
    # Patch each site's species.dion to zeros (don't mutate the originals).
    saved = {}
    for sp in b.species.values():
        saved[sp.label] = sp.dion.copy()
        sp.dion = np.zeros_like(sp.dion)
    try:
        dP = build_nonlocal_velocity_kspace(b, p, t, np.array([[0.1, 0.2, 0.3]]))
        np.testing.assert_allclose(dP, 0.0, atol=1e-14)
    finally:
        for sp in b.species.values():
            sp.dion = saved[sp.label]


def test_dP_linear_in_dion(si_tables):
    """Δp is linear in D: scaling D by λ scales Δp by λ."""
    b, p, _pairs, t, _a = si_tables
    k = np.array([[0.1, -0.2, 0.05]])
    dP1 = build_nonlocal_velocity_kspace(b, p, t, k)
    saved = {sp.label: sp.dion.copy() for sp in b.species.values()}
    try:
        for sp in b.species.values():
            sp.dion = sp.dion * 3.0
        dP3 = build_nonlocal_velocity_kspace(b, p, t, k)
        np.testing.assert_allclose(dP3, 3.0 * dP1, rtol=1e-12, atol=1e-14)
    finally:
        for sp in b.species.values():
            sp.dion = saved[sp.label]
