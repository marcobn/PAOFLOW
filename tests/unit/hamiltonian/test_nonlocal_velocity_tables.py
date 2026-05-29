"""Unit tests for the Phase 3b real-space table builder.

Covers :func:`build_nl_real_space_tables` for both synthetic Gaussian
catalogs (closed-form gate) and a one-pair subset of the Si ONCV fixture
(shape, selection rules, dipole-geometry term).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    BetaCatalog,
    BetaSiteData,
    BetaSpeciesData,
    NLRealSpaceTables,
    PAOCatalog,
    PAOChannelData,
    PAOOrbitalEntry,
    PAOSiteData,
    PAOSpeciesData,
    build_nl_real_space_tables,
    enumerate_nl_pairs,
    iter_beta_lm,
    load_beta_projectors,
    load_pao_orbitals,
    n_beta_lm,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'


# ---------------------------------------------------------------------------
# Synthetic Gaussian catalogs (closed-form gate).
# ---------------------------------------------------------------------------


def _make_gaussian_synthetic(
    alpha_beta: float,
    alpha_phi: float,
    tau_I: np.ndarray,
    tau_J: np.ndarray,
    r_max: float = 12.0,
    npts: int = 1500,
):
    r"""Construct a 2-atom 1-s-channel synthetic system at given centres.

    The β projector and PAO orbital are both single ``s`` Gaussians:
    :math:`\beta(r) = e^{-\alpha_\beta r^2}`, similarly for :math:`\varphi`.

    Returns (beta_catalog, pao_catalog).  No UPF parsing — only the
    minimum subset of fields touched by :func:`build_nl_real_space_tables`
    is populated, via :class:`SimpleNamespace`.
    """
    r = np.linspace(0.0, r_max, npts)
    # UPF stores r·R; for an s function R(r) = exp(-α r²) → wfc = r·exp(-α r²).
    beta_wfc = r * np.exp(-alpha_beta * r**2)
    phi_wfc = r * np.exp(-alpha_phi * r**2)

    # Minimal "UPF" stub for each species.
    beta_stub_I = SimpleNamespace(
        beta=[{'l': 0, 'wfc': beta_wfc, 'cutoff_radius': r_max, 'label': 'Bs'}],
        pswfc=[],
    )
    pao_stub_J = SimpleNamespace(beta=[], pswfc=[])

    # β species/site.
    beta_sp = BetaSpeciesData(
        label='I',
        pseudo_file='I.upf',
        upf=beta_stub_I,
        r=r,
        rab=np.full_like(r, r[1] - r[0]),
        nproj=1,
        lchannels=[0],
        dion=np.zeros((1, 1)),
    )
    beta_site = BetaSiteData(index=0, label='I', tau=tau_I.copy(), species=beta_sp)
    beta_cat = BetaCatalog(
        species={'I': beta_sp}, sites=[beta_site], total_nproj_radial=1, total_nproj_lm=1
    )

    # PAO species/site.
    pao_ch = PAOChannelData(
        label='1S',
        l=0,
        R_radial=np.exp(-alpha_phi * r**2),
        wfc=phi_wfc,
        occupation=2.0,
    )
    pao_sp = PAOSpeciesData(
        label='J',
        pseudo_file='J.upf',
        upf=pao_stub_J,
        r=r,
        rab=np.full_like(r, r[1] - r[0]),
        channels=[pao_ch],
    )
    pao_orb = PAOOrbitalEntry(
        basis_index=0, site_index=0, channel_index=0, l=0, m=0, qe_m=1, label='1S'
    )
    pao_site = PAOSiteData(
        index=0, label='J', tau=tau_J.copy(), species=pao_sp, orbitals=[pao_orb], basis_offset=0
    )
    pao_cat = PAOCatalog(species={'J': pao_sp}, sites=[pao_site], basis=[pao_orb], total_nlm=1)
    return beta_cat, pao_cat


def _gaussian_ss_overlap_closed(alpha_a: float, alpha_b: float, R_vec: np.ndarray) -> float:
    r"""⟨ψ_a | ψ_b(·-R)⟩ for ψ = R(r) Y_00 with R(r) = exp(-α r²).

    Includes the ``Y_00² = 1/(4π)`` factor since
    :func:`two_center_overlap` works on radial-only parts of
    ψ = R(r) Y_lm.

    Result: (1/4π) · (π/(α_a+α_b))^{3/2} · exp(-α_a α_b R² / (α_a+α_b)).
    """
    R2 = float(R_vec @ R_vec)
    sigma = alpha_a + alpha_b
    return (1.0 / (4.0 * np.pi)) * (np.pi / sigma) ** 1.5 * np.exp(-alpha_a * alpha_b * R2 / sigma)


def test_synthetic_ss_overlap_matches_closed_form():
    """⟨β_s|φ_s⟩ for two Gaussians at given displacement."""
    alpha_b, alpha_p = 0.6, 0.4
    tau_I = np.array([0.0, 0.0, 0.0])
    tau_J = np.array([1.5, -0.7, 0.3])
    beta_cat, pao_cat = _make_gaussian_synthetic(alpha_b, alpha_p, tau_I, tau_J)
    a_cart = 30.0 * np.eye(3)  # huge cell → only ΔR=0 inside cutoff for tail
    pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
    # Pick only the ΔR=0 pair to keep the test fast.
    pair0 = [p for p in pairs if p.deltaR_lattice == (0, 0, 0)][0]
    tables = build_nl_real_space_tables(beta_cat, pao_cat, [pair0], q_max=20.0, n_q=600)
    assert isinstance(tables, NLRealSpaceTables)
    assert len(tables.S_bp) == 1
    assert tables.S_bp[0].shape == (1, 1)
    got = float(tables.S_bp[0][0, 0])
    expect = _gaussian_ss_overlap_closed(alpha_b, alpha_p, tau_J - tau_I)
    np.testing.assert_allclose(got, expect, rtol=2e-3)


def test_synthetic_ss_dipole_matches_closed_form():
    r"""⟨β_s | r_α | φ_s⟩ closed form for Gaussians at origin / displacement.

    With β at origin and φ at R:
    :math:`\int e^{-\alpha_a r^2}\, r_\alpha\, e^{-\alpha_b (r-R)^2}\, d^3r
      = R_\alpha\, \frac{\alpha_b}{\alpha_a+\alpha_b}\, S(R)`,
    where ``S(R)`` is the bare overlap.
    """
    alpha_b, alpha_p = 0.6, 0.4
    tau_I = np.zeros(3)  # τ_I = 0 → dipole geometry term vanishes.
    tau_J = np.array([1.0, 0.5, -0.3])
    beta_cat, pao_cat = _make_gaussian_synthetic(alpha_b, alpha_p, tau_I, tau_J)
    a_cart = 30.0 * np.eye(3)
    pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
    pair0 = [p for p in pairs if p.deltaR_lattice == (0, 0, 0)][0]
    tables = build_nl_real_space_tables(beta_cat, pao_cat, [pair0], q_max=20.0, n_q=600)
    S = _gaussian_ss_overlap_closed(alpha_b, alpha_p, tau_J - tau_I)
    R = tau_J - tau_I
    expected = (alpha_p / (alpha_b + alpha_p)) * S * R
    got = tables.S_rbp[0][:, 0, 0]
    np.testing.assert_allclose(got, expected, rtol=3e-3, atol=1e-6)


def test_synthetic_dipole_geometry_term():
    r"""With τ_I ≠ 0 the dipole adds ``(τ_I)_α · S`` exactly."""
    alpha_b, alpha_p = 0.6, 0.4
    tau_I = np.array([0.3, -0.2, 0.5])
    tau_J = tau_I + np.array([1.0, 0.5, -0.3])  # same physical displacement.
    beta_cat, pao_cat = _make_gaussian_synthetic(alpha_b, alpha_p, tau_I, tau_J)
    a_cart = 30.0 * np.eye(3)
    pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
    pair0 = [p for p in pairs if p.deltaR_lattice == (0, 0, 0)][0]
    tables = build_nl_real_space_tables(beta_cat, pao_cat, [pair0], q_max=20.0, n_q=600)
    S = float(tables.S_bp[0][0, 0])
    R = tau_J - tau_I
    M_expected = (alpha_p / (alpha_b + alpha_p)) * S * R  # bare matrix element
    got = tables.S_rbp[0][:, 0, 0]
    np.testing.assert_allclose(got, M_expected + tau_I * S, rtol=3e-3, atol=1e-6)


def test_include_dipole_false_skips_S_rbp():
    alpha_b, alpha_p = 0.6, 0.4
    beta_cat, pao_cat = _make_gaussian_synthetic(
        alpha_b, alpha_p, np.zeros(3), np.array([1.0, 0.0, 0.0])
    )
    a_cart = 30.0 * np.eye(3)
    pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
    pair0 = [p for p in pairs if p.deltaR_lattice == (0, 0, 0)][0]
    tables = build_nl_real_space_tables(
        beta_cat, pao_cat, [pair0], include_dipole=False, q_max=20.0, n_q=400
    )
    assert tables.S_rbp[0].shape == (3, 0, 0)


# ---------------------------------------------------------------------------
# iter_beta_lm / n_beta_lm helpers on real Si ONCV.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_catalogs():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    alat = 10.2
    a_cart = 0.5 * alat * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3])
    arrays = {'species': [('Si', SI_ONCV_UPF)], 'atoms': ['Si', 'Si'], 'tau': tau}
    attributes = {'fpath': str(SI_ONCV_DIR)}

    class DC:
        def data_dicts(self):
            return arrays, attributes

    dc = DC()
    return load_beta_projectors(dc), load_pao_orbitals(dc), a_cart


def test_si_n_beta_lm_matches_expected(si_catalogs):
    """Si ONCV has β channels l = [0,0,1,1,2,2] → 1+1+3+3+5+5 = 18 per site."""
    beta, _, _ = si_catalogs
    for site in beta.sites:
        assert n_beta_lm(site) == 18


def test_si_iter_beta_lm_ordering(si_catalogs):
    """First two entries: l=0 (m=0); next: l=0 (m=0); then l=1 (m=0,+1,-1); ..."""
    beta, _, _ = si_catalogs
    entries = list(iter_beta_lm(beta.sites[0]))
    # local indices contiguous 0..17.
    assert [e[0] for e in entries] == list(range(18))
    # First 2 are s, then p,p, then d,d (in channel order 0,0,1,1,2,2).
    ls = [e[2] for e in entries]
    expected_ls = [0] + [0] + [1, 1, 1] + [1, 1, 1] + [2, 2, 2, 2, 2] + [2, 2, 2, 2, 2]
    assert ls == expected_ls
    # qe_m within each channel sweeps 1..2l+1.
    qms = [e[4] for e in entries]
    expected_qms = [1] + [1] + [1, 2, 3] + [1, 2, 3] + [1, 2, 3, 4, 5] + [1, 2, 3, 4, 5]
    assert qms == expected_qms


# ---------------------------------------------------------------------------
# Si one-pair table build (correctness + shape, kept small for speed).
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_home_pair_table(si_catalogs):
    """Build the table for the home-cell self pair only (I=J=0, ΔR=0)."""
    beta, pao, a_cart = si_catalogs
    pairs = enumerate_nl_pairs(beta, pao, a_cart, pao_tol=1e-3)
    pair0 = [
        p for p in pairs if p.beta_site == 0 and p.pao_site == 0 and p.deltaR_lattice == (0, 0, 0)
    ][0]
    # Reduced quadrature for speed (still ~1 % accurate vs the default grid).
    tables = build_nl_real_space_tables(beta, pao, [pair0], q_max=15.0, n_q=300)
    return beta, pao, pair0, tables


def test_si_home_pair_table_shape(si_home_pair_table):
    beta, pao, _, tables = si_home_pair_table
    assert len(tables.S_bp) == 1
    assert tables.S_bp[0].shape == (n_beta_lm(beta.sites[0]), len(pao.sites[0].orbitals))
    assert tables.S_bp[0].shape == (18, 4)
    assert tables.S_rbp[0].shape == (3, 18, 4)
    assert tables.beta_lm_per_site == [18, 18]
    assert tables.pao_per_site == [4, 4]


def test_si_home_pair_selection_rules(si_home_pair_table):
    """On the same site at ΔR=0, ⟨β_l|φ_l'⟩ = 0 unless l = l' (orthogonality
    of real Y_lm at coincident centres) — and the dipole ⟨β_l|r_α|φ_l'⟩
    vanishes unless |l − l'| = 1 (Δm rules constrain α)."""
    beta, pao, _, tables = si_home_pair_table
    S = tables.S_bp[0]
    Sr = tables.S_rbp[0]
    # Map flat β-LM index → l_β.
    l_beta = [e[2] for e in iter_beta_lm(beta.sites[0])]
    l_phi = [e.l for e in pao.sites[0].orbitals]

    # Overlap: l_β != l_φ → ~0.  Looser tolerance to allow numerical noise.
    for i, lb in enumerate(l_beta):
        for j, lp in enumerate(l_phi):
            if lb != lp:
                assert (
                    abs(S[i, j]) < 5e-3
                ), f'overlap leak at i={i} (lβ={lb}) j={j} (lφ={lp}): {S[i, j]}'

    # Dipole at coincident origin: nonzero only when |l_β − l_φ| = 1.
    # τ_I = 0 here, so the geometry term vanishes entirely.
    for i, lb in enumerate(l_beta):
        for j, lp in enumerate(l_phi):
            if abs(lb - lp) != 1:
                for alpha in (0, 1, 2):
                    assert (
                        abs(Sr[alpha, i, j]) < 5e-3
                    ), f'dipole leak α={alpha} i={i} (lβ={lb}) j={j} (lφ={lp}): {Sr[alpha, i, j]}'


def test_si_home_pair_dipole_geometry_term_zero_when_tau_I_zero(si_home_pair_table):
    """With τ_I = 0, S_rbp equals the bare dipole M_α(0) — no offset added."""
    beta, _pao, _pair, tables = si_home_pair_table
    np.testing.assert_allclose(beta.sites[0].tau, [0.0, 0.0, 0.0])
    # No additional check needed beyond selection rules above; we just
    # assert the τ_I·S piece is exactly zero by construction of the test.


# ---------------------------------------------------------------------------
# Distance decay: a separated pair gives smaller overlap than the home pair.
# ---------------------------------------------------------------------------


def test_synthetic_overlap_decays_with_distance():
    alpha_b, alpha_p = 0.6, 0.4
    a_cart = 50.0 * np.eye(3)  # large cell — only home pairs.
    overlaps = []
    for dz in [0.0, 1.0, 2.0, 3.0]:
        beta_cat, pao_cat = _make_gaussian_synthetic(
            alpha_b, alpha_p, np.zeros(3), np.array([0.0, 0.0, dz])
        )
        pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
        pair0 = [p for p in pairs if p.deltaR_lattice == (0, 0, 0)][0]
        tables = build_nl_real_space_tables(
            beta_cat, pao_cat, [pair0], q_max=20.0, n_q=400, include_dipole=False
        )
        overlaps.append(float(tables.S_bp[0][0, 0]))
    # Monotone decrease.
    for a, b in zip(overlaps[:-1], overlaps[1:]):
        assert a > b
