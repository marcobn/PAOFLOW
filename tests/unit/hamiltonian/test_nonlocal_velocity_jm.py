"""Unit tests for the Option A relativistic ``(j, m_j)`` non-local velocity.

Covers the j-coupled spinor builders added to
:mod:`PAOFLOW.hamiltonian.nonlocal_velocity`:

* :func:`_jm_shell_transform` — per-shell RSH-tesseral ⊗ spin → (j, m_j)
  unitary;
* :func:`_beta_jm_channel_info` / :func:`_dion_jm_expanded` — j-resolved D
  expansion;
* :func:`build_nonlocal_velocity_jm_kspace` — Hermiticity, time-reversal
  symmetry, and the Rydberg = 2 × Hartree scaling.

These tests use a synthetic two-atom ``p``-shell system (one j=1/2 and one
j=3/2 β per site, one ``p`` PAO channel) so they have no external-fixture
dependency.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    BetaCatalog,
    BetaSiteData,
    BetaSpeciesData,
    PAOCatalog,
    PAOChannelData,
    PAOOrbitalEntry,
    PAOSiteData,
    PAOSpeciesData,
    _beta_jm_channel_info,
    _dion_jm_expanded,
    _jm_shell_transform,
    build_nl_real_space_tables,
    build_nonlocal_velocity_jm_kspace,
    enumerate_nl_pairs,
    qe_m_index_to_std,
)

# ---------------------------------------------------------------------------
# _jm_shell_transform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_jm_shell_transform_unitary(l):
    T = _jm_shell_transform(l)
    n = 2 * (2 * l + 1)
    assert T.shape == (n, n)
    err = np.abs(T @ T.conj().T - np.eye(n)).max()
    assert err < 1e-12


def test_jm_shell_transform_s_is_spin_swap():
    """For l=0 the (j,m_j) basis is the pure spin basis, but reordered.

    The j=1/2 shell is stored with ``m_j`` ascending, i.e. row 0 is
    ``m_j=-1/2`` (= spin-down) and row 1 is ``m_j=+1/2`` (= spin-up), while
    the spin (column) basis is ordered (up, down).  The transform is
    therefore the real 2x2 anti-diagonal swap, not the identity.
    """
    T = _jm_shell_transform(0)
    np.testing.assert_allclose(T, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


# ---------------------------------------------------------------------------
# β channel info / D^{jm} expansion
# ---------------------------------------------------------------------------


def _make_p_beta_site():
    """β site with one ``p`` shell split into j=1/2 and j=3/2 channels."""
    sp = BetaSpeciesData(
        label='X',
        pseudo_file='X.upf',
        upf=SimpleNamespace(beta=[], pswfc=[]),
        r=np.linspace(0.0, 1.0, 4),
        rab=np.full(4, 1.0 / 3.0),
        nproj=2,
        lchannels=[1, 1],
        dion=np.diag([2.0, 5.0]),
        jbeta=[0.5, 1.5],
        has_spinorbit=True,
    )
    return BetaSiteData(index=0, label='X', tau=np.zeros(3), species=sp)


def test_beta_jm_channel_info_layout():
    info = _beta_jm_channel_info(_make_p_beta_site())
    # (channel_idx, l, j, s_row0, slot0, jrow0, n_m, n_mj)
    assert info[0] == (0, 1, 0.5, 0, 0, 0, 3, 2)  # j=l-1/2 → jrow0=0, 2 slots
    assert info[1] == (1, 1, 1.5, 3, 2, 2, 3, 4)  # j=l+1/2 → jrow0=2l=2, 4 slots


def test_dion_jm_expanded_diagonal_expansion():
    D = _dion_jm_expanded(_make_p_beta_site())
    # 2 (j=1/2) + 4 (j=3/2) = 6 slots.
    assert D.shape == (6, 6)
    # Hermitian / symmetric.
    np.testing.assert_allclose(D, D.T, atol=1e-14)
    # j=1/2 block carries D_rad[0,0]=2.0 on its 2 m_j slots.
    np.testing.assert_allclose(np.diag(D)[:2], [2.0, 2.0])
    # j=3/2 block carries D_rad[1,1]=5.0 on its 4 m_j slots.
    np.testing.assert_allclose(np.diag(D)[2:], [5.0, 5.0, 5.0, 5.0])
    # No cross (l,j) coupling.
    off = D - np.diag(np.diag(D))
    assert np.abs(off).max() == 0.0


def test_beta_jm_channel_info_requires_j():
    sp = BetaSpeciesData(
        label='Y',
        pseudo_file='Y.upf',
        upf=SimpleNamespace(beta=[], pswfc=[]),
        r=np.linspace(0.0, 1.0, 4),
        rab=np.full(4, 1.0 / 3.0),
        nproj=1,
        lchannels=[1],
        dion=np.diag([1.0]),
        jbeta=[None],
        has_spinorbit=False,
    )
    site = BetaSiteData(index=0, label='Y', tau=np.zeros(3), species=sp)
    with pytest.raises(RuntimeError):
        _beta_jm_channel_info(site)


# ---------------------------------------------------------------------------
# Synthetic relativistic p-shell system for Δp tests
# ---------------------------------------------------------------------------


def _make_rel_p_system(tau_I, tau_J, r_max=12.0, npts=1500):
    r"""Two-atom system: β site I has a ``p`` shell (j=1/2 & j=3/2);
    PAO site J has a single ``p`` channel (3 tesseral orbitals).

    Both radial parts are Gaussians; the actual numbers are irrelevant for
    the symmetry tests, only the index bookkeeping matters.
    """
    r = np.linspace(0.0, r_max, npts)
    alpha_b, alpha_p = 0.6, 0.4
    beta_wfc = r * np.exp(-alpha_b * r**2)  # p: wfc = r·R, R~exp(-αr²)
    phi_wfc = r * np.exp(-alpha_p * r**2)

    beta_stub = SimpleNamespace(
        beta=[
            {'l': 1, 'wfc': beta_wfc, 'cutoff_radius': r_max, 'label': 'p12', 'j': 0.5},
            {'l': 1, 'wfc': beta_wfc, 'cutoff_radius': r_max, 'label': 'p32', 'j': 1.5},
        ],
        pswfc=[],
    )
    beta_sp = BetaSpeciesData(
        label='I',
        pseudo_file='I.upf',
        upf=beta_stub,
        r=r,
        rab=np.full_like(r, r[1] - r[0]),
        nproj=2,
        lchannels=[1, 1],
        dion=np.diag([2.0, 5.0]),
        jbeta=[0.5, 1.5],
        has_spinorbit=True,
    )
    beta_site = BetaSiteData(index=0, label='I', tau=tau_I.copy(), species=beta_sp)
    beta_cat = BetaCatalog(
        species={'I': beta_sp}, sites=[beta_site], total_nproj_radial=2, total_nproj_lm=6
    )

    pao_ch = PAOChannelData(
        label='2P', l=1, R_radial=np.exp(-alpha_p * r**2), wfc=phi_wfc, occupation=2.0
    )
    pao_sp = PAOSpeciesData(
        label='J',
        pseudo_file='J.upf',
        upf=SimpleNamespace(beta=[], pswfc=[]),
        r=r,
        rab=np.full_like(r, r[1] - r[0]),
        channels=[pao_ch],
    )
    # QE stores the 2l+1 real harmonics 1-indexed (qe_m = 1, 2, 3); the
    # two-center machinery needs the standard m in [-l, l].
    orbs = [
        PAOOrbitalEntry(
            basis_index=i,
            site_index=0,
            channel_index=0,
            l=1,
            m=qe_m_index_to_std(i + 1, 1),
            qe_m=i + 1,
            label='2P',
        )
        for i in range(3)
    ]
    pao_site = PAOSiteData(
        index=0, label='J', tau=tau_J.copy(), species=pao_sp, orbitals=orbs, basis_offset=0
    )
    pao_cat = PAOCatalog(species={'J': pao_sp}, sites=[pao_site], basis=orbs, total_nlm=3)
    return beta_cat, pao_cat


def _build_tables(beta_cat, pao_cat, a_cart):
    pairs = enumerate_nl_pairs(beta_cat, pao_cat, a_cart, pao_tol=1e-6)
    return build_nl_real_space_tables(beta_cat, pao_cat, pairs, q_max=20.0, n_q=400)


def test_jm_delta_p_hermitian():
    a_cart = 12.0 * np.eye(3)
    beta_cat, pao_cat = _make_rel_p_system(np.zeros(3), np.array([1.3, -0.7, 0.4]))
    tables = _build_tables(beta_cat, pao_cat, a_cart)
    kpts = np.array([[0.11, -0.23, 0.07], [0.3, 0.1, -0.2]])
    dP = build_nonlocal_velocity_jm_kspace(beta_cat, pao_cat, tables, kpts, units='hartree')
    n_rel = 2 * pao_cat.total_nlm
    assert dP.shape == (2, 3, n_rel, n_rel)
    for ik in range(2):
        for a in range(3):
            M = dP[ik, a]
            assert np.abs(M - M.conj().T).max() < 1e-10


def _time_reversal_matrix_p_shell():
    r"""Kramers time-reversal operator :math:`\Theta = U_T K` for one ``p``
    shell in the (j, m_j) basis used by the builder.

    The basis is ordered j=1/2 block first (m_j ascending), then j=3/2
    block (m_j ascending)::

        (1/2,-1/2) (1/2,+1/2) (3/2,-3/2) (3/2,-1/2) (3/2,+1/2) (3/2,+3/2)

    Time reversal acts as :math:`\Theta|j,m_j\rangle = (-1)^{j-m_j}
    |j,-m_j\rangle`, so ``U_T`` flips ``m_j -> -m_j`` within each j block
    with the half-integer Kramers phase.
    """
    states = [(0.5, -0.5), (0.5, 0.5), (1.5, -1.5), (1.5, -0.5), (1.5, 0.5), (1.5, 1.5)]
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    U_T = np.zeros((n, n))
    for i, (j, mj) in enumerate(states):
        U_T[idx[(j, -mj)], i] = (-1.0) ** (j - mj)
    return U_T


def test_jm_delta_p_time_reversal():
    r"""Kramers time-reversal: in the spinor (j, m_j) basis the velocity
    correction obeys :math:`\Delta p_\alpha(-k) = -U_T\,\Delta p_\alpha(k)^*\,
    U_T^\dagger` with :math:`\Theta = U_T K` the half-integer time-reversal
    operator.  (The naive :math:`-\Delta p(k)^*` only holds for a spinless
    real basis and is *not* the correct relation for a spinor basis.)
    """
    a_cart = 12.0 * np.eye(3)
    beta_cat, pao_cat = _make_rel_p_system(np.zeros(3), np.array([1.3, -0.7, 0.4]))
    tables = _build_tables(beta_cat, pao_cat, a_cart)
    k = np.array([[0.21, 0.13, -0.05]])
    dP_p = build_nonlocal_velocity_jm_kspace(beta_cat, pao_cat, tables, k, units='hartree')
    dP_m = build_nonlocal_velocity_jm_kspace(beta_cat, pao_cat, tables, -k, units='hartree')
    U_T = _time_reversal_matrix_p_shell()
    expected = -np.einsum('ab,kxbc,cd->kxad', U_T, np.conj(dP_p), U_T.conj().T)
    np.testing.assert_allclose(dP_m, expected, atol=1e-10)


def test_jm_delta_p_rydberg_is_twice_hartree():
    a_cart = 12.0 * np.eye(3)
    beta_cat, pao_cat = _make_rel_p_system(np.zeros(3), np.array([1.3, -0.7, 0.4]))
    tables = _build_tables(beta_cat, pao_cat, a_cart)
    k = np.array([[0.21, 0.13, -0.05]])
    dP_ha = build_nonlocal_velocity_jm_kspace(beta_cat, pao_cat, tables, k, units='hartree')
    dP_ry = build_nonlocal_velocity_jm_kspace(beta_cat, pao_cat, tables, k, units='rydberg')
    np.testing.assert_allclose(dP_ry, 2.0 * dP_ha, atol=1e-12)
