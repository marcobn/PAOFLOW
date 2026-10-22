"""The vectorised two-centre blocks against the element-wise primitives, and the vectorised
non-local tables against the element-wise builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian._two_center import (
    radial_bessel_transform,
    two_center_dipole_overlap_precomputed,
    two_center_overlap_precomputed,
)
from PAOFLOW.hamiltonian.nonlocal_velocity import (
    _build_nl_real_space_tables_elementwise,
    build_nl_real_space_tables,
    enumerate_nl_pairs,
    load_beta_projectors,
    load_pao_orbitals,
    qe_m_index_to_std,
)
from PAOFLOW.hamiltonian.two_center_blocks import _dipole_bra_l, two_center_pair_blocks

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'


def _gaussian_channel(r, l, alpha):
    return np.sqrt(4.0 * np.pi) * r**l * np.exp(-alpha * r * r)


def test_pair_blocks_match_elementwise_primitives():
    r = np.linspace(0.0, 12.0, 2401)
    q = np.linspace(0.0, 20.0, 500)
    bra_spec = [(0, 0.5), (1, 0.4), (2, 0.35)]
    ket_spec = [(0, 0.45), (1, 0.3), (2, 0.5), (0, 0.8)]
    bra = []
    for l, a in bra_spec:
        radial = _gaussian_channel(r, l, a)
        bra.append(
            (
                l,
                radial_bessel_transform(r, radial, l, q),
                {lp: radial_bessel_transform(r, r * radial, lp, q) for lp in _dipole_bra_l(l)},
            )
        )
    ket = [(l, radial_bessel_transform(r, _gaussian_channel(r, l, a), l, q)) for l, a in ket_spec]
    d = np.array([1.3, -0.7, 2.1])
    overlap, dipole = two_center_pair_blocks(bra, ket, d, q)
    assert overlap.shape == (9, 10) and dipole.shape == (3, 9, 10)
    row = 0
    for la, ja, jg in bra:
        for qa in range(1, 2 * la + 2):
            ma = qe_m_index_to_std(qa, la)
            col = 0
            for lb, jb in ket:
                for qb in range(1, 2 * lb + 2):
                    mb = qe_m_index_to_std(qb, lb)
                    ref = two_center_overlap_precomputed(ja, jb, la, ma, lb, mb, d, q)
                    assert overlap[row, col] == pytest.approx(ref, abs=1e-13)
                    for alpha in range(3):
                        ref_d = two_center_dipole_overlap_precomputed(
                            jg, jb, la, ma, lb, mb, d, alpha, q
                        )
                        assert dipole[alpha, row, col] == pytest.approx(ref_d, abs=1e-13)
                    col += 1
            row += 1
    # overlap-only call leaves the dipole block zero
    overlap_only, zeros = two_center_pair_blocks(
        [(l, j, None) for l, j, _ in bra], ket, d, q, include_dipole=False
    )
    np.testing.assert_allclose(overlap_only, overlap, atol=1e-15)
    assert np.all(zeros == 0.0)


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


def test_nl_tables_vectorised_match_elementwise(si_catalogs):
    beta, pao, a_cart = si_catalogs
    pairs = sorted(enumerate_nl_pairs(beta, pao, a_cart, pao_tol=1e-2), key=lambda p: p.distance)[
        :12
    ]
    fast = build_nl_real_space_tables(beta, pao, pairs, q_max=15.0, n_q=300)
    slow = _build_nl_real_space_tables_elementwise(beta, pao, pairs, q_max=15.0, n_q=300)
    assert fast.beta_lm_per_site == slow.beta_lm_per_site and fast.pao_per_site == slow.pao_per_site
    for a, b in zip(fast.S_bp, slow.S_bp):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for a, b in zip(fast.S_rbp, slow.S_rbp):
        np.testing.assert_allclose(a, b, atol=1e-12)
    assert max(np.abs(a).max() for a in fast.S_rbp) > 0.1
