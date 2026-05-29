"""Parity tests: build_generic_soc must match the hardcoded soc_<l>_<token>
kernels bit-for-bit on the layouts they support, and remain Hermitian on
arbitrary shells lists (e.g. extended AE basis)."""

import numpy as np
import pytest

from PAOFLOW.hamiltonian.do_spin_orbit import (
    build_generic_soc,
    soc_d_spds,
    soc_d_sspd,
    soc_d_ssppd,
    soc_p_sp,
    soc_p_spd,
    soc_p_spds,
    soc_p_sspd,
    soc_p_ssppd,
)

ANGLES = [(0.0, 0.0), (0.7, 1.1), (np.pi / 3, np.pi / 5), (np.pi / 2, 0.0)]


# Layouts that have both a hardcoded kernel AND known to be Hermitian-clean
# in the existing code (drop spd from d-parity because soc_d_spd is missing
# the [8,7] Hermitian conjugate fill for the cTheta term).
@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_sp(theta, phi):
    norb = 4
    HR_p_ref = soc_p_sp(theta, phi, norb)
    HR_p, HR_d = build_generic_soc(theta, phi, [0, 1], norb)
    assert np.allclose(HR_p, HR_p_ref)
    assert np.allclose(HR_d, 0)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_spd_p(theta, phi):
    norb = 9
    HR_p_ref = soc_p_spd(theta, phi, norb)
    HR_p, _ = build_generic_soc(theta, phi, [0, 1, 2], norb)
    assert np.allclose(HR_p, HR_p_ref)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_sspd_p(theta, phi):
    norb = 10
    HR_p_ref = soc_p_sspd(theta, phi, norb)
    HR_p, _ = build_generic_soc(theta, phi, [0, 0, 1, 2], norb)
    assert np.allclose(HR_p, HR_p_ref)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_spds_p(theta, phi):
    norb = 10
    HR_p_ref = soc_p_spds(theta, phi, norb)
    HR_p, _ = build_generic_soc(theta, phi, [0, 1, 2, 0], norb)
    assert np.allclose(HR_p, HR_p_ref)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_ssppd_p(theta, phi):
    """ssppd p-block: ``soc_p_ssppd`` intentionally places SOC only on the
    *second* (valence) p shell at indices 5–7; the semicore p at 2–4
    carries no SOC by convention.  The generic builder's default
    first-occurrence mask would instead activate the *first* p shell, so
    pass an explicit ``active_shells`` to match the hardcoded
    convention."""
    norb = 13
    HR_p_ref = soc_p_ssppd(theta, phi, norb)
    active = [False, False, False, True, False]  # only second p (l=1)
    HR_p, _ = build_generic_soc(theta, phi, [0, 0, 1, 1, 2], norb, active_shells=active)
    assert np.allclose(HR_p, HR_p_ref)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_default_falloff_mask(theta, phi):
    """Default weight = 1/k**2 over the occurrence index of each l > 0.
    For an extended-Pt layout with 3 p-shells and 3 d-shells, the
    augmentation sub-blocks must equal 1/4 and 1/9 of the valence-only
    reference block respectively (same Hermitian kernel, just scaled)."""
    shells = [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2]
    norb = sum(2 * l + 1 for l in shells)
    HR_p, HR_d = build_generic_soc(theta, phi, shells, norb)
    starts = []
    off = 0
    for l in shells:
        starts.append(off)
        off += 2 * l + 1
    # Single-shell reference kernels (weight=1).
    HR_p1 = build_generic_soc(theta, phi, [1], 3)[0]
    HR_d1 = build_generic_soc(theta, phi, [2], 5)[1]

    def _spinor_block(H, s, n, total):
        idx = np.array(list(range(s, s + n)) + list(range(s + total, s + total + n)))
        return H[np.ix_(idx, idx)]

    ref_p = _spinor_block(HR_p1, 0, 3, 3)
    ref_d = _spinor_block(HR_d1, 0, 5, 5)
    expected_w = [1.0, 0.25, 1.0 / 9.0, 1.0 / 16.0]
    for k, pos in enumerate([i for i, l in enumerate(shells) if l == 1]):
        block = _spinor_block(HR_p, starts[pos], 3, norb)
        assert np.allclose(block, expected_w[k] * ref_p), f'p shell #{k}'
    for k, pos in enumerate([i for i, l in enumerate(shells) if l == 2]):
        block = _spinor_block(HR_d, starts[pos], 5, norb)
        assert np.allclose(block, expected_w[k] * ref_d), f'd shell #{k}'


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_spds_d(theta, phi):
    """spds d-block: generic must match soc_d_spds (which is the
    Hermitian-clean version derived in this session)."""
    norb = 10
    HR_d_ref = soc_d_spds(theta, phi, norb)
    _, HR_d = build_generic_soc(theta, phi, [0, 1, 2, 0], norb)
    assert np.allclose(HR_d, HR_d_ref)


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_sspd_d(theta, phi):
    norb = 10
    HR_d_ref = soc_d_sspd(theta, phi, norb)
    _, HR_d = build_generic_soc(theta, phi, [0, 0, 1, 2], norb)
    # soc_d_sspd has the same missing [.,.] Hermitian fill as soc_d_spd;
    # the generic builder produces the Hermitian-clean version, so compare
    # only on entries the hardcoded kernel actually wrote.
    mask = HR_d_ref != 0
    assert np.allclose(HR_d[mask], HR_d_ref[mask])


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_matches_ssppd_d(theta, phi):
    norb = 13
    HR_d_ref = soc_d_ssppd(theta, phi, norb)
    _, HR_d = build_generic_soc(theta, phi, [0, 0, 1, 1, 2], norb)
    mask = HR_d_ref != 0
    assert np.allclose(HR_d[mask], HR_d_ref[mask])


@pytest.mark.parametrize('theta,phi', ANGLES)
def test_generic_hermitian(theta, phi):
    """Generic SOC blocks must always be Hermitian."""
    shells = [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2]  # extended Pt
    norb = sum(2 * l + 1 for l in shells)
    HR_p, HR_d = build_generic_soc(theta, phi, shells, norb)
    assert np.allclose(HR_p, HR_p.conj().T)
    assert np.allclose(HR_d, HR_d.conj().T)


def test_generic_extended_pt_layout():
    """Extended-Pt shells list (11 shells, 31 orbitals) must build
    without error and produce nonzero p- and d-blocks."""
    shells = [0, 1, 2, 0, 0, 0, 1, 1, 1, 2, 2]
    norb = sum(2 * l + 1 for l in shells)
    assert norb == 31
    HR_p, HR_d = build_generic_soc(0.3, 0.7, shells, norb)
    assert HR_p.shape == (2 * norb, 2 * norb)
    assert HR_d.shape == (2 * norb, 2 * norb)
    assert np.any(HR_p != 0)
    assert np.any(HR_d != 0)
    # s shells contribute nothing
    s_indices = [0, 9, 10, 11]  # positions of the four s shells in the layout
    for i in s_indices:
        # both p and d should have a zero row/col at the s orbital index
        assert np.allclose(HR_p[i, :], 0)
        assert np.allclose(HR_p[:, i], 0)
        assert np.allclose(HR_d[i, :], 0)
        assert np.allclose(HR_d[:, i], 0)


def test_generic_norb_mismatch_raises():
    with pytest.raises(ValueError, match='does not match norb'):
        build_generic_soc(0.0, 0.0, [0, 1], norb=99)
