"""Spinor augmentation charges (QE ``qq_so``) of fully relativistic ultrasoft / PAW projectors.

The real-harmonic projectors of a ``|l j>`` channel are rotated into the ``|l j m_j>`` spinors
by ``fcoef``; the augmentation charges ``q_ij`` of the UPF become the ``(2, 2, nproj, nproj)``
matrix ``q^{s1 s2}_{kl} = sum_{ij s} q_ij fcoef(k, i, s1, s) fcoef(j, l, s, s2)`` and the
overlap of two spinors is ``<a|b> + sum becp_a^* q_so becp_b``.
"""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.pseudopotential_terms import (
    LMAXX,
    KleinmanBylanderProjectors,
    real_to_complex_harmonics,
    spin_orbit_coefficient,
)
from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.projection.augmentation import AugmentedOverlap

EXAMPLE17 = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'examples', 'qe_examples', 'example17'
)
UPFS = {
    'Sn': os.path.join(EXAMPLE17, 'Sn.rel-pbe-dn-kjpaw_psl.1.0.0.UPF'),
    'Te': os.path.join(EXAMPLE17, 'Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF'),
}


def _spinor_projector(l, j, rot):
    """``fcoef`` of one ``(l, j)`` channel as a ``(2(2l+1), 2(2l+1))`` matrix over ``(spin, m)``."""
    n = 2 * l + 1
    out = np.zeros((2 * n, 2 * n), dtype=complex)
    for s1 in (1, 2):
        for s2 in (1, 2):
            for mi in range(n):
                for mk in range(n):
                    out[(s1 - 1) * n + mi, (s2 - 1) * n + mk] = spin_orbit_coefficient(
                        l, j, mi, mk, s1, s2, rot
                    )
    return out


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_spinor_rotations_are_projectors_on_the_j_subspaces(l):
    rot = real_to_complex_harmonics(LMAXX)
    n = 2 * l + 1
    total = np.zeros((2 * n, 2 * n), dtype=complex)
    for j in [0.5] if l == 0 else [l - 0.5, l + 0.5]:
        f = _spinor_projector(l, j, rot)
        assert np.allclose(f, f.conj().T)
        assert np.allclose(f @ f, f)
        assert np.trace(f).real == pytest.approx(2 * j + 1)
        total += f
    assert np.allclose(total, np.eye(2 * n))


@pytest.fixture(scope='module')
def projectors():
    upfs = {name: UPF(path) for name, path in UPFS.items()}
    assert all(u.has_spinorbit for u in upfs.values())
    return KleinmanBylanderProjectors(
        upfs, ['Sn', 'Te'], 500.0, 8.0, spin_orbit=True, require_norm_conserving=False
    )


def test_spinor_charges_match_the_reference_loop(projectors):
    kb = projectors
    assert kb.Q is None and kb.Q_so is not None
    nproj = kb.nproj
    assert kb.Q_so.shape == (2, 2, nproj, nproj)
    # Hermitian in the combined (projector, spin) index
    for s1 in range(2):
        for s2 in range(2):
            assert np.allclose(kb.Q_so[s1, s2], kb.Q_so[s2, s1].conj().T)
    # QE init_us_1 loop on the projectors of the first atom with l <= 1
    rot = real_to_complex_harmonics(LMAXX)
    sel = np.where((kb.atom_of_projector == 0) & (kb.l_of_projector <= 1))[0]
    assert sel.size >= 8  # two s channels and two p channels (j = 1/2, 3/2)
    name = kb.species_of_atom[0]
    qqq = kb.qqq[name]

    def q_nt(i, j):
        return qqq[kb._nb[i], kb._nb[j]] if kb._lm[i] == kb._lm[j] else 0.0

    def fcoef(i, k, s1, s2):
        # QE: fcoef between projectors of different channels is zeroed before qq_so
        if kb._nb[i] != kb._nb[k]:
            return 0.0
        l = int(kb.l_of_projector[i])
        return spin_orbit_coefficient(
            l, float(kb.j_of_projector[i]), kb._lm[i] - l * l, kb._lm[k] - l * l, s1, s2, rot
        )

    for kh in sel:
        for lh in sel:
            for s1 in (1, 2):
                for s2 in (1, 2):
                    ref = 0.0
                    for ih in sel:
                        for jh in sel:
                            for s in (1, 2):
                                ref += q_nt(ih, jh) * fcoef(kh, ih, s1, s) * fcoef(jh, lh, s, s2)
                    assert kb.Q_so[s1 - 1, s2 - 1, kh, lh] == pytest.approx(ref, abs=1e-12)


def test_spinor_charges_sum_to_the_identity_when_j_is_averaged_out(projectors):
    """Charges blind to ``j`` give ``(sum_c P_{j_c})^2 = n_c^2`` on the ``(m, spin)`` space of each ``l``.

    With ``q_ik = 1`` between any two projectors of the same atom, ``l`` and ``m`` (any ``j``) the
    spinor charges of two channels are ``P_{j_c} P_{j_c'}``, the products of the projectors on the
    ``|l j m_j>`` subspaces; summed over the channels of one ``l`` they give the square of
    ``n_c (P_{l-1/2} + P_{l+1/2}) = n_c``, a spin-diagonal identity, when every ``j`` has the same
    number of channels ``n_c``.
    """
    kb = projectors
    rot = real_to_complex_harmonics(LMAXX)
    nproj = kb.nproj
    q = np.zeros((nproj, nproj))
    f = np.zeros((2, 2, nproj, nproj), dtype=complex)
    for i in range(nproj):
        for k in range(nproj):
            if kb.atom_of_projector[i] != kb.atom_of_projector[k]:
                continue
            if kb._lm[i] == kb._lm[k]:
                q[i, k] = 1.0
            if kb._nb[i] == kb._nb[k]:
                for s1 in (1, 2):
                    for s2 in (1, 2):
                        f[s1 - 1, s2 - 1, i, k] = kb._fcoef(i, k, s1, s2, rot)
    q_so = np.array(
        [[sum(f[s1, s] @ q @ f[s, s2] for s in range(2)) for s2 in range(2)] for s1 in range(2)]
    )
    checked = 0
    for atom in sorted(set(kb.atom_of_projector)):
        for l in sorted(set(kb.l_of_projector[kb.atom_of_projector == atom])):
            sel = np.where((kb.atom_of_projector == atom) & (kb.l_of_projector == l))[0]
            channels = {}
            for i in sel:
                channels.setdefault(float(kb.j_of_projector[i]), set()).add(int(kb._nb[i]))
            counts = {len(v) for v in channels.values()}
            assert len(counts) == 1, 'the test needs the same number of channels per j'
            n_c = counts.pop()
            n = 2 * l + 1
            total = np.zeros((2, 2, n, n), dtype=complex)
            for i in sel:
                for k in sel:
                    total[:, :, kb._lm[i] - l * l, kb._lm[k] - l * l] += q_so[:, :, i, k]
            for s1 in range(2):
                for s2 in range(2):
                    expect = (n_c**2 * np.eye(n)) if s1 == s2 else np.zeros((n, n))
                    assert np.allclose(total[s1, s2], expect, atol=1e-12)
            checked += 1
    assert checked >= 4


def test_spinor_overlap_matches_the_explicit_formula(projectors):
    rng = np.random.default_rng(3)
    kb = projectors
    tau = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    aug = AugmentedOverlap(kb, tau, 2.0 * np.pi / 8.62)
    assert aug.spin_orbit
    npw, na, nb = 40, 5, 7
    k_plus_g = rng.normal(size=(npw, 3))
    beta = aug.beta(k_plus_g)
    assert beta.shape == (kb.nproj, npw)
    a = rng.normal(size=(na, 2 * npw)) + 1j * rng.normal(size=(na, 2 * npw))
    b = rng.normal(size=(nb, 2 * npw)) + 1j * rng.normal(size=(nb, 2 * npw))
    becp_a = aug.becp(beta, a)
    assert becp_a.shape == (2, kb.nproj, na)
    ref = a.conj() @ b.T
    for s1 in range(2):
        for s2 in range(2):
            ca = beta.conj() @ a[:, s1 * npw : (s1 + 1) * npw].T
            cb = beta.conj() @ b[:, s2 * npw : (s2 + 1) * npw].T
            ref += ca.conj().T @ kb.Q_so[s1, s2] @ cb
    out = aug.overlap(a, b, beta)
    assert np.allclose(out, ref)
    assert np.allclose(aug.overlap(a, a, beta), aug.overlap(a, a, beta).conj().T)
    with pytest.raises(ValueError):
        aug.becp(beta, a[:, :npw])


SNTE_SAVE = os.path.join(EXAMPLE17, 'SnTe.save')


@pytest.mark.skipif(not os.path.isdir(SNTE_SAVE), reason='example17 QE data not available')
def test_snte_bands_are_orthonormal_in_the_spinor_metric(tmp_path):
    """End to end on the SnTe monolayer (FR PAW, ``npol = 2``): ``<psi|S|psi> = 1`` for the
    stored bands with the spinor charges, and far from it with the plain overlap."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.projection.augmentation import build_augmentation, k_plus_g_vectors
    from PAOFLOW.projection.do_atwfc_proj import read_QE_wfc

    pf = PAOFLOW.PAOFLOW(
        savedir=SNTE_SAVE, outputdir=str(tmp_path), smearing=None, npool=1, verbose=False
    )
    dc = pf.data_controller
    _, attr = dc.data_dicts()
    assert attr['dftSO']
    aug = build_augmentation(dc)
    assert aug is not None and aug.spin_orbit
    gkspace, wfc = read_QE_wfc(dc, 0, 0, orthonormalize=False)
    assert wfc['npol'] == 2
    beta = aug.beta(k_plus_g_vectors(gkspace))
    psi = wfc['wfc']
    ovp = aug.overlap(psi, psi, beta)
    assert np.abs(ovp - np.eye(ovp.shape[0])).max() < 1e-4
    plain = psi.conj() @ psi.T
    assert np.abs(plain - np.eye(plain.shape[0])).max() > 0.1
