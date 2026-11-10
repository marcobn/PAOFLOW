"""Augmented (ultrasoft / PAW) metric of the internal projections.

The algebra of :mod:`PAOFLOW.projection.augmentation` is checked on random plane-wave rows
(the Loewdin factor in a general metric, the reduction to the plain overlap when the charges
vanish) and the projector tables on the PAW pseudopotentials shipped with example10 (the block
structure of ``q_ij``, the norm-conserving default).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.elphon.pseudopotential_terms import KleinmanBylanderProjectors
from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.projection.augmentation import AugmentedOverlap, is_augmented, k_plus_g_vectors
from PAOFLOW.projection.do_atwfc_proj import loewdin_factor, ortho_atwfc_k

REPO_ROOT = Path(__file__).resolve().parents[3]
PAW = {
    'Ga': REPO_ROOT / 'examples' / 'qe_examples' / 'example10' / 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'As': REPO_ROOT / 'examples' / 'qe_examples' / 'example10' / 'As.pbe-n-kjpaw_psl.1.0.0.UPF',
}
NC = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV' / 'Si_ONCV_PBE_sr.UPF'


def _random_rows(rng, n, npw):
    return rng.normal(size=(n, npw)) + 1j * rng.normal(size=(n, npw))


class _FakeKB:
    def __init__(self, Q, beta):
        self.Q = Q
        self._beta = beta

    @property
    def nproj(self):
        return self.Q.shape[0]

    def at(self, kpg, tau_alat, tpiba):
        return self._beta


def test_loewdin_factor_orthonormalises_in_a_general_metric():
    rng = np.random.default_rng(3)
    n, npw, nproj = 5, 40, 6
    a = _random_rows(rng, n, npw)
    beta = _random_rows(rng, nproj, npw)
    q = rng.normal(size=(nproj, nproj))
    Q = 0.1 * (q @ q.T)  # positive semi-definite: S stays a metric
    aug = AugmentedOverlap(_FakeKB(Q, beta), np.zeros((1, 3)), 1.0)
    S = aug.overlap(a, a, beta)
    assert np.allclose(S, S.conj().T)
    explicit = a.conj() @ a.T + (a.conj() @ beta.T) @ Q @ (beta.conj() @ a.T)
    assert np.allclose(S, explicit, atol=1e-12)
    X = loewdin_factor(S)
    assert np.allclose(X.conj() @ S @ X.T, np.eye(n), atol=1e-10)
    # the orthonormalised rows are X @ a and are orthonormal in the metric S
    rows = X @ a
    assert np.allclose(aug.overlap(rows, rows, beta), np.eye(n), atol=1e-10)
    # ortho_atwfc_k with the metric gives the same rows
    assert np.allclose(ortho_atwfc_k(a, overlap=S), rows, atol=1e-12)


def test_vanishing_charges_reduce_to_the_plain_overlap():
    rng = np.random.default_rng(5)
    a = _random_rows(rng, 4, 30)
    beta = _random_rows(rng, 3, 30)
    aug = AugmentedOverlap(_FakeKB(np.zeros((3, 3)), beta), np.zeros((1, 3)), 1.0)
    S = aug.overlap(a, a, beta)
    assert np.allclose(S, a.conj() @ a.T, atol=1e-13)
    assert np.allclose(ortho_atwfc_k(a, overlap=S), ortho_atwfc_k(a), atol=1e-12)


def test_k_plus_g_vectors_matches_the_atomic_wavefunction_construction():
    rng = np.random.default_rng(1)
    mill = rng.integers(-3, 4, size=(3, 12))
    bg = rng.normal(size=(3, 3))
    xk = rng.normal(size=3)
    gk = {'mill': mill, 'bg': bg, 'xk': xk}
    expected = np.dot(mill.T, bg.T)
    for ig in range(12):
        expected[ig, :] += xk
    assert np.allclose(k_plus_g_vectors(gk), expected)


@pytest.mark.skipif(not all(p.exists() for p in PAW.values()), reason='example10 UPFs missing')
def test_paw_projectors_carry_block_diagonal_augmentation_charges():
    upfs = {name: UPF(str(path)) for name, path in PAW.items()}
    assert all(is_augmented(u) for u in upfs.values())
    with pytest.raises(NotImplementedError):
        KleinmanBylanderProjectors(upfs, ['Ga', 'As'], 300.0, 8.0)
    kb = KleinmanBylanderProjectors(upfs, ['Ga', 'As'], 300.0, 8.0, require_norm_conserving=False)
    assert kb.Q is not None and kb.Q.shape == (kb.nproj, kb.nproj)
    assert np.allclose(kb.Q, kb.Q.T)
    atom, lm = kb.atom_of_projector, kb._lm
    same = (atom[:, None] == atom[None, :]) & (lm[:, None] == lm[None, :])
    assert np.all(kb.Q[~same] == 0.0)
    for i in range(kb.nproj):
        for j in range(kb.nproj):
            if same[i, j]:
                name = ['Ga', 'As'][atom[i]]
                assert kb.Q[i, j] == upfs[name].qqq[kb._nb[i], kb._nb[j]]
    aug = AugmentedOverlap(kb, np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]), 2 * np.pi / 10.68)
    kpg = np.random.default_rng(0).normal(size=(20, 3))
    beta = aug.beta(kpg)
    assert beta.shape == (kb.nproj, 20)
    assert np.all(np.isfinite(beta))


@pytest.mark.skipif(not NC.exists(), reason='example16 UPF missing')
def test_norm_conserving_projectors_have_no_charges():
    upf = UPF(str(NC))
    assert not is_augmented(upf)
    kb = KleinmanBylanderProjectors({'Si': upf}, ['Si'], 270.0, 6.0, require_norm_conserving=False)
    assert kb.Q is None
    with pytest.raises(ValueError):
        AugmentedOverlap(kb, np.zeros((1, 3)), 1.0)


def test_unparsed_ultrasoft_file_falls_back_to_the_plain_overlap(monkeypatch, capsys):
    """A UPF v1 ultrasoft species has no parsed projectors or charges: warn and return None."""
    from PAOFLOW.projection import augmentation as module

    class _Upf:
        def __init__(self, path):
            self.ptype = 'US' if 'van' in path else 'NC'
            self.has_augmentation = self.ptype == 'US'
            self.qqq = None
            self.beta = []

    class _DC:
        rank = 0

        def data_dicts(self):
            arrays = {
                'species': [('Al', 'Al.pbe-rrkj.UPF'), ('H', 'H.pbe-van_bm.UPF')],
                'atoms': ['Al', 'H'],
                'tau': np.zeros((2, 3)),
            }
            attrs = {'dft': 'QE', 'fpath': '/nowhere', 'ecutwfc': 20.0, 'omega': 100.0, 'alat': 5.0}
            return arrays, attrs

    monkeypatch.setattr(module, 'UPF', _Upf)
    assert module.build_augmentation(_DC()) is None
    assert 'H' in capsys.readouterr().out
