"""Unit tests for the Phase 3a β-projector loader.

These tests mock a minimal ``DataController`` and load real norm-conserving
UPF files from ``examples/`` to verify :func:`load_beta_projectors` builds
the expected per-species / per-site catalog and refuses non-NC pseudos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    BetaCatalog,
    BetaSpeciesData,
    load_beta_projectors,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'
MNO_DIR = REPO_ROOT / 'examples' / 'acbn0_examples' / 'MnO'
USPP_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example03'
USPP_UPF = 'Pt.pz-n-rrkjus_psl.0.1.UPF'


class _StubDataController:
    """Minimal duck-typed DataController for loader tests."""

    def __init__(self, arrays, attributes):
        self._arrays = arrays
        self._attributes = attributes

    def data_dicts(self):
        return self._arrays, self._attributes


# ----------------------------------------------------------------------
# Single-species crystal (Si diamond, two equivalent atoms).
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_catalog():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    # Two atomic sites with the same species.
    tau = np.array([[0.0, 0.0, 0.0], [1.357, 1.357, 1.357]])
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': tau,
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    dc = _StubDataController(arrays, attributes)
    return load_beta_projectors(dc), tau


def test_si_catalog_shape(si_catalog):
    cat, tau = si_catalog
    assert isinstance(cat, BetaCatalog)
    assert set(cat.species.keys()) == {'Si'}
    assert len(cat.sites) == 2
    sp = cat.species['Si']
    assert isinstance(sp, BetaSpeciesData)
    # Si ONCV PBE sr ships 6 projectors: 2× (s, p, d).
    assert sp.nproj == 6
    assert sorted(sp.lchannels) == [0, 0, 1, 1, 2, 2]
    # total_nproj_radial = nproj × natoms (here 6 × 2 = 12).
    assert cat.total_nproj_radial == 12
    # total_nproj_lm = Σ_I Σ_i (2l_i+1) per site:
    #   2 × ((2·0+1)·2 + (2·1+1)·2 + (2·2+1)·2)  = 2 × (2+6+10) = 36.
    assert cat.total_nproj_lm == 36


def test_si_sites_share_species_data(si_catalog):
    cat, tau = si_catalog
    # The per-site species reference is the SAME object -- no duplicated
    # parsing / arrays per site.
    sp = cat.species['Si']
    for site in cat.sites:
        assert site.species is sp
    # tau is faithfully propagated.
    np.testing.assert_allclose(cat.sites[0].tau, tau[0])
    np.testing.assert_allclose(cat.sites[1].tau, tau[1])


def test_si_dion_in_hartree(si_catalog):
    cat, _ = si_catalog
    D = cat.species['Si'].dion
    assert D.shape == (6, 6)
    # Matches the Phase 3a reference (Ry/2 from PP_DIJ).
    expected_diag = (
        np.array(
            [
                1.0337930497e01,
                1.6597653773e00,
                5.1425645739e00,
                1.1566139757e00,
                -4.8546211098e00,
                -9.7619361064e-01,
            ]
        )
        / 2.0
    )
    np.testing.assert_allclose(np.diag(D), expected_diag, rtol=1e-12)


# ----------------------------------------------------------------------
# Multi-species crystal (MnO rock salt).
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def mno_catalog():
    if not (MNO_DIR / 'Mn.upf').exists() or not (MNO_DIR / 'O.upf').exists():
        pytest.skip(f'MnO UPFs missing under {MNO_DIR}')
    arrays = {
        'species': [('Mn', 'Mn.upf'), ('O', 'O.upf')],
        'atoms': ['Mn', 'O'],
        'tau': np.array([[0.0, 0.0, 0.0], [2.22, 2.22, 2.22]]),
    }
    attributes = {'fpath': str(MNO_DIR)}
    dc = _StubDataController(arrays, attributes)
    return load_beta_projectors(dc)


def test_mno_two_species(mno_catalog):
    cat = mno_catalog
    assert set(cat.species.keys()) == {'Mn', 'O'}
    assert [s.label for s in cat.sites] == ['Mn', 'O']
    # Each site references the right species block.
    assert cat.sites[0].species is cat.species['Mn']
    assert cat.sites[1].species is cat.species['O']


# ----------------------------------------------------------------------
# Error paths.
# ----------------------------------------------------------------------


def test_loader_rejects_uspp():
    """Phase 3 is NC only -- USPP must raise a clear error."""
    if not (USPP_DIR / USPP_UPF).exists():
        pytest.skip(f'USPP fixture missing: {USPP_DIR / USPP_UPF}')
    arrays = {
        'species': [('Pt', USPP_UPF)],
        'atoms': ['Pt'],
        'tau': np.zeros((1, 3)),
    }
    attributes = {'fpath': str(USPP_DIR)}
    dc = _StubDataController(arrays, attributes)
    with pytest.raises(RuntimeError, match='norm-conserving'):
        load_beta_projectors(dc)


def test_loader_rejects_unknown_species():
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Ge'],  # 'Ge' has no species entry.
        'tau': np.zeros((2, 3)),
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    dc = _StubDataController(arrays, attributes)
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    with pytest.raises(RuntimeError, match="species 'Ge'"):
        load_beta_projectors(dc)


def test_loader_rejects_tau_atoms_mismatch():
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': np.zeros((3, 3)),  # 3 rows vs 2 atoms.
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    dc = _StubDataController(arrays, attributes)
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    with pytest.raises(RuntimeError, match='tau'):
        load_beta_projectors(dc)
