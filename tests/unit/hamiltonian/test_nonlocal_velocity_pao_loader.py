"""Unit tests for the Phase 3b PAO atomic-orbital catalog loader.

Exercises :func:`PAOFLOW.hamiltonian.nonlocal_velocity.load_pao_orbitals`
on the same real norm-conserving UPFs used by the β-projector loader
tests.  Covers:

* Per-species PSWFC channel inventory (l, label, occupation).
* PAOFLOW basis ordering (channel-major, QE-indexed m sweep) and the
  ``qe_m → standard m`` conversion via
  :func:`qe_m_index_to_std`.
* Conversion of UPF's ``r·R(r)`` to the bare radial :math:`R(r)`,
  including the ``r=0`` extrapolation for ``l=0``.
* Multi-species (MnO) site → basis mapping with correct
  ``basis_offset`` values.
* USPP rejection and species/tau mismatch error paths (shared with the
  β loader).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    PAOCatalog,
    PAOSpeciesData,
    load_pao_orbitals,
    qe_m_index_to_std,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'
MNO_DIR = REPO_ROOT / 'examples' / 'acbn0_examples' / 'MnO'
USPP_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example03'
USPP_UPF = 'Pt.pz-n-rrkjus_psl.0.1.UPF'


class _StubDataController:
    def __init__(self, arrays, attributes):
        self._arrays = arrays
        self._attributes = attributes

    def data_dicts(self):
        return self._arrays, self._attributes


# ----------------------------------------------------------------------
# qe_m_index_to_std — closed-form mapping.
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'l,qe_m,expected',
    [
        (0, 1, 0),
        (1, 1, 0),
        (1, 2, +1),
        (1, 3, -1),
        (2, 1, 0),
        (2, 2, +1),
        (2, 3, -1),
        (2, 4, +2),
        (2, 5, -2),
        (3, 1, 0),
        (3, 2, +1),
        (3, 3, -1),
        (3, 4, +2),
        (3, 5, -2),
        (3, 6, +3),
        (3, 7, -3),
    ],
)
def test_qe_m_index_to_std(l, qe_m, expected):
    assert qe_m_index_to_std(qe_m, l) == expected


@pytest.mark.parametrize('bad', [(0, 0), (1, 4), (2, 6)])
def test_qe_m_index_out_of_range_raises(bad):
    l, qe_m = bad
    with pytest.raises(ValueError, match='out of range'):
        qe_m_index_to_std(qe_m, l)


# ----------------------------------------------------------------------
# Si ONCV PBE (sp-only PAO basis: 3S, 3P → 4 orbitals/site).
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_pao_catalog():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    tau = np.array([[0.0, 0.0, 0.0], [1.357, 1.357, 1.357]])
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': tau,
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    return load_pao_orbitals(_StubDataController(arrays, attributes)), tau


def test_si_pao_catalog_shape(si_pao_catalog):
    cat, _ = si_pao_catalog
    assert isinstance(cat, PAOCatalog)
    assert set(cat.species.keys()) == {'Si'}
    sp = cat.species['Si']
    assert isinstance(sp, PAOSpeciesData)
    # Si ONCV ships 3S, 3P.
    assert [ch.label for ch in sp.channels] == ['3S', '3P']
    assert [ch.l for ch in sp.channels] == [0, 1]
    # Two sites, each with 1 + 3 = 4 PAO orbitals → 8 total.
    assert len(cat.sites) == 2
    assert cat.total_nlm == 8
    assert len(cat.basis) == 8


def test_si_pao_basis_ordering(si_pao_catalog):
    cat, _ = si_pao_catalog
    # Channel-major, m sweeps inside each channel; site 0 then site 1.
    # Expected (site, channel, qe_m, l, m_std):
    expected = [
        (0, 0, 1, 0, 0),  # site 0, 3S
        (0, 1, 1, 1, 0),  # site 0, 3P, qe_m=1 → m_std=0 (pz)
        (0, 1, 2, 1, +1),  # site 0, 3P, qe_m=2 → m_std=+1 (px)
        (0, 1, 3, 1, -1),  # site 0, 3P, qe_m=3 → m_std=-1 (py)
        (1, 0, 1, 0, 0),  # site 1, 3S
        (1, 1, 1, 1, 0),
        (1, 1, 2, 1, +1),
        (1, 1, 3, 1, -1),
    ]
    assert len(cat.basis) == len(expected)
    for entry, (s, c, qm, l, m) in zip(cat.basis, expected):
        assert (entry.site_index, entry.channel_index, entry.qe_m, entry.l, entry.m) == (
            s,
            c,
            qm,
            l,
            m,
        )
    # basis_index is contiguous 0..7.
    assert [e.basis_index for e in cat.basis] == list(range(8))


def test_si_pao_site_offsets(si_pao_catalog):
    cat, _ = si_pao_catalog
    # Site 0 owns indices 0..3, site 1 owns 4..7.
    assert cat.sites[0].basis_offset == 0
    assert cat.sites[1].basis_offset == 4
    assert [e.basis_index for e in cat.sites[0].orbitals] == [0, 1, 2, 3]
    assert [e.basis_index for e in cat.sites[1].orbitals] == [4, 5, 6, 7]


def test_si_pao_radial_extrapolation_for_l0(si_pao_catalog):
    """For Si 3S (l=0), R(r=0) is finite — verify the extrapolation."""
    cat, _ = si_pao_catalog
    sp = cat.species['Si']
    ch_s = next(ch for ch in sp.channels if ch.l == 0)
    r = sp.r
    R = ch_s.R_radial
    wfc = ch_s.wfc
    assert r[0] == 0.0  # QE mesh anchor
    # wfc = r·R, so for r>0, R = wfc/r — verify a few points.
    np.testing.assert_allclose(R[1:5], wfc[1:5] / r[1:5], rtol=1e-12)
    # R(0) extrapolation must be finite and close to R(r[1]).
    assert np.isfinite(R[0])
    # The extrapolant should sit between R[1] and a linear continuation.
    # Sanity: |R(0)| comparable in magnitude to R(r[1..3]).
    assert abs(R[0]) < 10.0 * max(abs(R[1]), abs(R[2]), abs(R[3]))


def test_si_pao_radial_zero_at_origin_for_l1(si_pao_catalog):
    """For Si 3P (l=1), R(0) must be exactly 0."""
    cat, _ = si_pao_catalog
    sp = cat.species['Si']
    ch_p = next(ch for ch in sp.channels if ch.l == 1)
    assert sp.r[0] == 0.0
    assert ch_p.R_radial[0] == 0.0
    # Non-zero away from origin.
    assert abs(ch_p.R_radial[10]) > 0.0


def test_si_pao_shared_species_reference(si_pao_catalog):
    cat, tau = si_pao_catalog
    sp = cat.species['Si']
    for site in cat.sites:
        assert site.species is sp
    np.testing.assert_allclose(cat.sites[0].tau, tau[0])
    np.testing.assert_allclose(cat.sites[1].tau, tau[1])


# ----------------------------------------------------------------------
# Multi-species (MnO rock salt: 3S, 3P, 3D, 4S × Mn  +  2S, 2P × O).
# ----------------------------------------------------------------------


@pytest.fixture(scope='module')
def mno_pao_catalog():
    if not (MNO_DIR / 'Mn.upf').exists() or not (MNO_DIR / 'O.upf').exists():
        pytest.skip(f'MnO UPFs missing under {MNO_DIR}')
    arrays = {
        'species': [('Mn', 'Mn.upf'), ('O', 'O.upf')],
        'atoms': ['Mn', 'O'],
        'tau': np.array([[0.0, 0.0, 0.0], [2.22, 2.22, 2.22]]),
    }
    attributes = {'fpath': str(MNO_DIR)}
    return load_pao_orbitals(_StubDataController(arrays, attributes))


def test_mno_pao_two_species_and_basis_dim(mno_pao_catalog):
    cat = mno_pao_catalog
    assert set(cat.species.keys()) == {'Mn', 'O'}
    mn = cat.species['Mn']
    o = cat.species['O']
    # Mn 3S, 3P, 3D, 4S → 1+3+5+1 = 10 PAO orbitals.
    assert [ch.label for ch in mn.channels] == ['3S', '3P', '3D', '4S']
    assert [ch.l for ch in mn.channels] == [0, 1, 2, 0]
    # O 2S, 2P → 1+3 = 4.
    assert [ch.label for ch in o.channels] == ['2S', '2P']
    assert [ch.l for ch in o.channels] == [0, 1]
    # Total nawf = 10 + 4 = 14.
    assert cat.total_nlm == 14
    assert len(cat.basis) == 14
    # Site 0 (Mn) owns 0..9; site 1 (O) owns 10..13.
    assert cat.sites[0].basis_offset == 0
    assert cat.sites[1].basis_offset == 10
    assert len(cat.sites[0].orbitals) == 10
    assert len(cat.sites[1].orbitals) == 4


def test_mno_pao_d_channel_m_sweep(mno_pao_catalog):
    """Mn 3D should produce 5 entries with qe_m = 1..5 → m_std = 0,+1,-1,+2,-2."""
    cat = mno_pao_catalog
    mn_orbitals = cat.sites[0].orbitals
    d_orbitals = [e for e in mn_orbitals if e.l == 2 and e.label == '3D']
    assert len(d_orbitals) == 5
    qe_ms = [e.qe_m for e in d_orbitals]
    m_stds = [e.m for e in d_orbitals]
    assert qe_ms == [1, 2, 3, 4, 5]
    assert m_stds == [0, +1, -1, +2, -2]


# ----------------------------------------------------------------------
# Error paths (shared with the β loader; light coverage).
# ----------------------------------------------------------------------


def test_pao_loader_rejects_uspp():
    if not (USPP_DIR / USPP_UPF).exists():
        pytest.skip(f'USPP fixture missing: {USPP_DIR / USPP_UPF}')
    arrays = {
        'species': [('Pt', USPP_UPF)],
        'atoms': ['Pt'],
        'tau': np.zeros((1, 3)),
    }
    attributes = {'fpath': str(USPP_DIR)}
    with pytest.raises(RuntimeError, match='norm-conserving'):
        load_pao_orbitals(_StubDataController(arrays, attributes))


def test_pao_loader_rejects_unknown_species():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Ge'],
        'tau': np.zeros((2, 3)),
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    with pytest.raises(RuntimeError, match="species 'Ge'"):
        load_pao_orbitals(_StubDataController(arrays, attributes))


def test_pao_loader_rejects_tau_atoms_mismatch():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': np.zeros((3, 3)),
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    with pytest.raises(RuntimeError, match='tau'):
        load_pao_orbitals(_StubDataController(arrays, attributes))


# ----------------------------------------------------------------------
# atomic_basis path — consumes PAOFLOW.projections() output.
# ----------------------------------------------------------------------


def _si_atomic_basis_records():
    """Build minimal ``arry['atomic_basis']`` records mimicking the dicts
    emitted by :func:`build_pswfc_basis_all` for a 2-Si-atom cell.

    We synthesize the radial functions on a mesh that differs from the
    UPF mesh so the test fails if the loader silently falls back to
    ``upf.pswfc``.
    """
    r = np.linspace(0.01, 4.0, 401)  # distinct from UPF mesh
    # Shell labels and l values
    shells = [('3S', 0), ('3P', 1)]
    # r * R(r), one wfc per shell — pick distinct profiles so we can
    # tell them apart in the catalog.
    wfcs = {
        '3S': r * np.exp(-0.7 * r * r),
        '3P': r * r * np.exp(-0.5 * r * r),
    }
    basis = []
    taus = [np.zeros(3), np.array([2.55, 2.55, 2.55])]
    for site_i, tau in enumerate(taus):
        for label, l in shells:
            for m in range(1, 2 * l + 2):
                basis.append(
                    {
                        'atom': 'Si',
                        'tau': tau,
                        'l': l,
                        'm': m,
                        'label': label,
                        'r': r,
                        'wfc': wfcs[label],
                    }
                )
    return basis, r, shells


def test_pao_loader_uses_atomic_basis_when_present():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    basis_records, r_basis, shells = _si_atomic_basis_records()
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': np.array([[0.0, 0.0, 0.0], [2.55, 2.55, 2.55]]),
        'atomic_basis': basis_records,
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    cat = load_pao_orbitals(_StubDataController(arrays, attributes))
    sp = cat.species['Si']
    # Channels were taken from atomic_basis (not from UPF.pswfc) → r
    # mesh must match the synthetic one.
    assert sp.r.shape == r_basis.shape
    np.testing.assert_allclose(sp.r, r_basis)
    assert len(sp.channels) == len(shells)
    for ch, (lbl, l) in zip(sp.channels, shells):
        assert ch.label == lbl
        assert ch.l == l
    # Total basis = 2 atoms * (1 + 3) = 8 orbitals.
    assert cat.total_nlm == 8


def test_pao_loader_atomic_basis_rejects_mixed_meshes():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    basis_records, _, _ = _si_atomic_basis_records()
    # Corrupt one record to use a different mesh.
    rogue = dict(basis_records[-1])
    rogue['r'] = rogue['r'] + 0.1
    basis_records[-1] = rogue
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': np.array([[0.0, 0.0, 0.0], [2.55, 2.55, 2.55]]),
        'atomic_basis': basis_records,
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    with pytest.raises(RuntimeError, match='radial mesh'):
        load_pao_orbitals(_StubDataController(arrays, attributes))


def test_pao_loader_falls_back_when_atomic_basis_missing():
    """When ``atomic_basis`` is absent the loader must reproduce the
    legacy ``upf.pswfc`` channel inventory."""
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': np.array([[0.0, 0.0, 0.0], [2.55, 2.55, 2.55]]),
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    cat = load_pao_orbitals(_StubDataController(arrays, attributes))
    sp = cat.species['Si']
    # UPF Si_ONCV mesh starts at r[0] = 0.
    assert sp.r[0] == 0.0
    assert [ch.label for ch in sp.channels] == ['3S', '3P']
