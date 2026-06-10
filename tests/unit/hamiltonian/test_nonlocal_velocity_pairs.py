"""Unit tests for the Phase 3b real-space enumeration helpers.

Covers:

* :func:`pao_cutoff_radius` on a synthetic Gaussian and on real PAO
  channels (Si ONCV).
* :func:`_lattice_search_bounds` for cubic and triclinic cells.
* :func:`enumerate_nl_pairs` on a Si diamond cell (β and PAO catalogs
  built from the Si ONCV fixture): home-cell self pair, first-neighbor
  bonds, periodic-image inversion symmetry, distance cutoff.
* Mismatched-catalog error path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    NLPair,
    PAOChannelData,
    _lattice_search_bounds,
    enumerate_nl_pairs,
    load_beta_projectors,
    load_pao_orbitals,
    pao_cutoff_radius,
    site_beta_cutoff,
    site_pao_cutoff,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'


class _StubDataController:
    def __init__(self, arrays, attributes):
        self._arrays = arrays
        self._attributes = attributes

    def data_dicts(self):
        return self._arrays, self._attributes


# ---------------------------------------------------------------------------
# pao_cutoff_radius
# ---------------------------------------------------------------------------


def test_pao_cutoff_radius_gaussian():
    """Synthetic wfc = r·exp(-r²/2) → cutoff ≈ √(-2 ln tol) for small tol."""
    r = np.linspace(0.0, 12.0, 2000)
    wfc = r * np.exp(-(r**2) / 2.0)
    ch = PAOChannelData(label='1S', l=0, R_radial=np.zeros_like(r), wfc=wfc, occupation=2.0)
    tol = 1e-4
    rc = pao_cutoff_radius(ch, r, tol=tol)
    # Peak of r·exp(-r²/2) is at r=1, value 1/√e ≈ 0.607.
    # Threshold = tol * 0.607 ≈ 6.07e-5; solve r·exp(-r²/2) = threshold → r ≈ 4.6.
    assert 3.5 < rc < 6.5
    # Stronger: every grid point past rc must be below threshold.
    peak = wfc.max()
    assert np.all(wfc[r > rc + 1e-9] <= tol * peak + 1e-12)


def test_pao_cutoff_radius_zero_peak():
    r = np.linspace(0.0, 5.0, 100)
    ch = PAOChannelData(
        label='X', l=0, R_radial=np.zeros_like(r), wfc=np.zeros_like(r), occupation=0.0
    )
    assert pao_cutoff_radius(ch, r) == 0.0


def test_site_pao_cutoff_picks_largest_channel():
    """Helper picks the maximum across channels."""
    r = np.linspace(0.0, 12.0, 2000)
    # Channel A: tight Gaussian → small cutoff.
    chA = PAOChannelData(
        label='1S', l=0, R_radial=np.zeros_like(r), wfc=r * np.exp(-(r**2) / 0.4), occupation=2.0
    )
    # Channel B: broader Gaussian → larger cutoff.
    chB = PAOChannelData(
        label='2P', l=1, R_radial=np.zeros_like(r), wfc=r**2 * np.exp(-(r**2) / 4.0), occupation=6.0
    )
    rcA = pao_cutoff_radius(chA, r)
    rcB = pao_cutoff_radius(chB, r)
    assert rcB > rcA

    # Stub site that pretends to own these two channels.
    class _Sp:
        r = None
        channels = [chA, chB]

    class _Site:
        species = _Sp()

    _Sp.r = r
    assert site_pao_cutoff(_Site(), tol=1e-4) == pytest.approx(rcB)


# ---------------------------------------------------------------------------
# _lattice_search_bounds
# ---------------------------------------------------------------------------


def test_lattice_search_bounds_cubic():
    a = 10.0 * np.eye(3)
    N1, N2, N3 = _lattice_search_bounds(a, max_radius=15.0)
    # Spacing = 10 → need ceil(15/10)+1 = 2+1 = ... actually ceil(1.5)+1 = 3.
    assert (N1, N2, N3) == (3, 3, 3)
    # max_radius = 5 → ceil(0.5)+1 = 1+1 = 2.
    assert _lattice_search_bounds(a, 5.0) == (2, 2, 2)
    # max_radius = 0 → ceil(0)+1 = 1.
    assert _lattice_search_bounds(a, 0.0) == (1, 1, 1)


def test_lattice_search_bounds_fcc_si():
    """FCC primitive cell with alat = 10.2 Bohr."""
    alat = 10.2
    a = 0.5 * alat * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    # |a_i| = alat/√2 ≈ 7.21; interplanar spacing along each a_i = ?
    # For the standard FCC primitive, the spacing h_i = alat/√3 ≈ 5.89.
    N = _lattice_search_bounds(a, max_radius=10.0)
    # N_i = ceil(10 / 5.89) + 1 = 2 + 1 = 3.
    assert N == (3, 3, 3)


def test_lattice_search_bounds_degenerate_raises():
    a = np.array([[1, 0, 0], [2, 0, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(RuntimeError, match='Degenerate'):
        _lattice_search_bounds(a, 1.0)


# ---------------------------------------------------------------------------
# enumerate_nl_pairs on a real Si diamond cell.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_catalogs_and_lattice():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    alat = 10.2  # Bohr, conventional Si lattice constant.
    # FCC primitive lattice vectors in Cartesian Bohr.
    a_cart = 0.5 * alat * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    # Si diamond basis: atoms at (0,0,0) and alat*(1/4,1/4,1/4).
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat, 0.25 * alat, 0.25 * alat]])
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': tau,
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    dc = _StubDataController(arrays, attributes)
    beta = load_beta_projectors(dc)
    pao = load_pao_orbitals(dc)
    return beta, pao, a_cart, tau, alat


def test_si_per_site_cutoffs_are_finite(si_catalogs_and_lattice):
    beta, pao, _, _, _ = si_catalogs_and_lattice
    rb = [site_beta_cutoff(s) for s in beta.sites]
    rp = [site_pao_cutoff(s) for s in pao.sites]
    # Si ONCV β cutoff_radius is finite (~ few Bohr); PAO numerical tail
    # can be longer (~ 10-20 Bohr) since pseudo-wavefunctions decay slowly.
    for v in rb + rp:
        assert v > 0.0
        assert v < 30.0
    # PAO cutoff usually exceeds β cutoff for sp orbitals.
    assert max(rp) >= max(rb) - 1e-9


def test_si_enumerate_pairs_contains_home_cell(si_catalogs_and_lattice):
    """The (I=J, ΔR=0) self pair must be present for both Si atoms."""
    beta, pao, a_cart, _, _ = si_catalogs_and_lattice
    pairs = enumerate_nl_pairs(beta, pao, a_cart)
    assert len(pairs) > 0
    for site_i in range(2):
        match = [
            p
            for p in pairs
            if p.beta_site == site_i and p.pao_site == site_i and p.deltaR_lattice == (0, 0, 0)
        ]
        assert len(match) == 1
        assert match[0].distance == pytest.approx(0.0, abs=1e-12)


def test_si_enumerate_pairs_contains_first_neighbor(si_catalogs_and_lattice):
    """For Si diamond, the (0→1, ΔR=0) pair has distance alat·√3/4."""
    beta, pao, a_cart, _, alat = si_catalogs_and_lattice
    pairs = enumerate_nl_pairs(beta, pao, a_cart)
    nn = alat * np.sqrt(3.0) / 4.0  # ≈ 4.42 Bohr for alat=10.2.
    match = [
        p for p in pairs if p.beta_site == 0 and p.pao_site == 1 and p.deltaR_lattice == (0, 0, 0)
    ]
    assert len(match) == 1
    assert match[0].distance == pytest.approx(nn, rel=1e-12)
    np.testing.assert_allclose(match[0].displacement, [0.25 * alat] * 3, atol=1e-12)


def test_si_enumerate_pairs_all_within_cutoff(si_catalogs_and_lattice):
    beta, pao, a_cart, _, _ = si_catalogs_and_lattice
    pairs = enumerate_nl_pairs(beta, pao, a_cart)
    for p in pairs:
        assert p.distance <= p.cutoff_used + 1e-12


def test_si_enumerate_pairs_inversion_symmetry(si_catalogs_and_lattice):
    """Diamond Si has inversion symmetry about the midpoint of the two atoms,
    so for every (I, J, ΔR) pair there is a partner (J, I, ΔR') with the
    same |d|."""
    beta, pao, a_cart, _, _ = si_catalogs_and_lattice
    pairs = enumerate_nl_pairs(beta, pao, a_cart)
    # Bucket distances by (I, J) and check sets agree with (J, I).
    buckets: dict[tuple, list[float]] = {}
    for p in pairs:
        buckets.setdefault((p.beta_site, p.pao_site), []).append(round(p.distance, 8))
    a = sorted(buckets.get((0, 1), []))
    b = sorted(buckets.get((1, 0), []))
    assert a == b


def test_si_enumerate_pairs_pad_increases_count(si_catalogs_and_lattice):
    beta, pao, a_cart, _, _ = si_catalogs_and_lattice
    base = enumerate_nl_pairs(beta, pao, a_cart)
    padded = enumerate_nl_pairs(beta, pao, a_cart, extra_pad=2.0)
    assert len(padded) >= len(base)


# ---------------------------------------------------------------------------
# Error paths.
# ---------------------------------------------------------------------------


def test_enumerate_nl_pairs_catalog_size_mismatch(si_catalogs_and_lattice):
    beta, pao, a_cart, tau, _ = si_catalogs_and_lattice
    # Build a 1-atom PAO catalog (different #sites) by stubbing.
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si'],
        'tau': tau[:1].copy(),
    }
    attributes = {'fpath': str(SI_ONCV_DIR)}
    pao1 = load_pao_orbitals(_StubDataController(arrays, attributes))
    with pytest.raises(RuntimeError, match='same crystal'):
        enumerate_nl_pairs(beta, pao1, a_cart)


def test_enumerate_nl_pairs_bad_lattice_shape(si_catalogs_and_lattice):
    beta, pao, _, _, _ = si_catalogs_and_lattice
    with pytest.raises(RuntimeError, match='shape'):
        enumerate_nl_pairs(beta, pao, np.eye(2))


def test_nlpair_dataclass_is_frozen():
    p = NLPair(
        beta_site=0,
        pao_site=0,
        deltaR_lattice=(0, 0, 0),
        deltaR_cart=np.zeros(3),
        displacement=np.zeros(3),
        distance=0.0,
        cutoff_used=1.0,
    )
    with pytest.raises((AttributeError, Exception)):
        p.beta_site = 1  # type: ignore[misc]
