"""Unit tests for :mod:`PAOFLOW.hamiltonian.position_dipoles`.

Synthetic Gaussian orbitals pin the on-site dipole against its closed form and the Löwdin
formula against its exact properties (Hermiticity, k-derivative, translation invariance, the
compact-orbital limit); the Si ONCV fixture of the non-local velocity tests checks the real
basis path and :func:`position_dipole_operator`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.nonlocal_velocity import (
    PAOCatalog,
    PAOChannelData,
    PAOOrbitalEntry,
    PAOSiteData,
    PAOSpeciesData,
    load_pao_orbitals,
    qe_m_index_to_std,
)
from PAOFLOW.hamiltonian.position_dipoles import (
    assemble_position_kspace,
    build_position_tables,
    dipole_velocity_term,
    enumerate_pao_pairs,
    hermiticity_residual,
    lowdin_dipole_kspace,
    on_site_dipoles,
    overlap_sqrt_and_derivative,
    position_dipole_operator,
    tesseral_signs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'
SI_ONCV_UPF = 'Si_ONCV_PBE_sr.UPF'


# ---------------------------------------------------------------------------
# Synthetic Gaussian catalogs: normalised s and p orbitals.
# ---------------------------------------------------------------------------


def _gaussian_species(label: str, exponents: dict[str, float], r: np.ndarray) -> PAOSpeciesData:
    """Species with one normalised Gaussian channel per entry of ``exponents`` ('S' or 'P')."""
    channels = []
    for shell, alpha in exponents.items():
        if shell == 'S':
            norm = (2.0 * alpha / np.pi) ** 0.75  # f = N exp(-a r^2)
            radial = np.sqrt(4.0 * np.pi) * norm * np.exp(-alpha * r * r)
            l = 0
        else:
            norm = np.sqrt(2.0 * (2.0 * alpha) ** 2.5 / np.pi**1.5)  # f = N x exp(-a r^2)
            radial = np.sqrt(4.0 * np.pi / 3.0) * norm * r * np.exp(-alpha * r * r)
            l = 1
        channels.append(
            PAOChannelData(label='1' + shell, l=l, R_radial=radial, wfc=r * radial, occupation=0.0)
        )
    return PAOSpeciesData(
        label=label,
        pseudo_file='',
        upf=None,
        r=r,
        rab=np.full_like(r, r[1] - r[0]),
        channels=channels,
    )


def _catalog(species_of_site: list[PAOSpeciesData], tau: np.ndarray) -> PAOCatalog:
    sites, basis, offset = [], [], 0
    for site_index, species in enumerate(species_of_site):
        orbitals = []
        for channel_index, channel in enumerate(species.channels):
            for qe_m in range(1, 2 * channel.l + 2):
                orbitals.append(
                    PAOOrbitalEntry(
                        basis_index=offset + len(orbitals),
                        site_index=site_index,
                        channel_index=channel_index,
                        l=channel.l,
                        m=qe_m_index_to_std(qe_m, channel.l),
                        qe_m=qe_m,
                        label=channel.label,
                    )
                )
        sites.append(
            PAOSiteData(
                index=site_index,
                label=species.label,
                tau=np.asarray(tau[site_index], float),
                species=species,
                orbitals=orbitals,
                basis_offset=offset,
            )
        )
        basis += orbitals
        offset += len(orbitals)
    return PAOCatalog(
        species={s.label: s for s in species_of_site}, sites=sites, basis=basis, total_nlm=offset
    )


@pytest.fixture(scope='module')
def radial_mesh():
    return np.linspace(0.0, 12.0, 2401)


def test_on_site_s_p_dipole_matches_closed_form(radial_mesh):
    """One site with s (alpha) and p (beta) Gaussians: <s|z|p_z> = N_s N_p pi^{3/2} / (2 gamma^{5/2})."""
    alpha, beta = 0.6, 0.4
    species = _gaussian_species('A', {'S': alpha, 'P': beta}, radial_mesh)
    catalog = _catalog([species], np.zeros((1, 3)))
    cell = 40.0 * np.eye(3)  # no periodic images within the cutoffs
    pairs = enumerate_pao_pairs(catalog, cell)
    assert len(pairs) == 1 and pairs[0].deltaR_lattice == (0, 0, 0)
    tables = build_position_tables(catalog, pairs)
    overlap, dipole = tables.overlap[0], tables.dipole[0]
    np.testing.assert_allclose(overlap, np.eye(4), atol=2e-4)  # normalised, s _|_ p
    norm_s = (2.0 * alpha / np.pi) ** 0.75
    norm_p = np.sqrt(2.0 * (2.0 * beta) ** 2.5 / np.pi**1.5)
    closed = norm_s * norm_p * np.pi**1.5 / (2.0 * (alpha + beta) ** 2.5)
    # basis order: s, p_z (m=0), p_x (m=+1), p_y (m=-1)
    assert dipole[2, 0, 1] == pytest.approx(closed, rel=2e-3)
    assert dipole[0, 0, 2] == pytest.approx(closed, rel=2e-3)
    assert dipole[1, 0, 3] == pytest.approx(closed, rel=2e-3)
    assert abs(dipole[0, 0, 1]) < 1e-8 and abs(dipole[1, 0, 1]) < 1e-8  # Gaunt selection
    assert abs(dipole[2, 0, 0]) < 1e-8 and abs(dipole[2, 1, 1]) < 1e-8  # parity
    np.testing.assert_allclose(dipole, np.transpose(dipole, (0, 2, 1)), atol=1e-10)
    # S = 1: the Löwdin formula reduces to the on-site block at any k
    k = np.array([[0.1, -0.2, 0.3], [0.0, 0.0, 0.0]])
    on_site = on_site_dipoles(tables)
    full = lowdin_dipole_kspace(*assemble_position_kspace(tables, k), tables.tau_orbitals)
    intra = lowdin_dipole_kspace(
        *assemble_position_kspace(tables, k),
        tables.tau_orbitals,
        level='intra-atomic',
        on_site=on_site,
    )
    np.testing.assert_allclose(full, intra, atol=2e-3)
    np.testing.assert_allclose(intra[0], on_site, atol=1e-12)


@pytest.fixture(scope='module')
def two_site_tables(radial_mesh):
    """Two s+p sites 2.6 bohr apart in a 7 bohr cubic cell: overlapping, k-dependent S(k)."""
    species = _gaussian_species('B', {'S': 0.3, 'P': 0.25}, radial_mesh)
    tau = np.array([[0.0, 0.0, 0.0], [1.5, 1.2, 1.7]])
    catalog = _catalog([species, species], tau)
    pairs = enumerate_pao_pairs(catalog, 7.0 * np.eye(3), pao_tol=1e-3)
    return catalog, build_position_tables(catalog, pairs, n_q=400)


def test_pair_list_is_symmetric_and_has_on_site_pairs(two_site_tables):
    _, tables = two_site_tables
    keys = {(p.bra_site, p.ket_site, p.deltaR_lattice) for p in tables.pairs}
    assert (0, 0, (0, 0, 0)) in keys and (1, 1, (0, 0, 0)) in keys
    for bra, ket, (n1, n2, n3) in keys:
        assert (ket, bra, (-n1, -n2, -n3)) in keys
    assert len(keys) == len(tables.pairs)


def test_overlap_gradient_matches_finite_differences(two_site_tables):
    _, tables = two_site_tables
    k = np.array([[0.17, -0.05, 0.31]])
    overlap_k, gradient_k, _ = assemble_position_kspace(tables, k)
    assert np.abs(overlap_k - np.conj(np.transpose(overlap_k, (0, 2, 1)))).max() < 1e-6
    step = 1e-5
    for alpha in range(3):
        shift = np.zeros(3)
        shift[alpha] = step
        plus, _, _ = assemble_position_kspace(tables, k + shift)
        minus, _, _ = assemble_position_kspace(tables, k - shift)
        np.testing.assert_allclose(gradient_k[:, alpha], (plus - minus) / (2 * step), atol=1e-6)
    sqrt_k, inverse_sqrt_k, sqrt_gradient_k = overlap_sqrt_and_derivative(overlap_k, gradient_k)
    np.testing.assert_allclose(sqrt_k @ sqrt_k, overlap_k, atol=1e-10)
    np.testing.assert_allclose(
        sqrt_k @ inverse_sqrt_k, np.eye(overlap_k.shape[1])[None], atol=1e-10
    )
    for alpha in range(3):
        shift = np.zeros(3)
        shift[alpha] = step
        plus, _, _ = assemble_position_kspace(tables, k + shift)
        minus, _, _ = assemble_position_kspace(tables, k - shift)
        values_p, vectors_p = np.linalg.eigh(plus[0])
        values_m, vectors_m = np.linalg.eigh(minus[0])
        sqrt_p = (vectors_p * np.sqrt(values_p)) @ vectors_p.conj().T
        sqrt_m = (vectors_m * np.sqrt(values_m)) @ vectors_m.conj().T
        np.testing.assert_allclose(
            sqrt_gradient_k[0, alpha], (sqrt_p - sqrt_m) / (2 * step), atol=1e-6
        )


def test_lowdin_dipole_is_hermitian_and_translation_invariant(two_site_tables):
    _, tables = two_site_tables
    k = np.array([[0.0, 0.0, 0.0], [0.23, -0.41, 0.07], [0.5, 0.5, 0.5]])
    overlap_k, gradient_k, dipole_k = assemble_position_kspace(tables, k)
    dipole = lowdin_dipole_kspace(overlap_k, gradient_k, dipole_k, tables.tau_orbitals)
    scale = np.abs(dipole).max()
    assert scale > 0.05  # overlapping Gaussians give sizeable dipoles
    assert hermiticity_residual(dipole) < 1e-4 * scale
    shifted = lowdin_dipole_kspace(
        overlap_k, gradient_k, dipole_k, tables.tau_orbitals + np.array([3.1, -2.2, 0.4])
    )
    np.testing.assert_allclose(shifted, dipole, atol=1e-10)
    # the velocity term is zero on the diagonal of any eigenbasis of H
    rng = np.random.default_rng(3)
    nawf = tables.nawf
    hamiltonian = rng.normal(size=(3, nawf, nawf)) + 1j * rng.normal(size=(3, nawf, nawf))
    hamiltonian = hamiltonian + np.conj(np.transpose(hamiltonian, (0, 2, 1)))
    energies, vectors = np.linalg.eigh(hamiltonian)
    term = dipole_velocity_term(hamiltonian, dipole)
    band = np.einsum('kim,kaij,kjn->kamn', vectors.conj(), term, vectors)
    expected = (
        1j
        * (energies[:, None, :, None] - energies[:, None, None, :])
        * np.einsum('kim,kaij,kjn->kamn', vectors.conj(), dipole, vectors)
    )
    np.testing.assert_allclose(band, expected, atol=1e-10)
    assert np.abs(np.einsum('kann->kan', band)).max() < 1e-10


def test_full_level_reduces_to_intra_atomic_for_compact_orbitals(radial_mesh):
    """Compact Gaussians far apart: S -> 1 and the Löwdin terms vanish."""
    species = _gaussian_species('C', {'S': 2.0, 'P': 1.5}, radial_mesh)
    tau = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    catalog = _catalog([species, species], tau)
    tables = build_position_tables(catalog, enumerate_pao_pairs(catalog, 12.0 * np.eye(3)))
    k = np.array([[0.1, 0.2, -0.3]])
    on_site = on_site_dipoles(tables)
    full = lowdin_dipole_kspace(*assemble_position_kspace(tables, k), tables.tau_orbitals)
    np.testing.assert_allclose(full[0], on_site, atol=1e-5)


# ---------------------------------------------------------------------------
# Real Si basis (ONCV pseudo-wavefunctions 3s, 3p) through load_pao_orbitals.
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def si_data_controller():
    if not (SI_ONCV_DIR / SI_ONCV_UPF).exists():
        pytest.skip(f'Si ONCV UPF missing under {SI_ONCV_DIR}')
    alat = 10.2
    a_vectors = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])
    tau = np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3])
    arrays = {
        'species': [('Si', SI_ONCV_UPF)],
        'atoms': ['Si', 'Si'],
        'tau': tau,
        'a_vectors': a_vectors,
    }
    attributes = {'fpath': str(SI_ONCV_DIR), 'alat': alat, 'dftSO': False}

    class DC:
        def data_dicts(self):
            return arrays, attributes

    return DC(), alat, a_vectors


def test_si_on_site_dipole_equals_radial_integral(si_data_controller):
    dc, alat, a_vectors = si_data_controller
    catalog = load_pao_orbitals(dc)
    pairs = [p for p in enumerate_pao_pairs(catalog, a_vectors * alat) if p.distance < 1e-8]
    assert len(pairs) == 2
    tables = build_position_tables(catalog, pairs)
    species = catalog.species['Si']
    s_channel = next(c for c in species.channels if c.l == 0)
    p_channel = next(c for c in species.channels if c.l == 1)
    # <s|z|p_z> = (1/sqrt 3) int R_s R_p r^3 dr
    radial = np.trapezoid(s_channel.R_radial * p_channel.R_radial * species.r**3, species.r)
    s_index = next(o.basis_index for o in catalog.sites[0].orbitals if o.l == 0)
    pz_index = next(o.basis_index for o in catalog.sites[0].orbitals if o.l == 1 and o.m == 0)
    assert tables.dipole[0][2, s_index, pz_index] == pytest.approx(radial / np.sqrt(3.0), rel=3e-3)
    assert abs(tables.dipole[0][0, s_index, pz_index]) < 1e-8
    assert abs(radial) > 0.1  # a real on-site dipole, bohr


def test_si_position_dipole_operator(si_data_controller):
    dc, alat, a_vectors = si_data_controller
    bg = np.linalg.inv(a_vectors).T
    for level in ('full', 'intra-atomic'):
        dipoles = position_dipole_operator(dc, bg, alat, level=level, pao_tol=1e-2, n_q=300)
        k_cryst = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.25], [1.1, -0.6, 0.3]])
        dipole = dipoles(k_cryst)
        assert dipole.shape == (3, 3, 8, 8)
        assert hermiticity_residual(dipole) < 1e-12
        # periodic in k: the folded point equals the unfolded one
        np.testing.assert_allclose(
            dipole[2], dipoles(k_cryst[2:3] - np.array([1, -1, 0]))[0], atol=1e-12
        )
        if level == 'intra-atomic':
            np.testing.assert_allclose(dipole[0], dipole[1], atol=1e-12)
    catalog = load_pao_orbitals(dc)
    signs = tesseral_signs(catalog)
    assert (
        signs.tolist() == [1.0, -1.0, 1.0, 1.0] * 2
    )  # s, p_z, p_x, p_y per site: (-1)^l (-1)^(odd m)
