"""Radial tables, interpolation and spherical harmonics of the bare pseudopotential terms."""

import numpy as np
import pytest
from scipy.special import eval_legendre

from PAOFLOW.elphon.pseudopotential_terms import (
    LocalPotentialTable,
    average_relativistic_projectors,
    lagrange4,
    qe_simpson,
    radial_cutoff_index,
    real_spherical_harmonics_qe,
)

from .toy_plane_waves import toy_upf

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('mesh', [101, 100])
def test_qe_simpson_integrates_polynomials(mesh):
    r = np.linspace(0.0, 2.0, mesh)
    rab = np.full(mesh, r[1] - r[0])
    assert qe_simpson(r**2, rab) == pytest.approx(8.0 / 3.0, rel=1e-6)
    assert qe_simpson(r**3, rab) == pytest.approx(4.0, rel=1e-6)


def test_radial_cutoff_index_is_odd_and_past_rcut():
    r = 1e-3 * np.exp(0.01 * np.arange(1000))
    msh = radial_cutoff_index(r, 1.0)
    assert msh % 2 == 1 and r[msh - 2] <= 1.0 < r[msh]


def test_lagrange4_is_exact_on_cubics_and_zero_past_the_table():
    dq = 0.01
    q = np.arange(50) * dq
    table = 1.0 - 2.0 * q + 0.5 * q**2 - 3.0 * q**3
    x = np.array([0.0, 0.123, 0.2995, 0.4])
    assert np.allclose(lagrange4(table, x, dq), 1.0 - 2.0 * x + 0.5 * x**2 - 3.0 * x**3)
    assert lagrange4(table, np.array([0.49]), dq)[0] == 0.0


def test_spherical_harmonics_addition_theorem():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((40, 3)), rng.standard_normal((40, 3))
    ya, yb = real_spherical_harmonics_qe(3, a), real_spherical_harmonics_qe(3, b)
    cos = np.einsum('ni,ni->n', a, b) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
    for l in range(4):
        s = np.einsum('nm,nm->n', ya[:, l * l : (l + 1) ** 2], yb[:, l * l : (l + 1) ** 2])
        assert np.allclose(s, (2 * l + 1) / (4 * np.pi) * eval_legendre(l, cos))


def test_local_table_matches_direct_quadrature():
    upf = toy_upf(zp=2.0)
    omega = 100.0
    table = LocalPotentialTable(upf, omega, 5.0)
    q = np.array([0.734, 2.0155])
    r, rab, vloc = upf.r, upf.rab, 2.0 * upf.vloc
    from scipy.special import erf

    msh = radial_cutoff_index(r)
    direct = []
    for qq in q:
        aux = (r * vloc + 2.0 * 2.0 * erf(r)) * np.sin(qq * r) / qq
        direct.append(
            qe_simpson(aux[:msh], rab[:msh]) * 4 * np.pi / omega
            - 4 * np.pi / omega * 2.0 * 2.0 * np.exp(-(qq**2) / 4) / qq**2
        )
    assert np.allclose(table(q), direct, rtol=1e-6)
    assert table(np.array([0.0]))[0] == pytest.approx(
        qe_simpson(r[:msh] * (r[:msh] * vloc[:msh] + 4.0), rab[:msh]) * 4 * np.pi / omega
    )


def test_average_relativistic_projectors_pairs_j_channels():
    r = 1e-3 * np.exp(0.012 * np.arange(300))
    mk = lambda l, j, scale: {
        'l': l,
        'j': j,
        'wfc': scale * r * np.exp(-(r**2)),
        'cutoff_index': 300,
        'cutoff_radius': None,
        'label': '',
    }
    upf = type('U', (), {})()
    upf.beta = [mk(0, 0.5, 1.0), mk(1, 0.5, 1.0), mk(1, 1.5, 2.0)]
    upf.dion = np.diag([1.0, 2.0, 6.0]) / 2.0  # Hartree: Ry values 1, 2, 6
    out = average_relativistic_projectors(upf)
    assert out['lll'] == [0, 1]
    vionl = (2 * 6.0 + 1 * 2.0) / 3.0
    assert out['dion_ry'][1, 1] == pytest.approx(vionl)
    expected = (
        2 * np.sqrt(6.0 / vionl) * upf.beta[2]['wfc']
        + 1 * np.sqrt(2.0 / vionl) * upf.beta[1]['wfc']
    ) / 3.0
    assert np.allclose(out['betas'][1], expected)


def test_spin_orbit_coefficients_identity_for_s_and_hermitian_for_p():
    """``fcoef`` of an s projector is the spin identity; the spinor ``D`` of a p shell is Hermitian."""
    from PAOFLOW.elphon.pseudopotential_terms import (
        real_to_complex_harmonics,
        spin_orbit_coefficient,
    )

    rot = real_to_complex_harmonics(3)
    for s1 in (1, 2):
        for s2 in (1, 2):
            assert spin_orbit_coefficient(0, 0.5, 0, 0, s1, s2, rot) == pytest.approx(
                1.0 if s1 == s2 else 0.0
            )
    for j in (0.5, 1.5):
        d = np.array(
            [
                [
                    [spin_orbit_coefficient(1, j, mi, mk, s1, s2, rot) for mk in range(3)]
                    for mi in range(3)
                ]
                for s1 in (1, 2)
                for s2 in (1, 2)
            ]
        ).reshape(2, 2, 3, 3)
        full = np.block([[d[0, 0], d[0, 1]], [d[1, 0], d[1, 1]]])  # (spin x m) matrix
        assert np.allclose(full, full.conj().T)
        # trace = number of |j m_j> states projected onto this (l, j) manifold
        assert np.trace(full).real == pytest.approx(2 * j + 1)


def test_spin_orbit_projectors_from_a_relativistic_upf():
    from PAOFLOW.elphon.pseudopotential_terms import KleinmanBylanderProjectors

    r = 1e-3 * np.exp(0.012 * np.arange(300))
    mk = lambda l, j, scale: {
        'l': l,
        'j': j,
        'wfc': scale * r * np.exp(-(r**2)),
        'cutoff_index': 300,
        'cutoff_radius': None,
        'label': '',
    }
    upf = type('U', (), {})()
    upf.r, upf.rab = r, 0.012 * r
    upf.beta = [mk(0, 0.5, 1.0), mk(1, 0.5, 1.0), mk(1, 1.5, 2.0)]
    upf.dion = np.diag([1.0, 2.0, 6.0]) / 2.0
    upf.ptype, upf.has_augmentation, upf.has_spinorbit = 'NC', False, True
    kb = KleinmanBylanderProjectors({'X': upf}, ['X'], 100.0, 3.0, spin_orbit=True)
    assert kb.nproj == 1 + 3 + 3 and kb.D is None and kb.D_so.shape == (2, 2, 7, 7)
    full = np.block([[kb.D_so[0, 0], kb.D_so[0, 1]], [kb.D_so[1, 0], kb.D_so[1, 1]]])
    assert np.allclose(full, full.conj().T)
    assert kb.D_so[0, 0, 0, 0] == pytest.approx(1.0) and kb.D_so[0, 1, 0, 0] == 0.0
    scalar = KleinmanBylanderProjectors({'X': upf}, ['X'], 100.0, 3.0)
    assert scalar.nproj == 1 + 3  # the two j channels averaged into one p projector
