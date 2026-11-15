"""Harmonic conventions, dipole Gaunt coefficients and radial integrals of the ARPES module."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import spherical_jn

from PAOFLOW.arpes.atomic_radial import (
    atomic_number,
    hydrogenic_radial,
    radial_table,
    slater_effective_charge,
    slater_radial,
)
from PAOFLOW.arpes.harmonics import dipole_gaunt_table, qe_harmonic, real_harmonics
from PAOFLOW.arpes.radial import RadialIntegrals, radial_integral
from PAOFLOW.hamiltonian._two_center import real_spherical_harmonic
from PAOFLOW.projection.do_atwfc_proj import calc_ylmg


def test_qe_harmonic_order_matches_calc_ylmg():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(8, 3))
    ylmg = calc_ylmg(v, np.linalg.norm(v, axis=1))
    for l in range(4):
        for m_qe in range(1, 2 * l + 2):
            m, sign = qe_harmonic(l, m_qe)
            assert np.allclose(ylmg[:, l * l + m_qe - 1], sign * real_spherical_harmonic(l, m, v))


def test_real_harmonics_are_orthonormal_on_the_sphere():
    n = 200000
    rng = np.random.default_rng(1)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    Y = np.concatenate([real_harmonics(l, v) for l in range(3)], axis=1)  # (n, 9)
    overlap = 4.0 * np.pi * Y.T @ Y / n
    assert np.allclose(overlap, np.eye(9), atol=0.03)


@pytest.mark.parametrize('l', [0, 1, 2, 3])
def test_dipole_gaunt_sum_rule_and_selection(l):
    # sum_{alpha, l', m'} G^2 = int Y_lm^2 (sum_alpha r_alpha^2 / r^2) dOmega = 1 for every m
    total = np.zeros(2 * l + 1)
    for lp in (l - 1, l + 1):
        if lp < 0:
            continue
        G = dipole_gaunt_table(l, lp)
        total += np.sum(G**2, axis=(0, 2))
    assert np.allclose(total, 1.0, atol=1e-10)
    # l -> l (no parity change) and l -> l +- 2 vanish
    assert np.allclose(dipole_gaunt_table(l, l), 0.0, atol=1e-12)
    assert np.allclose(dipole_gaunt_table(l, l + 2), 0.0, atol=1e-12)


def test_s_to_p_gaunt_is_the_polarisation_direction():
    G = dipole_gaunt_table(0, 1)[:, 0, :]  # (alpha, m')
    expected = np.zeros((3, 3))
    expected[0, 2] = expected[1, 0] = expected[2, 1] = 1.0 / np.sqrt(3.0)  # x->m=+1, y->-1, z->0
    assert np.allclose(G, expected)


def test_radial_integral_hydrogen_1s_against_quadrature():
    r = np.linspace(0.0, 40.0, 8001)
    chi = r * hydrogenic_radial(1, 1, 0, r)  # 2 r e^{-r}
    assert np.isclose(np.trapezoid(chi**2, r), 1.0, atol=1e-6)
    for k in (0.3, 1.0, 2.5):
        reference = quad(lambda x: 2.0 * x**3 * np.exp(-x) * spherical_jn(1, k * x), 0.0, 60.0)[0]
        value = radial_integral(r, chi, 1, k)[0]
        assert np.isclose(value, 1j * reference, rtol=1e-6, atol=1e-10)


def test_radial_integrals_table_interpolation_phase_and_fixed():
    r = np.linspace(0.0, 30.0, 3001)
    chi = r * slater_radial(14, 3, 1, r)
    functions = {('Si', '3P', 1): (r, chi, 1)}
    table = RadialIntegrals(functions, 3.0, nk=301)
    k = np.array([0.4, 1.1, 2.2])
    direct = radial_integral(r, chi, 2, k)
    assert np.allclose(table(('Si', '3P', 1), 2, k), direct, rtol=2e-3, atol=1e-6)
    shifted = RadialIntegrals(
        functions, 3.0, nk=301, phase_shifts={(('Si', '3P', 1), 0): np.exp(0.4j)}
    )
    assert np.allclose(shifted(('Si', '3P', 1), 0, k), np.exp(0.4j) * table(('Si', '3P', 1), 0, k))
    fixed = RadialIntegrals(functions, 3.0, nk=301, fixed={(('Si', '3P', 1), 2): 0.7 - 0.1j})
    assert np.allclose(fixed(('Si', '3P', 1), 2, k), 0.7 - 0.1j)


def test_slater_rules_and_atomic_tables():
    assert atomic_number('Si1') == 14 and atomic_number('C') == 6 and atomic_number('Pb') == 82
    # carbon 2p: Z_eff = 6 - (2 * 0.85 + 3 * 0.35) = 3.25 (Slater's textbook value)
    assert np.isclose(slater_effective_charge(6, 2, 1), 3.25)
    # sodium 3s: 11 - (2 * 1.0 + 8 * 0.85) = 2.2
    assert np.isclose(slater_effective_charge(11, 3, 0), 2.2)
    r, chi = radial_table('slater', 'C', 2, 1)
    assert np.isclose(np.trapezoid(chi**2, r), 1.0, atol=1e-3)
    r, chi = radial_table('hydrogenic', 'H', 2, 1)
    assert np.isclose(np.trapezoid(chi**2, r), 1.0, atol=1e-6)
