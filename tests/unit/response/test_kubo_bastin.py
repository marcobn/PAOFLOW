"""Kubo-Bastin kernels and PAOFLOW.kubo_response (Kubo-Bastin campaign P2/P3, 14 November 2026).

- the closed-form kernels against a brute-force quadrature of the Bastin formula, at T = 0 and T > 0;
- the clean limit of the sea term (Berry curvature) and the T -> 0 limit;
- kubo_response against the tutorial03 routes (same numbers, documented signs);
- physical limits: Haldane Chern number, Kane-Mele spin Chern number, time-reversal parity,
  Boltzmann (intraband) limit, custom operators.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from scipy.integrate import quad

from PAOFLOW.response import kubo_kernels as kk

from . import _kubo_cases as kc

pytestmark = pytest.mark.unit


def _random_system(seed: int = 3, nb: int = 6):
    rng = np.random.default_rng(seed)

    def hermitian():
        m = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        return (m + m.conj().T) / 2

    energies = np.sort(rng.normal(size=nb) * 1.5)
    energies[3] = energies[2] + 0.004  # near-degenerate pair (series branch)
    energies[-1] = energies[-2]  # exact degeneracy
    return energies, hermitian(), hermitian()


def _bastin_quadrature(energies, a, b, eta, mu, occupation):
    """-2 int f(e) Im Tr[A dG^R/de B delta(e - H)] with H diagonal, adaptive quadrature."""

    def integrand(eps):
        lorentz = (eta / np.pi) / ((eps - energies) ** 2 + eta**2)
        dgreen = -1.0 / (eps - energies + 1j * eta) ** 2
        return (
            -2.0 * occupation(eps) * np.imag(np.sum(a * dgreen[None, :] * b.T * lorentz[:, None]))
        )

    breaks = np.unique(np.concatenate([energies - 10 * eta, energies, energies + 10 * eta]))
    edges = np.concatenate([[-1.0e4, -50.0], breaks[breaks < mu + 5.0], [mu + 5.0]])
    edges = np.unique(np.concatenate([edges, [mu]]))
    return sum(
        quad(integrand, lo, hi, limit=400, epsabs=1e-13, epsrel=1e-11)[0]
        for lo, hi in zip(edges[:-1], edges[1:])
    )


def test_kernels_reproduce_the_bastin_formula_at_zero_temperature():
    energies, a, b = _random_system()
    eta = 0.05
    mu = np.array([-2.0, energies[2] + 0.002, energies[4], 2.5])
    pair = a * b.T
    surface, _ = kk.surface_kernel(energies, pair, mu, eta)
    sea = kk.sea_kernel(energies, pair, mu, eta)
    for m, got in zip(mu, surface + sea):
        want = _bastin_quadrature(energies, a, b, eta, m, lambda e, m=m: float(e < m))
        assert got == pytest.approx(want, rel=1e-9, abs=1e-10)


def test_thermal_convolution_reproduces_the_bastin_formula_at_finite_temperature():
    energies, a, b = _random_system(seed=7, nb=4)
    eta, kT = 0.05, 0.03
    mu = np.array([energies[1] + 0.01, 0.4])
    pair = a * b.T
    grid = kk.thermal_grid(mu, eta, kT)
    surface, _ = kk.surface_kernel(energies, pair, grid, eta)
    zero_t = surface + kk.sea_kernel(energies, pair, grid, eta)
    got = kk.thermal_average(grid, zero_t, mu, kT)
    for m, value in zip(mu, got):
        fermi = lambda e, m=m: 0.5 * (1.0 - np.tanh(0.5 * (e - m) / kT))  # noqa: E731
        want = _bastin_quadrature(energies, a, b, eta, m + 40 * kT, fermi)
        assert value == pytest.approx(want, rel=1e-7, abs=1e-10)


def test_sea_tends_to_the_berry_curvature_sum():
    energies, a, b = _random_system()
    energies = energies + np.array([0, 0, 0, 0.3, 0.2, 0.5])  # no degeneracies
    mu = np.array([-1.0, 0.1, 3.0])
    sea = kk.sea_kernel(energies, a * b.T, mu, 1.0e-5)
    omega = kk.berry_curvature(energies, a, b, 0.0, cutoff=1e-20)
    occupations = (energies[None, :] < mu[:, None]).astype(float)
    assert np.allclose(sea, -(occupations @ omega), rtol=1e-4, atol=1e-8)


def test_symmetric_and_antisymmetric_parts_separate():
    energies, a, _ = _random_system()
    # A = B: P is real and symmetric, no sea term
    assert np.allclose(
        kk.sea_kernel(energies, a * a.T, np.array([0.0, 1.0]), 0.05), 0.0, atol=1e-14
    )


@pytest.fixture(scope='module')
def reference():
    with np.load(kc.REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope='module')
def rashba():
    return kc.prepared('rashba', tempfile.mkdtemp())


def _close(got, want, rtol=1e-10, scale=None):
    """Agreement to rtol of ``scale`` (default: the largest value of ``want``). A component that
    vanishes by symmetry needs the scale of its response, not of its own round-off."""
    scale = np.abs(want).max() if scale is None else scale
    return np.allclose(got, want, rtol=rtol, atol=rtol * scale)


def test_kubo_response_reproduces_the_tutorial03_routes(rashba, reference):
    """Same kernels as Eq. 1 and Eq. 3. Conductivity and spin Hall have the same sign (the spin Hall
    value includes the electron charge, as in Guo 2008). Rashba-Edelstein is opposite: the old routes
    use dHksp = -hbar v as the velocity. The Eq. 3 AHC is opposite too: it is the value for the reversed
    magnetization."""
    emin, emax = kc.WINDOWS['rashba']
    kw = dict(emin=emin, emax=emax, ne=kc.NE, eta=0.01, regulariser=0.05, write=False)
    ref = lambda case, f: reference[f'rashba/{case}|{f}'][:, 1]  # noqa: E731
    r = rashba.kubo_response('charge_current', ['xx', 'xy'], terms=('surface', 'intrinsic'), **kw)
    assert _close(r['xx']['surface'], ref('cond_eq1', 'Cond_xx.dat'))
    assert _close(r['xy']['intrinsic'], -ref('ahc_eq3', 'AHC_xy.dat'))
    assert _close(r['xy']['intrinsic'], -ref('anomalous_Hall', 'ahcEf_xy.dat'))
    r = rashba.kubo_response('spin_density', ['yx', 'xy'], terms=('surface', 'intrinsic'), **kw)
    assert _close(r['yx']['surface'], -ref('ree_eq1', 'REE_EVEN_yx.dat'))
    # the T-odd xy part vanishes here (8e-24 against 6e-7 for the even part): compare at the scale
    # of the response
    even_scale = np.abs(ref('ree_eq1', 'REE_EVEN_yx.dat')).max()
    assert _close(r['xy']['intrinsic'], -ref('ree_eq3_todd', 'REE_ODD_xy.dat'), scale=even_scale)
    r = rashba.kubo_response('spin_current', ['zxy'], terms=('intrinsic',), **kw)
    assert _close(r['z_xy']['intrinsic'], ref('shc_eq3', 'SHC_EVEN_z_xy.dat'))


def test_terms_add_up_and_write_one_file_per_component(rashba):
    r = rashba.kubo_response(
        'charge_current', ['xy'], terms=('total', 'surface', 'sea', 'surface_intraband', 'surface_interband'),
        eta=0.05, emin=-3.0, emax=-1.0, ne=7,
    )  # fmt: skip
    xy = r['xy']
    assert np.allclose(xy['total'], xy['surface'] + xy['sea'])
    assert np.allclose(xy['surface'], xy['surface_intraband'] + xy['surface_interband'])
    path = os.path.join(
        rashba.data_controller.data_attributes['opath'], 'kubo_charge_current_xy.dat'
    )
    data = np.loadtxt(path)
    assert data.shape == (7, 6) and np.allclose(data[:, 1], xy['total'], rtol=1e-9)


def test_zero_temperature_limit(rashba):
    kw = dict(terms=('total',), eta=0.05, emin=-2.5, emax=-1.5, ne=3, write=False)
    cold = rashba.kubo_response('charge_current', ['xx', 'xy'], **kw)
    warm = rashba.kubo_response('charge_current', ['xx', 'xy'], temperature=20.0, **kw)
    hot = rashba.kubo_response('charge_current', ['xx', 'xy'], temperature=300.0, **kw)
    for c in ('xx', 'xy'):
        # Sommerfeld: relative change ~ (pi kT / eta)^2 / 6 = 2e-3 at 20 K
        assert _close(warm[c]['total'], cold[c]['total'], rtol=5e-3)
        assert np.max(np.abs(hot[c]['total'] / cold[c]['total'] - 1.0)) > 0.02


def test_custom_operator_is_the_spin_density(rashba):
    kw = dict(terms=('total', 'intrinsic'), eta=0.05, emin=-2.5, emax=-1.5, ne=3, write=False)
    spin = rashba.kubo_response('spin_density', ['yx'], **kw)['yx']
    sy = rashba.data_controller.data_arrays['Sj'][1]
    custom = rashba.kubo_response('custom', ['x'], operator=sy, **kw)['x']
    from PAOFLOW.response.kubo_units import BOHR_TO_CM

    for t in kw['terms']:
        assert np.allclose(custom[t] * BOHR_TO_CM, spin[t])


def test_haldane_chern_number():
    """Haldane model in its gap: 2 pi sigma_xy = 2 C (e^2/hbar, both spins), every route to it."""
    p = kc.prepared('haldane', tempfile.mkdtemp())
    r = p.kubo_response(
        'charge_current', ['xy'], terms=('total', 'sea', 'intrinsic'), eta=0.01,
        emin=-0.05, emax=0.05, ne=3, twoD=True, write=False,
    )  # fmt: skip
    for term in ('total', 'sea', 'intrinsic'):
        assert np.allclose(2 * np.pi * r['xy'][term], 2.0, atol=5e-3), term


def test_kane_mele_spin_chern_number(tmp_path):
    """Kane-Mele without Rashba coupling (S_z conserved): 2 pi sigma^z_xy = +1 in the gap (electron
    charge included, as the spin Hall conductivity of Guo 2008)."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import kane_mele_model

    model = kane_mele_model(rashba=0.0, delta=0.0)  # gap 6 sqrt(3) soc = 0.62 eV
    p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), smearing='gauss', verbose=False)
    p.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=1)
    p.pao_eigh()
    p.gradient_and_momenta()
    p.adaptive_smearing()
    r = p.kubo_response('spin_current', ['zxy'], terms=('total', 'intrinsic'), eta=0.005, emin=-0.02, emax=0.02, ne=3, twoD=True, write=False)  # fmt: skip
    for term in ('total', 'intrinsic'):
        assert np.allclose(2 * np.pi * r['z_xy'][term], 1.0, atol=0.01), (term, r['z_xy'][term])
    # time-reversal symmetric: no charge Hall response
    r = p.kubo_response('charge_current', ['xx', 'xy'], terms=('total',), eta=0.05, emin=-2.0, emax=-1.0, ne=3, write=False)  # fmt: skip
    assert np.abs(r['xy']['total']).max() < 1e-10 * np.abs(r['xx']['total']).max()


def test_intraband_term_is_the_boltzmann_conductivity(tmp_path):
    """Small broadening: pi sum_n v_n^2 L_n^2 -> sum_n v_n^2 delta(mu - e_n) / (2 eta), the Boltzmann
    conductivity with tau = hbar / (2 eta) (tutorial03 Eq. 4 with its adaptive delta function)."""
    from PAOFLOW import PAOFLOW

    p = PAOFLOW.PAOFLOW(model=kc.model('rashba'), outputdir=str(tmp_path), smearing='gauss', verbose=False)  # fmt: skip
    p.interpolated_hamiltonian(nfft1=160, nfft2=160, nfft3=1)
    p.pao_eigh()
    p.gradient_and_momenta()
    p.adaptive_smearing()
    eta = 0.08
    r = p.kubo_response('charge_current', ['xx'], terms=('surface_intraband',), eta=eta, emin=-2.8, emax=-1.2, ne=5, write=False)  # fmt: skip
    p.linear_response(response='cond', intraband=True, gamma=eta, a_tensor=[[0, 0]], eminH=-2.8, emaxH=-1.2, esize=5)  # fmt: skip
    boltzmann = np.loadtxt(os.path.join(str(tmp_path), 'Cond_xx.dat'))[:, 1]
    assert np.allclose(r['xx']['surface_intraband'], boltzmann, rtol=0.05)


def test_orbital_hall_without_spin_orbit_is_the_orbital_current_response(tmp_path):
    """Scalar s + p model: orbital_Hall (Berry form, delta = 0.05) is the intrinsic term."""
    p = kc.prepared('sp_square', str(tmp_path))
    p.orbital_Hall(o_tensor=[[0, 1, 2]], emin=-2.0, emax=2.0, ne=kc.NE)
    legacy = np.loadtxt(os.path.join(str(tmp_path), 'ohcEf_z_xy.dat'))[:, 1]
    r = p.kubo_response(
        'orbital_current', ['zxy'], terms=('intrinsic',), regulariser=0.05,
        emin=-2.0, emax=2.0, ne=kc.NE, write=False,
    )  # fmt: skip
    assert np.abs(legacy).max() > 10.0
    assert _close(r['z_xy']['intrinsic'], legacy, rtol=1e-6)


def test_band_sums_stop_at_the_projectability_shift(tmp_path):
    """DFT runs: the bands above the projectability shift (the shifted complement of the projection) are
    left out by default, as in dielectric_tensor; models (shift = inf) keep every band."""
    p = kc.prepared('kane_mele', str(tmp_path))
    attr = p.data_controller.data_attributes
    kw = dict(terms=('total', 'intrinsic'), eta=0.05, emin=-0.5, emax=0.5, ne=5, write=False)
    every = p.kubo_response('charge_current', ['xx'], **kw)['xx']
    assert attr['bnd'] == attr['nawf'] and np.isinf(attr['shift'])
    attr['shift'], attr['bnd'] = 2.5, 2  # as after a DFT projection with two bands below the shift
    default = p.kubo_response('charge_current', ['xx'], **kw)['xx']
    two = p.kubo_response('charge_current', ['xx'], bands=2, **kw)['xx']
    four = p.kubo_response('charge_current', ['xx'], bands=4, **kw)['xx']
    for t in kw['terms']:
        assert np.allclose(default[t], two[t]) and np.allclose(four[t], every[t])
    assert not np.allclose(default['total'], every['total'])
    o2 = p.kubo_optical_response(
        'charge_current', ['xx'], eta=0.05, omega_max=3.0, nomega=5, write=False
    )['xx']
    o4 = p.kubo_optical_response('charge_current', ['xx'], eta=0.05, omega_max=3.0, nomega=5, bands=4, write=False)['xx']  # fmt: skip
    assert not np.allclose(o2['interband'], o4['interband'])
