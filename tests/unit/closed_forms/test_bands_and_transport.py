"""Density of states, band curvature, Boltzmann transport and doping against closed forms.

Each test runs a PAOFLOW property on a tight-binding model whose answer is known exactly:
the van Hove DOS of a chain, the curvature of the eigenvalues, the Drude weight and the laws of
Wiedemann-Franz and Mott, the chemical potential of a degenerate electron gas.
"""

from __future__ import annotations

import numpy as np
import pytest
from _models import bloch, model_run, output
from scipy import constants as C
from scipy.optimize import brentq

from PAOFLOW.models.models import parabolic_model, single_band_model

pytestmark = pytest.mark.unit

BOHR = C.physical_constants['Bohr radius'][0]


def cubic_band(t: float = 1.0, alat: float = 5.0, nelec: float = 1.0):
    """One s band on a simple-cubic lattice: :math:`-2t\\sum_\\alpha\\cos k_\\alpha a`."""
    hops = {(1, 0, 0): -t, (0, 1, 0): -t, (0, 0, 1): -t}
    return single_band_model(hops, np.eye(3), alat, nelec, (3, 3, 3))


def test_chain_dos_is_the_van_hove_form():
    r""":math:`g(E) = 1/(\pi\sqrt{4t^2 - E^2})` per band and cell, normalized to one state."""
    chain = single_band_model({(1, 0, 0): -1.0}, np.diag([1.0, 10.0, 10.0]), 5.0, 1.0, (3, 1, 1))
    p = model_run(chain, (4000, 1, 1), smearing=None)
    p.dos(emin=-3.0, emax=3.0, ne=601, delta=0.01)
    energy, dos = output(p, 'dos_0.dat').T
    assert np.trapezoid(dos, energy) == pytest.approx(1.0, abs=1e-5)
    inside = np.abs(energy) < 1.5
    exact = 1.0 / (np.pi * np.sqrt(4.0 - energy[inside] ** 2))
    np.testing.assert_allclose(dos[inside], exact, rtol=2e-4)


def test_band_curvature_is_the_second_derivative_of_the_bands():
    """``topology(eff_mass=True)``: the inverse-mass tensor of both bands of the two-band model, along
    the path, equals finite differences of the exact eigenvalues (eV bohr^2)."""
    model = parabolic_model(n_bands=2)
    p = model_run(model)
    p.bands(ibrav=0, nk=41, band_path='G-X', high_sym_points={'G': (0, 0, 0), 'X': (0.5, 0, 0)})
    p.topology(eff_mass=True, Berry=False, ipol=0, jpol=0, spol=2)
    curvature = output(p, 'effmass_xx_0.dat')
    kq = p.data_controller.data_arrays['kq']
    step = 1e-4
    for ik in (0, 7, 20, 33):
        k = kq[:, ik] * 2 * np.pi / model.alat
        bands = [np.linalg.eigvalsh(bloch(model, k + [s, 0, 0])) for s in (-step, 0.0, step)]
        exact = (bands[0] - 2 * bands[1] + bands[2]) / step**2
        np.testing.assert_allclose(curvature[ik, 1:3], exact, rtol=1e-4, atol=1e-3)


@pytest.fixture(scope='module')
def cubic_transport():
    """sigma, S, kappa of a half-filled cubic band at 2000 K on three chemical potentials."""
    model = cubic_band()
    p = model_run(model, (48, 48, 48), smearing=None)
    p.gradient_and_momenta()
    temperature, mus = 2000.0, np.array([-3.05, -3.0, -2.95])
    p.transport(tmin=temperature, tmax=temperature, nt=1, emin=mus[0], emax=mus[-1], ne=3,
                save_tensors=True, write_to_file=False)  # fmt: skip
    arrays = p.data_controller.data_arrays
    tensors = {key: arrays[key][0, 0, :].copy() for key in ('sigma', 'S', 'kappa')}
    return model, temperature, mus, tensors


def test_conductivity_is_the_drude_weight(cubic_transport):
    r""":math:`\sigma_{xx}/\tau = (2e^2/\Omega)\langle v_x^2(-\partial f/\partial E)\rangle_k`, and by parts
    :math:`(2e^2/\Omega)\langle f\,\partial^2\varepsilon/\hbar^2\partial k_x^2\rangle_k` (SI, per second of
    relaxation time)."""
    model, temperature, mus, tensors = cubic_transport
    n, a, t = 48, model.alat * BOHR, 1.0
    k = 2 * np.pi * np.arange(n) / n
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    energy = -2 * t * (np.cos(kx) + np.cos(ky) + np.cos(kz))
    velocity = 2 * t * np.sin(kx) * a * C.e / C.hbar
    curvature = 2 * t * np.cos(kx) * a**2 * C.e / C.hbar**2
    volume = model.volume * BOHR**3
    kT = C.k * temperature / C.e
    f = 1 / (np.exp((energy - mus[1]) / kT) + 1)
    surface = 2 * C.e**2 / volume * np.mean(velocity**2 * f * (1 - f) / (kT * C.e))
    sea = 2 * C.e**2 / volume * np.mean(f * curvature)
    assert tensors['sigma'][1] == pytest.approx(surface, rel=1e-6)
    assert tensors['sigma'][1] == pytest.approx(sea, rel=1e-4)


def test_wiedemann_franz_and_mott(cubic_transport):
    r"""Lorenz number :math:`\kappa/\sigma T \to \pi^2 k_B^2/3e^2` (degenerate, up to
    :math:`O((k_BT/W)^2)`) and the Mott formula :math:`S = -(\pi^2k_B^2T/3e)\,d\ln\sigma/dE`."""
    _model, temperature, mus, tensors = cubic_transport
    sigma, seebeck, kappa = tensors['sigma'], tensors['S'], tensors['kappa']
    lorenz = kappa[1] / (sigma[1] * temperature)
    assert lorenz == pytest.approx(np.pi**2 / 3 * (C.k / C.e) ** 2, rel=0.02)
    slope = (np.log(sigma[2]) - np.log(sigma[0])) / ((mus[2] - mus[0]) * C.e)
    mott = -(np.pi**2 * C.k**2 * temperature / (3 * C.e)) * slope
    assert seebeck[1] < 0  # electrons below half filling
    assert seebeck[1] == pytest.approx(mott, rel=1e-3)


def test_doping_of_a_degenerate_electron_gas():
    r"""``doping``: an empty cubic band with :math:`10^{20}` electrons per cm^3. The chemical potential
    counts the electrons exactly on the k mesh, and follows the free-electron gas
    :math:`E_F = \hbar^2k_F^2/2m^*`, :math:`n = k_F^3/3\pi^2`, with the Sommerfeld correction
    :math:`-(\pi^2/12)(k_BT)^2/E_F` (the cosine band is parabolic to about 1 % at this filling)."""
    t, alat, n = 1.0, 5.0, 64
    model = cubic_band(t, alat, nelec=0.0)
    p = model_run(model, (n, n, n), smearing=None)
    conc = -1.0e20
    temperatures = np.array([300.0, 600.0])
    mu = p.doping(tmin=300, tmax=600, nt=2, emin=-6.5, emax=-4.0, ne=2501, doping_conc=conc,
                  delta=0.005)  # fmt: skip
    electrons = abs(conc) * model.volume * (BOHR * 100) ** 3
    bands = p.data_controller.data_arrays['E_k'][:, 0, 0]
    for T, value in zip(temperatures, mu):
        kT = C.k * T / C.e
        count = brentq(lambda x: 2 * np.mean(1 / (np.exp(np.clip((bands - x) / kT, -700, 700)) + 1))
                       - electrons, -8, 0)  # fmt: skip
        assert value == pytest.approx(count, abs=2e-4)
    k_fermi = (3 * np.pi**2 * abs(conc) * 1e6) ** (1 / 3)  # 1/m
    fermi = t * (k_fermi * alat * BOHR) ** 2  # hbar^2 k_F^2 / 2 m*, m* = hbar^2 / (2 t a^2)
    kT = C.k * temperatures[0] / C.e
    assert mu[0] == pytest.approx(
        -6 * t + fermi * (1 - np.pi**2 / 12 * (kT / fermi) ** 2), abs=3e-3
    )
