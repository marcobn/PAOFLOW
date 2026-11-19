"""Band curvature, Hall coefficient and Nernst tensor of a one-band cubic model."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf

from PAOFLOW.hamiltonian.do_d2Hd2k import energy_and_velocity_degenerate_groups

pytestmark = pytest.mark.unit

BOHR, CHARGE, KB = 0.529177210903e-10, 1.602176634e-19, 1.380649e-23
ALAT, HOPPING = 5.0, -1.0


def test_degenerate_groups_need_equal_energies():
    velocity_groups = [np.array([0, 1]), np.array([3, 4, 5])]
    energy_groups = [np.array([4, 5])]  # bands 0 and 1 share a velocity but not an energy
    groups = energy_and_velocity_degenerate_groups(velocity_groups, energy_groups)
    assert [g.tolist() for g in groups] == [[4, 5]]
    assert energy_and_velocity_degenerate_groups(velocity_groups, []) == []


def _cubic_run(tmp_path: Path, nelec: float, nfft: int, remote: float = 12.0):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import single_band_model

    hoppings = {(1, 0, 0): HOPPING, (0, 1, 0): HOPPING, (0, 0, 1): HOPPING}
    model = single_band_model(
        hoppings, np.eye(3), ALAT, nelec, (3, 3, 3), remote_band_energy=remote
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), smearing='gauss', verbose=False)
        p.interpolated_hamiltonian(nfft1=nfft, nfft2=nfft, nfft3=nfft)
        p.pao_eigh()
        p.gradient_and_momenta(band_curvature=True)
    return p


def test_curvature_is_not_swapped_with_a_flat_band(tmp_path):
    p = _cubic_run(tmp_path, 1.0, 20)
    arrays = p.data_controller.data_arrays
    energies = np.asarray(arrays['E_k'])[:, 0, 0]
    curvature = np.asarray(arrays['d2Ed2k'])  # (6, nk, bands, nspin), eV bohr^2
    k_crystal = np.asarray(arrays['kgrid']).T  # Cartesian 2 pi / alat, cubic: = crystal
    exact = 2 * abs(HOPPING) * ALAT**2 * np.cos(2 * np.pi * k_crystal)  # d2E/dk_a^2 of the band
    for axis in range(3):
        assert np.allclose(curvature[axis, :, 0, 0], exact[:, axis], atol=1e-6)
        assert np.allclose(curvature[axis, :, 1, 0], 0.0, atol=1e-6)  # the flat remote band
    assert energies.min() == pytest.approx(-6.0)


@pytest.fixture(scope='module')
def cubic_hall(tmp_path_factory):
    """The 40^3 cubic model with curvatures and adaptive widths, for the Hall and Nernst tests."""
    out = tmp_path_factory.mktemp('cubic_hall')
    p = _cubic_run(out, 1.0, 40, remote=30.0)
    with contextlib.redirect_stdout(io.StringIO()):
        p.adaptive_smearing(min_width=0.002)
    return p, out


def _fermi_level(p, nelec: float) -> float:
    energies = np.asarray(p.data_controller.data_arrays['E_k'])[:, 0, 0]
    return brentq(
        lambda mu: 2 * np.mean(0.5 * (1 - erf((energies - mu) / 0.002))) - nelec,
        energies.min(),
        energies.max(),
    )


def _transport(p, out: Path, mu: float, half_width: float, ne: int) -> dict[str, np.ndarray]:
    """xx (or xyz) components against mu at 300 K; constant tau, so sigma and N are per unit tau."""
    with contextlib.redirect_stdout(io.StringIO()):
        p.transport(
            tmin=300,
            tmax=300,
            nt=1,
            emin=mu - half_width,
            emax=mu + half_width,
            ne=ne,
            do_hall=True,
        )
    sigma = np.loadtxt(out / 'sigmagauss_0.dat')
    return {
        'mu': sigma[:, 1],
        'sigma': sigma[:, 2],
        'seebeck': np.loadtxt(out / 'Seebeckgauss_0.dat')[:, 2],
        'hall': np.loadtxt(out / 'hall_trace_gauss_0.dat')[:, 2],
        'nernst': np.loadtxt(out / 'nernst_gauss_0.dat')[:, 2],  # N_xyz
    }


@pytest.mark.parametrize('nelec, sign', [(0.01, -1.0), (1.99, 1.0)])
def test_hall_coefficient_parabolic_limit(cubic_hall, nelec, sign):
    p, out = cubic_hall
    hall = _transport(p, out, _fermi_level(p, nelec), 0.01, 3)['hall'][1]
    carriers = min(nelec, 2 - nelec) / (ALAT * BOHR) ** 3
    expected = sign / (carriers * CHARGE)  # 1/(ne): negative for electrons, positive for holes
    assert np.sign(hall) == sign
    assert hall / expected == pytest.approx(1.0, abs=0.15)  # 40^3 grid, 3 % non-parabolicity


def test_nernst_is_the_energy_derivative_of_the_hall_mobility(cubic_hall):
    """N = R_H nu + sigma^-1 nu_H = (pi^2/3)(k_B^2 T/|e|) d(R_H sigma)/dE for degenerate carriers.

    The Hall mobility R_H sigma of a parabolic band with constant tau does not depend on energy,
    so the two terms cancel (Sondheimer). The Hall mobility of the cosine band falls with energy:
    N is small, positive for electrons near the band bottom, and its ratio to the cancelling terms
    shrinks with the filling.
    """
    p, out = cubic_hall
    ratios = []
    for nelec in (0.03, 0.01):
        r = _transport(p, out, _fermi_level(p, nelec), 0.05, 11)  # 10 meV mu grid, below k_B T
        slope = np.gradient(r['hall'] * r['sigma'], r['mu'] * CHARGE)[5]
        mott = np.pi**2 / 3 * KB**2 * 300 / CHARGE * slope
        nernst = r['nernst'][5]
        assert nernst > 0
        assert nernst / mott == pytest.approx(1.0, abs=0.03)
        ratios.append(nernst / (r['hall'][5] * r['sigma'][5] * r['seebeck'][5]))
    assert ratios[1] < 0.05
    assert ratios[1] < 0.6 * ratios[0]
