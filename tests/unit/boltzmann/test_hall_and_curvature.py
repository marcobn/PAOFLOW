"""Band curvature and Hall coefficient against the parabolic limit of a one-band cubic model."""

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

BOHR, CHARGE = 0.529177210903e-10, 1.602176634e-19
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


@pytest.mark.parametrize('nelec, sign', [(0.01, -1.0), (1.99, 1.0)])
def test_hall_coefficient_parabolic_limit(tmp_path, nelec, sign):
    p = _cubic_run(tmp_path, nelec, 40, remote=30.0)
    with contextlib.redirect_stdout(io.StringIO()):
        p.adaptive_smearing(min_width=0.002)
    energies = np.asarray(p.data_controller.data_arrays['E_k'])[:, 0, 0]
    fermi = brentq(
        lambda mu: 2 * np.mean(0.5 * (1 - erf((energies - mu) / 0.002))) - nelec,
        energies.min(),
        energies.max(),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p.transport(
            tmin=300, tmax=300, nt=1, emin=fermi - 0.01, emax=fermi + 0.01, ne=3, do_hall=True
        )
    hall = np.loadtxt(tmp_path / 'hall_trace_gauss_0.dat')[1, 2]
    carriers = min(nelec, 2 - nelec) / (ALAT * BOHR) ** 3
    expected = sign / (carriers * CHARGE)  # 1/(ne): negative for electrons, positive for holes
    assert np.sign(hall) == sign
    assert hall / expected == pytest.approx(1.0, abs=0.15)  # 40^3 grid, 3 % non-parabolicity
