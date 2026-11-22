"""Spin splitting of quantum-oscillation orbits (PAOFLOW.topology.orbit_spin_splitting)."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs
from PAOFLOW.hamiltonian.magnetic_field import BOHR_MAGNETON_EV_PER_TESLA, ELECTRON_SPIN_G_FACTOR
from PAOFLOW.response.orbital_moment import moment_and_curvature
from PAOFLOW.topology.orbit_spin_splitting import (
    HBAR_OVER_E_TESLA_BOHR2,
    field_direction,
    loop_eigenphases,
    orbit_spin_splitting,
    pair_orbital_moment,
    read_orbit_outlines,
    resample_closed_contour,
    splitting_summary,
)

HOPPING, LATTICE = 1.0, 4.0  # eV, bohr
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def _spinor_s_band():
    """Spinor s orbital on a simple cubic lattice: H(k) = -2t sum_i cos(k_i a), spin-degenerate."""
    grid = 5
    hrs = np.zeros((2, 2, grid, grid, grid, 1), dtype=complex)
    for axis in range(3):
        for step in (1, -1):
            cell = [0, 0, 0]
            cell[axis] = step % grid
            hrs[:, :, cell[0], cell[1], cell[2], 0] = -HOPPING * np.eye(2)
    terms = bloch_terms_from_HRs(hrs, np.eye(3), LATTICE, np.zeros((2, 3)))
    return terms, np.eye(3) * LATTICE


def _circle(n_hat, radius, npoints=240):
    n_hat = np.asarray(n_hat, float) / np.linalg.norm(n_hat)
    first = np.cross(n_hat, [0.0, 1.0, 0.0] if abs(n_hat[1]) < 0.9 else [1.0, 0.0, 0.0])
    first /= np.linalg.norm(first)
    second = np.cross(n_hat, first)
    angle = np.linspace(0.0, 2.0 * np.pi, npoints, endpoint=False)
    return radius * (np.outer(np.cos(angle), first) + np.outer(np.sin(angle), second))


@pytest.mark.parametrize('polar', [0.0, 35.0])
def test_free_spin_band_gives_g_s_and_the_band_mass(polar):
    terms, lattice = _spinor_s_band()
    n_hat = field_direction(20.0, polar)
    spin = (
        BOHR_MAGNETON_EV_PER_TESLA
        * ELECTRON_SPIN_G_FACTOR
        * 0.5
        * np.tensordot(n_hat, PAULI, axes=(0, 0))
    )
    excess = 0.02  # eV above the band bottom
    radius = np.sqrt(excess / (HOPPING * LATTICE**2))
    points = _circle(n_hat, 1.15 * radius)  # off the contour: the Newton steps move it
    result = orbit_spin_splitting(
        terms, spin, None, np.zeros((2, 3)), lattice, points, n_hat, band=0,
        fermi_energy=-6.0 * HOPPING + excess, newton_steps=4,
    )  # fmt: skip
    summary = splitting_summary(result)
    band_mass = 2.0 * 13.605693122994 / (2.0 * HOPPING * LATTICE**2)
    assert result.max_energy_error_ev < 1e-8
    assert result.max_pair_splitting_ev < 1e-12
    assert summary['m_star'] == pytest.approx(band_mass, rel=5e-3)
    for level in ('spin', 'spin_atomic', 'total'):
        assert summary[f'g_{level}'] == pytest.approx(ELECTRON_SPIN_G_FACTOR, rel=1e-9)
        assert summary[f'g_{level}_prop'] == pytest.approx(ELECTRON_SPIN_G_FACTOR, rel=1e-9)
    assert summary['Rs_spin'] == pytest.approx(
        np.cos(0.5 * np.pi * ELECTRON_SPIN_G_FACTOR * summary['m_star']), abs=1e-9
    )
    assert abs(summary['berry_rotation']) < 1e-9


def test_scalar_band_takes_the_analytic_spin_factor():
    terms, lattice = _spinor_s_band()
    n_hat = field_direction(0.0, 0.0)
    radius = np.sqrt(0.02 / (HOPPING * LATTICE**2))
    result = orbit_spin_splitting(
        terms, None, None, np.zeros((2, 3)), lattice, _circle(n_hat, radius), n_hat, band=0,
        spinor=False, fermi_energy=-6.0 * HOPPING + 0.02, newton_steps=3, g_s=2.0,
    )  # fmt: skip
    assert splitting_summary(result)['g_total'] == pytest.approx(2.0, rel=1e-9)


def test_pair_moment_of_one_band_is_the_abelian_moment():
    rng = np.random.default_rng(3)
    nb = 7
    energies = np.sort(rng.normal(size=nb))
    velocities = rng.normal(size=(3, nb, nb)) + 1j * rng.normal(size=(3, nb, nb))
    velocities = 0.5 * (velocities + np.conjugate(np.swapaxes(velocities, 1, 2)))
    reference, _ = moment_and_curvature(energies, velocities, 1e-8)
    for n in range(nb):
        moment = pair_orbital_moment(energies, velocities, np.array([n]), 1e-8)
        assert np.allclose(moment[:, 0, 0].real, reference[n], atol=1e-12)
        assert np.allclose(moment[:, 0, 0].imag, 0.0, atol=1e-12)


def test_loop_eigenphases_are_gauge_invariant():
    rng = np.random.default_rng(5)
    npts = 9

    def unitary():
        q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
        return q * (np.diag(r) / np.abs(np.diag(r)))

    zeeman = []
    for _ in range(npts):
        m = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        zeeman.append(1e-4 * (m + m.conj().T))
    links = [unitary() for _ in range(npts)]
    thetas = rng.uniform(100.0, 2000.0, size=npts)
    reference = loop_eigenphases(zeeman, links, thetas)
    gauge = [unitary() for _ in range(npts)]
    rotated_z = [g.conj().T @ z @ g for g, z in zip(gauge, zeeman)]
    rotated_links = [gauge[j].conj().T @ links[j] @ gauge[(j + 1) % npts] for j in range(npts)]
    assert np.allclose(loop_eigenphases(rotated_z, rotated_links, thetas), reference, atol=1e-10)


def test_constant_zeeman_gives_the_accumulated_phase():
    splitting = 2.5e-4  # eV per tesla between the two levels
    thetas = np.full(10, 600.0)
    zeeman = [0.5 * splitting * PAULI[2]] * 10
    links = [np.eye(2, dtype=complex)] * 10
    expected = np.angle(np.exp(1j * splitting * thetas.sum()))
    assert loop_eigenphases(zeeman, links, thetas)[0] == pytest.approx(abs(expected), abs=1e-10)
    assert HBAR_OVER_E_TESLA_BOHR2 == pytest.approx(2.350517566e5, rel=1e-9)


def test_outline_files_of_pyskeaf_read_back(tmp_path):
    from PAOFLOW.pyskeaf.results import Orbit, SKEAFResult, write_orbit_outlines

    rng = np.random.default_rng(1)
    orbits = []
    for theta, phi, freq in ((0.3, 0.0, 1.25), (0.3, 0.6, 0.5)):
        contour = rng.normal(size=(17, 3))
        orbits.append(
            Orbit(
                theta=theta,
                phi=phi,
                frequency_kT=freq,
                freq_uncertainty_kT=0.0,
                effective_mass=1.0,
                rep_frequency_kT=freq + 0.001,
                rep_slice_index=4,
                outline_ang=contour,
                outline_au=contour * 0.529177209,
            )
        )
    result = SKEAFResult(config_filename='', bxsf_filename='', fermi_energy=0.0, orbits=orbits)
    path_ang, path_au = tmp_path / 'outline_ang.out', tmp_path / 'outline_au.out'
    write_orbit_outlines(result, path_ang, path_au)
    outlines = read_orbit_outlines(path_ang)
    assert len(outlines) == 2
    for outline, orbit in zip(outlines, orbits):
        assert outline.azimuthal_deg == pytest.approx(np.degrees(orbit.theta), abs=1e-6)
        assert outline.polar_deg == pytest.approx(np.degrees(orbit.phi), abs=1e-6)
        assert outline.frequency_kT == pytest.approx(orbit.frequency_kT, abs=1e-4)
        assert outline.slice_index == 4
        assert np.allclose(outline.points_inv_angstrom, orbit.outline_ang, rtol=1e-6, atol=1e-6)


def test_resampled_contour_is_equally_spaced():
    angle = np.linspace(0.0, 2.0 * np.pi, 13, endpoint=False) ** 1.3 / (2.0 * np.pi) ** 0.3
    contour = np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])
    resampled = resample_closed_contour(contour, 50)
    steps = np.linalg.norm(np.roll(resampled, -1, axis=0) - resampled, axis=1)
    assert resampled.shape == (50, 3)
    assert np.ptp(steps[:-1]) < 0.05 * np.mean(steps)
