"""Three-dimensional electrostatics of finite-bias transport: Gaussian charges between grounded planes."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.fft import dst, idst
from scipy.optimize import brentq
from scipy.special import k0

from PAOFLOW.transport.negf.charge_model import (
    atomic_charge_widths,
    gaussian_width_of_self_energy,
    hartree_self_energy,
    valence_charge_width,
)
from PAOFLOW.transport.negf.contour import KB_EV, fermi_function
from PAOFLOW.transport.negf.electrostatics import ElectrostaticModel, PlanarLayers, poisson_kernel
from PAOFLOW.transport.negf.poisson_3d import HARTREE_EV, DirichletSlab, GaussianChargesInSlab

pytestmark = pytest.mark.unit

KT = KB_EV * 300.0


def test_width_from_the_hartree_self_energy():
    """Hydrogen 1s: J = 5/8 hartree; the Gaussian of the same self-energy has 1/(sqrt(pi) sigma) = J."""
    radii = np.linspace(0.0, 40.0, 200001)
    assert hartree_self_energy(radii, 4 * radii**2 * np.exp(-2 * radii)) == pytest.approx(
        0.625, rel=1e-7
    )
    upf = SimpleNamespace(
        r=radii, pswfc=[{'label': '1S', 'occ': 1.0, 'wfc': 2 * radii * np.exp(-radii)}]
    )
    assert valence_charge_width(upf) == pytest.approx(
        gaussian_width_of_self_energy(0.625), rel=1e-6
    )
    assert gaussian_width_of_self_energy(0.625) == pytest.approx(1 / (np.sqrt(np.pi) * 0.625))


def test_kernel_of_charges_between_grounded_plates():
    """Far apart, two Gaussians interact as point charges (mean-value property): the Fourier-Bessel series of
    the Green's function between plates. On site: 1/(sqrt(pi) sigma) plus the image series."""
    length, side, width = 20.0, 120.0, 1.2
    slab = DirichletSlab(np.diag([side, side]), 0.0, length)
    in_plane, heights = np.array([[0.0, 0.0], [6.0, 3.0]]), np.array([8.0, 12.5])
    kernel = GaussianChargesInSlab(slab, in_plane, heights, np.full(2, width)).kernel() / HARTREE_EV
    k = np.pi * np.arange(1, 4000) / length
    distance = np.linalg.norm(in_plane[1] - in_plane[0])
    green = 4 / length * np.sum(np.sin(k * heights[0]) * np.sin(k * heights[1]) * k0(k * distance))
    assert kernel[0, 1] == pytest.approx(green, rel=2e-5)
    images = np.arange(-200000, 200001)
    same_sign = 2 * images[images != 0] * length
    opposite = 2 * images * length - 2 * heights[0]
    on_site = (
        1 / (np.sqrt(np.pi) * width) + np.sum(1 / np.abs(same_sign)) - np.sum(1 / np.abs(opposite))
    )
    assert kernel[0, 0] == pytest.approx(on_site, rel=1e-7)


def test_planar_limit_of_the_kernel():
    """G = 0 only: the planar kernel of sheets, minus the self-energy (4 pi / A) sigma / sqrt(pi) of the Gaussian
    profile on the diagonal. The full kernel is symmetric positive definite."""
    heights = np.array([0.0, 2.0, 4.0, 6.0])
    slab = DirichletSlab(np.diag([5.0, 5.0]), -1.0, 7.0)
    charges = GaussianChargesInSlab(slab, np.zeros((4, 2)), heights, np.full(4, 0.2), sampling=0.1)
    layers = PlanarLayers(np.arange(4), heights, -1.0, 7.0, 25.0)
    profile_self_energy = 4 * np.pi / 25.0 * 0.2 / np.sqrt(np.pi) * HARTREE_EV
    expected = poisson_kernel(layers) - profile_self_energy * np.eye(4)
    assert np.allclose(charges.kernel(in_plane_cutoff=0.0), expected, atol=5e-5)
    full = charges.kernel()
    assert np.allclose(full, full.T) and np.all(np.linalg.eigvalsh(full) > 0)
    assert np.all(np.diag(full) > np.diag(expected))  # the in-plane structure adds on-site energy


def test_kernel_against_a_grid_solution():
    """Skewed transverse cell with overlapping periodic images: the analytic in-plane sums against FFT x DST on
    a real-space grid."""
    a1, a2, length = np.array([5.0, 0.0]), np.array([2.0, 4.6]), 12.0
    in_plane = np.array([[0.3, 0.2], [2.5, 2.4], [4.0, 1.0]])
    heights, widths = np.array([3.0, 6.1, 8.5]), np.array([1.4, 0.9, 1.7])
    charges = GaussianChargesInSlab(
        DirichletSlab(np.vstack([a1, a2]), 0.0, length), in_plane, heights, widths
    )
    n_plane, n_z = 40, 191
    spacing = length / (n_z + 1)
    z = spacing * np.arange(1, n_z + 1)
    fractions = np.stack(
        np.meshgrid(np.arange(n_plane) / n_plane, np.arange(n_plane) / n_plane, indexing='ij'), -1
    )
    points = fractions @ np.vstack([a1, a2])
    reciprocal = 2 * np.pi * np.linalg.inv(np.vstack([a1, a2])).T
    m = np.fft.fftfreq(n_plane, 1 / n_plane)
    g2 = np.sum((m[:, None, None] * reciprocal[0] + m[None, :, None] * reciprocal[1]) ** 2, -1)
    k = np.pi * np.arange(1, n_z + 1) / length
    volume = abs(np.linalg.det(np.vstack([a1, a2]))) / n_plane**2 * spacing

    def gaussian(centre, height, width):
        profile = sum(
            sign * np.exp(-((z - c) ** 2) / (2 * width**2))
            for c, sign in ((height, 1), (-height, -1), (2 * length - height, -1))
        )
        values = np.zeros((n_plane, n_plane, n_z))
        for i in range(-3, 4):
            for j in range(-3, 4):
                d = points - centre - i * a1 - j * a2
                values += np.exp(-np.sum(d**2, -1) / (2 * width**2))[:, :, None] * profile
        return values / (values.sum() * volume)

    def solve(density):
        spectrum = dst(np.fft.fft2(density, axes=(0, 1)), type=1, axis=2)
        spectrum *= 4 * np.pi / (g2[:, :, None] + k**2)
        return np.fft.ifft2(idst(spectrum, type=1, axis=2), axes=(0, 1)).real

    shapes = [gaussian(*arguments) for arguments in zip(in_plane, heights, widths)]
    grid = np.array([[np.sum(a * solve(b)) * volume for b in shapes] for a in shapes])
    assert np.allclose(charges.kernel(), grid * HARTREE_EV, rtol=1e-3)
    q = np.array([0.01, -0.02, 0.015])
    potential = solve(sum(qi * s for qi, s in zip(q, shapes))) * HARTREE_EV
    assert np.allclose(charges.potential_on_grid(q, (n_plane, n_plane), z), potential, atol=2e-4)
    assert np.allclose(charges.planar_average(q, z), potential.mean(axis=(0, 1)), atol=2e-4)


class _SingleLevel:
    """A level between two ideal chains: only the level is charged (charging energy U0), at a fraction eta of the
    way between the electrode potentials; the chains in the conductor follow their lead rigidly."""

    name = 'single_level'

    def __init__(self, group_of_orbital, eta, charging):
        self.group_of_orbital = group_of_orbital
        self.eta, self.charging = eta, charging

    n_groups = 3
    coordinates = np.arange(3.0)

    def external_potential(self, left, right):
        return np.array([left, self.eta * left + (1 - self.eta) * right, right])

    def kernel(self):
        return np.diag([0.0, self.charging, 0.0])


def _chain(tmp_path: Path, transverse=2.0, on_site=None, couplings=None, delta=1e-3):
    """One-band chain along z (t = -1 eV, 5 bohr), 8 sites in the conductor, leads of 2 sites."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.Transport import Transport
    from PAOFLOW.models.models import single_band_model

    model = single_band_model(
        {(0, 0, 1): -1.0}, np.diag([transverse, transverse, 1.0]), 5.0, 1.0, (1, 1, 4)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        p.doubling_Hamiltonian(0, 0, 3)
        hrs = p.data_controller.data_arrays['HRs']
        band = np.flatnonzero(np.abs(np.real(np.diag(hrs[:, :, 0, 0, 0, 0]))) < 1e-9)
        for site, energy in (on_site or {}).items():
            hrs[band[site], band[site], 0, 0, 0, 0] = energy
        for (a, b), hopping in (couplings or {}).items():
            hrs[band[a], band[b], 0, 0, 0, 0] = hrs[band[b], band[a], 0, 0, 0, 0] = hopping
        transport = Transport(p.data_controller)
        transport.define_partition(
            central_atoms='ALL', left_lead_layers=2, right_lead_layers=2, transport_direction='z'
        )
        transport.configure_outputs(output_dir=str(tmp_path), postfix='_3d')
        transport.configure_energy_grid(emin=-0.4, emax=0.4, ne=161, delta=delta, nk=(1, 1))
        transport.build_hamiltonian_blocks(
            calculation_type='conductor', use_sym=False, do_overlap_transformation=False
        )
    return transport


def test_single_level_with_charging(tmp_path):
    """Datta's single level with a charging energy, through a custom ElectrostaticModel: U, dq and I against the
    scalar self-consistent solution of the level between two semi-infinite chains (Datta, Quantum Transport:
    Atom to Transistor, ch. 1 and 3)."""
    from PAOFLOW.transport.negf.density_matrix import open_conductor_from
    from PAOFLOW.transport.negf.self_consistency import SelfConsistencySettings, finite_bias_scan

    level_energy, coupling, charging, eta, site = 0.2, -0.3, 1.0, 0.5, 3
    transport = _chain(
        tmp_path,
        on_site={site: level_energy},
        couplings={(site - 1, site): coupling, (site, site + 1): coupling},
        delta=1e-5,
    )
    atoms = transport._partition.central_atoms
    heights = np.array([atom.position[2] for atom in atoms])
    level_height = np.unique(np.round(heights, 6))[site]
    group = np.where(
        np.abs(heights - level_height) < 1e-6, 1, np.where(heights < level_height, 0, 2)
    )
    model = _SingleLevel(
        np.concatenate([np.full(a.dimension, g) for a, g in zip(atoms, group)]), eta, charging
    )
    assert isinstance(model, ElectrostaticModel)
    biases = [0.0, 0.5, -0.5]
    scan = finite_bias_scan(
        open_conductor_from(transport.conductor_data, transport.blc_blocks),
        model,
        SelfConsistencySettings(window_spacing=0.004, tolerance=1e-8),
        biases,
    )
    energies = np.linspace(-2.6, 2.6, 260001)

    def surface(z):  # semi-infinite chain with t = -1: the root with Im g <= 0
        root = np.sqrt(z - 2.0 + 0j) * np.sqrt(z + 2.0 + 0j)
        g = (z - root) / 2.0
        return np.where(g.imag > 0, (z + root) / 2.0, g)

    def level(shift, left, right):
        sigma_left = coupling**2 * surface(energies - left + 1e-12j)
        sigma_right = coupling**2 * surface(energies - right + 1e-12j)
        gamma_left, gamma_right = -2 * sigma_left.imag, -2 * sigma_right.imag
        green2 = np.abs(1 / (energies - level_energy - shift - sigma_left - sigma_right)) ** 2
        f_left = fermi_function(energies, left, KT).real
        f_right = fermi_function(energies, right, KT).real
        electrons = (
            np.trapezoid(green2 * (gamma_left * f_left + gamma_right * f_right), energies) / np.pi
        )
        current = np.trapezoid(gamma_left * gamma_right * green2 * (f_left - f_right), energies)
        return electrons, current

    reference = level(0.0, 0.0, 0.0)[0]
    for bias, point in zip(biases, scan.points):
        left, right = 0.5 * bias, -0.5 * bias
        external = eta * left + (1 - eta) * right
        shift = brentq(
            lambda u: u - external - charging * (level(u, left, right)[0] - reference),
            -2.0,
            2.0,
            xtol=1e-12,
        )
        electrons, current = level(shift, left, right)
        assert point.converged
        assert point.group_potential[1] == pytest.approx(shift, abs=1e-5)
        assert point.group_charge[1] == pytest.approx(electrons - reference, abs=1e-5)
        assert point.current == pytest.approx(current, rel=5e-4, abs=1e-8)
        assert np.allclose(point.group_potential[[0, 2]], [left, right])


def test_three_dimensional_scan_of_a_wire(tmp_path):
    """Chain with a two-site barrier, 3D electrostatics through the Transport API: I(-V) = -I(V), U mirror
    symmetric, linear response, zero bias = DFT Hamiltonian, Hamiltonian restored, outputs."""
    transport = _chain(tmp_path, on_site={3: 1.0, 4: 1.0})
    reference = np.array(transport.blc_blocks['blc_00C'].H, copy=True)
    transport.configure_bias_self_consistency(
        electrostatics='3d', charge_widths=1.5, window_spacing=0.004, tolerance=1e-6
    )
    scan = transport.compute_current_self_consistent(
        bias=[0.0, 0.1, -0.1, 0.005], potential_grid_biases=[0.1]
    )
    zero, forward, backward, small = scan.points
    assert scan.electrostatics == '3d' and scan.group_coordinates.shape == (16, 3)
    assert all(point.converged for point in scan.points)
    assert zero.current == 0.0 and np.allclose(zero.group_potential, 0.0)
    assert forward.current > 0 and forward.current == pytest.approx(-backward.current, rel=1e-5)
    order = np.lexsort((np.arange(16), -scan.group_coordinates[:, 2]))  # atoms mirrored along z
    assert np.allclose(forward.group_potential, backward.group_potential[order], atol=1e-5)
    assert np.allclose(forward.group_charge, backward.group_charge[order], atol=1e-6)
    energies, transmission = small.window_energies, small.transmission
    minus_df = -np.gradient(fermi_function(energies, 0.0, KT).real, energies)
    assert small.current / 0.005 == pytest.approx(
        np.trapezoid(transmission * minus_df, energies), rel=2e-3
    )
    assert np.array_equal(transport.blc_blocks['blc_00C'].H, reference)
    # planar average: the electrode potentials at the planes, the atom potentials in between
    average = forward.planar_average_potential
    assert average[0] == pytest.approx(0.05) and average[-1] == pytest.approx(-0.05)
    for name in ('current', 'potential', 'charge', 'transmission', 'potential_planar_average'):
        assert (tmp_path / f'{name}_self_consistent_3d.dat').exists()
    xsf = (tmp_path / 'potential_grid_self_consistent_3d_V+0.1000.xsf').read_text().split('\n')
    start = xsf.index(' BEGIN_DATAGRID_3D_potential_energy_eV')
    shape = [int(x) for x in xsf[start + 1].split()]
    values = np.array(' '.join(xsf[start + 6 : xsf.index(' END_DATAGRID_3D')]).split(), dtype=float)
    assert values.size == np.prod(shape)
    grid = values.reshape(shape[::-1])  # the first index runs fastest
    assert np.allclose(grid[0], 0.05) and np.allclose(grid[-1], -0.05)  # the electrode planes


def test_charge_widths_and_geometry_checks():
    from PAOFLOW.transport.negf.atomic_electrostatics import transverse_lattice_vectors

    assert np.allclose(atomic_charge_widths(['A', 'B'], 1.2), [1.2, 1.2])
    assert np.allclose(atomic_charge_widths(['A', 'B', 'A'], {'A': 1.0, 'B': 2.0}), [1, 2, 1])
    with pytest.raises(ValueError, match="no charge width for 'B'"):
        atomic_charge_widths(['A', 'B'], {'A': 1.0})
    with pytest.raises(ValueError, match='positive'):
        atomic_charge_widths(['A'], -1.0)
    cell = np.array([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 20.0]])
    assert np.allclose(transverse_lattice_vectors(cell, 2), cell[:2])
    with pytest.raises(NotImplementedError, match='perpendicular'):
        transverse_lattice_vectors(np.array([[5.0, 0, 1.0], [0, 5.0, 0], [0, 0, 20.0]]), 2)
    with pytest.raises(ValueError, match='between the electrode planes'):
        GaussianChargesInSlab(
            DirichletSlab(np.eye(2), 0.0, 1.0), np.zeros((1, 2)), np.array([1.5]), np.ones(1)
        )
