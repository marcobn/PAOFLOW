"""Finite-bias transport: contour, electrostatics, open-system density and the self-consistent scan."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.transport.negf.contour import (
    KB_EV,
    bias_window,
    equilibrium_contour,
    fermi_function,
)
from PAOFLOW.transport.negf.electrostatics import HARTREE_EV, PlanarLayers, planar_potential
from PAOFLOW.transport.negf.self_consistency import anderson_update

pytestmark = pytest.mark.unit

KT = KB_EV * 300.0


def test_contour_integrates_bound_states():
    """-Im sum_j w_j / (z_j - e) / pi is the occupation f(e) of a level e, for any e above the start."""
    levels = np.array([-5.0, -0.5, -0.05, 0.0, 0.02, 0.3, 2.0])
    for mu in (0.0, 0.4):
        contour = equilibrium_contour(-8.0, mu, KT)
        occupations = [
            -np.imag(np.sum(contour.weights / (contour.points - e))) / np.pi for e in levels
        ]
        assert np.allclose(occupations, fermi_function(levels, mu, KT).real, atol=1e-6)
    with pytest.raises(ValueError, match='below'):
        equilibrium_contour(0.5, 0.0, KT)


def test_bias_window():
    assert len(bias_window(0.1, 0.1, KT, 0.01)) == 0
    window = bias_window(0.05, -0.05, KT, 0.005)
    energies = window.points.real
    assert energies[0] < -0.05 - 9 * KT and energies[-1] > 0.05 + 9 * KT
    assert np.diff(energies).max() <= 0.005 + 1e-12
    assert window.weights.real.sum() == pytest.approx(energies[-1] - energies[0])


def test_planar_potential_of_a_sheet_charge():
    layers = PlanarLayers(
        layer_of_orbital=np.arange(4), coordinates=np.array([0.0, 2.0, 4.0, 6.0]),
        left_boundary=-1.0, right_boundary=7.0, area=10.0,
    )  # fmt: skip
    ramp = planar_potential(np.zeros(4), layers, 0.2, -0.2)
    assert np.allclose(ramp, 0.2 - 0.4 * (layers.coordinates + 1.0) / 8.0)
    charge = np.array([0.0, 0.01, 0.0, 0.0])
    potential = planar_potential(charge, layers, 0.0, 0.0)
    # Dirichlet Green's function of [a, b]: (z< - a)(b - z>)/(b - a), times 4 pi dq / A
    expected = 4 * np.pi * 0.01 / 10.0 * np.array([1 * 5, 3 * 5, 3 * 3, 3 * 1]) / 8.0 * HARTREE_EV
    assert np.allclose(potential, expected)
    assert np.all(potential > 0)  # extra electrons raise the potential energy


def test_anderson_update_solves_a_linear_fixed_point():
    rng = np.random.default_rng(0)
    matrix = 0.3 * rng.standard_normal((6, 6))
    offset = rng.standard_normal(6)
    exact = np.linalg.solve(np.eye(6) - matrix, offset)
    x, inputs, residuals = np.zeros(6), [], []
    for _ in range(30):
        residual = matrix @ x + offset - x
        if np.abs(residual).max() < 1e-10:
            break
        inputs, residuals = (inputs + [x])[-6:], (residuals + [residual])[-6:]
        x = anderson_update(inputs, residuals, 0.3)
    assert np.allclose(x, exact, atol=1e-9)


def _cubic_conductor(tmp_path: Path, impurity_sites=(), impurity=1.0, nk=3):
    """One-band cubic metal, 8 sites along z, leads of 2 sites, on-site barriers on `impurity_sites`."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.Transport import Transport
    from PAOFLOW.models.models import single_band_model

    model = single_band_model(
        {(1, 0, 0): -1.0, (0, 1, 0): -1.0, (0, 0, 1): -1.0}, np.eye(3), 5.0, 1.0, (4, 4, 4)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        p.doubling_Hamiltonian(0, 0, 3)
        hrs = p.data_controller.data_arrays['HRs']
        band = np.flatnonzero(np.abs(np.real(np.diag(hrs[:, :, 0, 0, 0, 0]))) < 1e-9)
        for site in impurity_sites:
            hrs[band[site], band[site], 0, 0, 0, 0] += impurity
        transport = Transport(p.data_controller)
        transport.define_partition(
            central_atoms='ALL', left_lead_layers=2, right_lead_layers=2, transport_direction='z'
        )
        transport.configure_outputs(output_dir=str(tmp_path), postfix='_fb')
        transport.configure_energy_grid(emin=-0.4, emax=0.4, ne=161, delta=1e-3, nk=(nk, nk))
        transport.build_hamiltonian_blocks(
            calculation_type='conductor', use_sym=False, do_overlap_transformation=False
        )
    return transport, band


def test_open_crystal_occupations_are_those_of_the_infinite_crystal(tmp_path):
    from PAOFLOW.transport.negf.density_matrix import open_conductor_from, open_system_density

    transport, band = _cubic_conductor(tmp_path)
    system = open_conductor_from(transport.conductor_data, transport.blc_blocks)
    empty = bias_window(0.0, 0.0, KT, 0.01)
    k = 2 * np.pi * np.arange(3) / 3
    e_parallel = (-2 * (np.cos(k)[:, None] + np.cos(k)[None, :])).ravel()
    kz = np.linspace(-np.pi, np.pi, 100001)[:-1]
    for mu in (0.0, -1.0):
        contour = equilibrium_contour(-8.0, mu, KT)
        occupations = open_system_density(
            system, contour, contour, empty, mu, mu, KT, 0.0, 0.0, 2.0
        ).occupations
        exact = 2 * np.mean(
            [np.mean(fermi_function(e - 2 * np.cos(kz), mu, KT).real) for e in e_parallel]
        )
        assert np.allclose(occupations[band], exact, atol=1e-6)
        assert np.abs(np.delete(occupations, band)).max() < 1e-6  # the remote band is empty


def test_two_decompositions_keep_the_mirror_symmetry(tmp_path):
    """Two barriers on sites 3 and 4: n_l(+V) = n_(7-l)(-V), bound states in the window included."""
    from PAOFLOW.transport.negf.density_matrix import open_conductor_from, open_system_density

    transport, band = _cubic_conductor(tmp_path, impurity_sites=(3, 4), nk=5)
    system = open_conductor_from(transport.conductor_data, transport.blc_blocks)
    window = bias_window(0.05, -0.05, KT, 0.004)
    left, right = equilibrium_contour(-8.0, 0.05, KT), equilibrium_contour(-8.0, -0.05, KT)
    forward = open_system_density(system, left, right, window, 0.05, -0.05, KT, 0.05, -0.05, 2.0)
    backward = open_system_density(system, right, left, window, -0.05, 0.05, KT, -0.05, 0.05, 2.0)
    assert np.allclose(forward.occupations[band], backward.occupations[band][::-1], atol=1e-12)
    assert np.allclose(forward.transmission, backward.transmission, atol=1e-12)


def test_identical_leads_are_computed_apart_under_a_shift(tmp_path):
    from PAOFLOW.transport.conductor_kpoint import compute_kpoint_self_energies

    transport, _ = _cubic_conductor(tmp_path)
    common = dict(blc_blocks=transport.blc_blocks, ik=1, ie_g=0, egrid=np.array([0.3]), delta=1e-3,
                  shift_C=0.0, shift_corr=0.0, niterx=200, transfer_thr=1e-10, nfailx=5, surface=False)  # fmt: skip
    shared, _, _ = compute_kpoint_self_energies(
        leads_are_identical=True, shift_L=0.2, shift_R=-0.2, **common
    )
    separate, _, _ = compute_kpoint_self_energies(
        leads_are_identical=False, shift_L=0.2, shift_R=-0.2, **common
    )
    assert np.allclose(shared, separate, atol=1e-12)
    unshifted, _, _ = compute_kpoint_self_energies(
        leads_are_identical=True, shift_L=0.0, shift_R=0.0, **common
    )
    assert not np.allclose(shared, unshifted, atol=1e-6)  # the left lead moved with its shift


def test_self_consistent_scan(tmp_path):
    """Symmetric junction: I(-V) = -I(V), U and dq mirror-antisymmetric; zero bias is the DFT Hamiltonian;
    I/V at small bias is the zero-bias conductance; the conductor Hamiltonian is restored."""
    transport, _ = _cubic_conductor(tmp_path, impurity_sites=(3, 4))
    reference = np.array(transport.blc_blocks['blc_00C'].H, copy=True)
    transport.configure_bias_self_consistency(window_spacing=0.004, mixing=0.2, tolerance=1e-5)
    scan = transport.compute_current_self_consistent(bias=[0.0, 0.05, -0.05, 0.005])
    zero, forward, backward, small = scan.points
    assert zero.current == 0.0 and np.allclose(zero.layer_potential, 0.0) and zero.converged
    assert all(point.converged for point in scan.points)
    assert forward.current > 0  # mu_L > mu_R: electrons flow from left to right
    assert forward.current == pytest.approx(-backward.current, rel=1e-5)
    assert np.allclose(forward.layer_potential, backward.layer_potential[::-1], atol=1e-4)
    assert np.allclose(forward.layer_charge, backward.layer_charge[::-1], atol=1e-5)
    # linear response: I = V * int T(E) (-df/dE) dE at V -> 0, from the transmission in the window
    energies, transmission = small.window_energies, small.transmission
    minus_df = -np.gradient(fermi_function(energies, 0.0, KT).real, energies)
    conductance = np.trapezoid(transmission * minus_df, energies)
    assert small.current / 0.005 == pytest.approx(conductance, rel=2e-3)
    assert np.array_equal(transport.blc_blocks['blc_00C'].H, reference)
    for name in ('current', 'potential', 'charge', 'transmission'):
        assert (tmp_path / f'{name}_self_consistent_fb.dat').exists()
    linear = transport.compute_current_self_consistent(bias=[0.05], profile='linear', write=False)
    assert linear.points[0].iterations == 0
    assert np.allclose(
        np.diff(linear.points[0].layer_potential), np.diff(linear.points[0].layer_potential)[0]
    )


def test_compute_current_integrates_the_whole_fermi_window():
    """A constant T must give I = T V whatever the grid spacing (the window used to end at sigma + 3 dE).

    The sign is that of the Landauer formula: positive for mu_L > mu_R, as in the self-consistent scan.
    """
    from PAOFLOW.transport.calculators.current import compute_current_vs_bias

    energies = np.linspace(-0.4, 0.4, 801)
    bias = np.array([0.01, 0.2])
    current = compute_current_vs_bias(energies, np.full_like(energies, 0.4), bias, 0.5, -0.5, KT)
    assert np.allclose(current, 0.4 * bias, rtol=2e-4)
    swapped = compute_current_vs_bias(energies, np.full_like(energies, 0.4), bias, -0.5, 0.5, KT)
    assert np.allclose(swapped, -current, rtol=1e-12)


def test_finite_bias_needs_a_conductor_partition(tmp_path):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.Transport import Transport
    from PAOFLOW.models.models import single_band_model

    model = single_band_model({(0, 0, 1): -1.0}, np.eye(3), 5.0, 1.0, (4, 4, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        transport = Transport(p.data_controller)
        transport.define_partition(central_atoms='ALL', transport_direction='z')
        transport.configure_energy_grid(emin=-1.0, emax=1.0, ne=11, delta=1e-3)
        transport.build_hamiltonian_blocks(
            calculation_type='bulk', use_sym=False, do_overlap_transformation=False
        )
    with pytest.raises(RuntimeError, match='conductor partition'):
        transport.compute_current_self_consistent(bias=[0.1])
    with pytest.raises(ValueError, match='planar'):
        transport.configure_bias_self_consistency(electrostatics='3d')


def test_thomas_fermi_preconditioner_reaches_the_same_solution(tmp_path):
    from PAOFLOW.transport.negf.electrostatics import poisson_kernel

    transport, _ = _cubic_conductor(tmp_path, impurity_sites=(3, 4))
    results = {}
    for preconditioned, mixing in ((True, 0.5), (False, 0.2)):
        transport.configure_bias_self_consistency(
            window_spacing=0.004,
            mixing=mixing,
            tolerance=1e-6,
            thomas_fermi_preconditioner=preconditioned,
        )
        results[preconditioned] = transport.compute_current_self_consistent(
            bias=[0.05], write=False
        ).points[0]
    assert results[True].converged and results[False].converged
    assert results[True].iterations <= results[False].iterations
    assert np.allclose(results[True].layer_potential, results[False].layer_potential, atol=1e-5)
    assert results[True].current == pytest.approx(results[False].current, rel=1e-6)
    layers = PlanarLayers(
        layer_of_orbital=np.arange(3), coordinates=np.array([0.0, 1.0, 2.0]),
        left_boundary=-0.5, right_boundary=2.5, area=4.0,
    )  # fmt: skip
    kernel = poisson_kernel(layers)
    assert np.allclose(kernel, kernel.T) and np.all(np.linalg.eigvalsh(kernel) > 0)
