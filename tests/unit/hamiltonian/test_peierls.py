"""Peierls phases of a uniform magnetic field, and the orbital positions of supercells they rely on.

* square lattice: H(k) equals the Harper matrix; the Hofstadter bands at 1/3 flux carry C = 1, -2, 1;
* a six-site ring: exact Aharonov-Bohm levels, and the electron's sign (l = +1 rises for B along +z);
* graphene ribbon: the n = 1 Landau level at sqrt(2) hbar v_F / l_B;
* Hermiticity, and the flux rule of periodic cells;
* doubling_Hamiltonian: model positions in alat units, Dnm = tau_n - tau_m between the replicas, and no
  spurious band-diagonal velocity (16 Nov 2026).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
from PAOFLOW.hamiltonian.orbital_centres import orbital_centres_bohr
from PAOFLOW.models.models import HoppingModel, honeycomb_model

pytestmark = pytest.mark.unit

T_HOP = -1.0


def _paoflow(model, tmp_path):
    from PAOFLOW import PAOFLOW

    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)


def _hamiltonian(p, k) -> np.ndarray:
    arrays, attributes = p.data_controller.data_dicts()
    terms = bloch_terms_from_HRs(
        np.asarray(arrays['HRs']), np.asarray(arrays['a_vectors']), float(attributes['alat']), None
    )
    return hamiltonian_block(terms, np.atleast_2d(k))


def _harper_model(q: int) -> HoppingModel:
    hoppings = []
    for n in range(q):
        m = (n + 1) % q
        cell = (1, 0, 0) if n == q - 1 else (0, 0, 0)
        hoppings += [(n, m, cell, T_HOP), (m, n, tuple(-np.array(cell)), T_HOP)]
        hoppings += [(n, n, (0, 1, 0), T_HOP), (n, n, (0, -1, 0), T_HOP)]
    positions = np.array([[n / q, 0.0, 0.0] for n in range(q)])
    return HoppingModel(
        'harper', tuple(hoppings), q, np.diag([float(q), 1.0, 10.0]), 1.0, positions, 1.0, (3, 3, 1)
    )


def _harper_matrix(q: int, k1: float, k2: float) -> np.ndarray:
    """Landau gauge A = (0, B x, 0), one flux quantum per q plaquettes."""
    h = np.zeros((q, q), complex)
    for n in range(q):
        h[n, n] = 2 * T_HOP * np.cos(2 * np.pi * k2 + 2 * np.pi * n / q)
        h[n, (n + 1) % q] += T_HOP * (np.exp(2j * np.pi * k1) if n == q - 1 else 1.0)
        h[(n + 1) % q, n] += T_HOP * (np.exp(-2j * np.pi * k1) if n == q - 1 else 1.0)
    return h


def test_square_lattice_is_the_harper_hamiltonian(tmp_path):
    p = _paoflow(_harper_model(3), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        assert p.peierls_field(flux_quanta=1) == 1.0
    for k1, k2 in np.random.default_rng(0).random((5, 2)):
        np.testing.assert_allclose(
            _hamiltonian(p, [k1, k2, 0.0])[0], _harper_matrix(3, k1, k2), atol=1e-14
        )


def test_hofstadter_bands_carry_the_tknn_chern_numbers(tmp_path):
    p = _paoflow(_harper_model(3), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.peierls_field(flux_quanta=1)
    n = 18
    grid = np.arange(n) / n
    vectors = np.stack(
        [
            np.linalg.eigh(_hamiltonian(p, np.c_[np.full(n, k1), grid, np.zeros(n)]))[1]
            for k1 in grid
        ]
    )
    cherns = []
    for band in range(3):
        u = vectors[:, :, :, band]
        u1, u2 = np.roll(u, -1, 0), np.roll(u, -1, 1)
        u12 = np.roll(u1, -1, 1)
        link = lambda a, b: np.einsum('ijn,ijn->ij', a.conj(), b)
        flux = np.angle(link(u, u1) * link(u1, u12) * link(u12, u2) * link(u2, u))
        # the plaquette sum of Fukui, Hatsugai and Suzuki is -(1/2 pi) int Omega for the Berry
        # connection i<u|grad u>: C = -sum / 2 pi in that (Kubo, berry_phase) convention
        cherns.append(-int(np.rint(flux.sum() / (2 * np.pi))))
    assert cherns == [1, -2, 1]  # field along +z (a1 x a2), flux 1/3 per plaquette


def _ring(radius: float = 1.5) -> HoppingModel:
    hoppings = tuple(
        h
        for j in range(6)
        for h in ((j, (j + 1) % 6, (0, 0, 0), T_HOP), ((j + 1) % 6, j, (0, 0, 0), T_HOP))
    )
    angles = 2 * np.pi * np.arange(6) / 6
    positions = np.c_[
        0.5 + radius * np.cos(angles) / 20, 0.5 + radius * np.sin(angles) / 20, np.full(6, 0.5)
    ]
    return HoppingModel(
        'ring', hoppings, 6, np.diag([20.0, 20.0, 20.0]), 1.0, positions, 1.0, (1, 1, 1)
    )


def test_ring_aharonov_bohm_levels_and_the_electron_sign(tmp_path):
    flux_per_cell, radius = 2.0, 1.5
    p = _paoflow(_ring(radius), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.peierls_field(flux_quanta=flux_per_cell)
    energies, vectors = np.linalg.eigh(
        np.asarray(p.data_controller.data_arrays['HRs'])[:, :, 0, 0, 0, 0]
    )
    ring_flux = flux_per_cell * (1.5 * np.sqrt(3) * radius**2) / 400.0  # hexagon area / cell area
    exact = np.sort([2 * T_HOP * np.cos(2 * np.pi * (l + ring_flux) / 6) for l in range(-2, 4)])
    np.testing.assert_allclose(energies, exact, atol=1e-12)
    # field along +z (a1 x a2): the counter-clockwise state l = +1 rises, its moment is antiparallel to B
    plane_wave = np.exp(2j * np.pi * np.arange(6) / 6) / np.sqrt(6)
    weight = np.abs(plane_wave.conj() @ vectors) ** 2
    assert energies[np.argmax(weight)] == pytest.approx(
        2 * T_HOP * np.cos(2 * np.pi * (1 + ring_flux) / 6)
    )
    assert energies[np.argmax(weight)] > 2 * T_HOP * np.cos(2 * np.pi / 6)


def test_graphene_ribbon_landau_level(tmp_path):
    p = _paoflow(honeycomb_model(delta=0.0, t=T_HOP), tmp_path)
    flux_per_cell = 1.0 / 150.0
    with contextlib.redirect_stdout(io.StringIO()):
        p.doubling_Hamiltonian(5, 0, 0)  # 32 cells along a1
        p.cutting_Hamiltonian(x=True)  # open along a1: a ribbon periodic along a2
        p.peierls_field(flux_quanta=32 * flux_per_cell)
    energies = np.concatenate(
        [np.linalg.eigvalsh(_hamiltonian(p, [0.0, k2, 0.0])[0]) for k2 in np.arange(120) / 120]
    )
    first_level = abs(T_HOP) * np.sqrt(
        2 * np.sqrt(3) * np.pi * flux_per_cell
    )  # sqrt(2) hbar v_F / l_B
    window = energies[(energies > 0.7 * first_level) & (energies < 1.3 * first_level)]
    counts, edges = np.histogram(window, bins=60)
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
    assert peak == pytest.approx(first_level, rel=0.03)


def test_hermitian_and_the_flux_rule(tmp_path):
    p = _paoflow(honeycomb_model(delta=0.2, t=T_HOP, t2=0.1, phi=0.3), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.doubling_Hamiltonian(1, 0, 0)
        with pytest.raises(ValueError):
            p.peierls_field(flux_quanta=0.5)  # periodic along a1: needs an integer number of quanta
        p.peierls_field(flux_quanta=1)
    hamiltonian = _hamiltonian(p, [0.13, 0.41, 0.0])[0]
    np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-14)


def test_doubled_model_positions_and_velocities(tmp_path):
    p = _paoflow(honeycomb_model(delta=0.4, t=T_HOP), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.doubling_Hamiltonian(1, 0, 0)
    arrays, attributes = p.data_controller.data_dicts()
    np.testing.assert_allclose(
        np.asarray(arrays['tau'])[2:],
        np.asarray(arrays['tau'])[:2] + 0.5 * np.asarray(arrays['a_vectors'])[0],
        atol=1e-12,
    )
    centres = orbital_centres_bohr(arrays, attributes)
    np.testing.assert_allclose(arrays['Dnm'], centres[:, None] - centres[None, :], atol=1e-12)
    # the D_nm term i(tau_n - tau_m)H_nm has no band-diagonal part (it was 2.3 eV bohr before the fix)
    for k in np.random.default_rng(1).random((5, 3)) * [1, 1, 0]:
        h = _hamiltonian(p, k)[0]
        vectors = np.linalg.eigh(h)[1]
        for axis in range(2):
            term = 1j * np.asarray(arrays['Dnm'])[:, :, axis] * h
            assert np.abs(np.diag(vectors.conj().T @ term @ vectors)).max() < 1e-12
