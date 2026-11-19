"""Model crystals for electron-phonon coupling (``PAOFLOW.models.electron_phonon``).

The vertex of the model obeys the sum rules of a crystal; the mode-resolved coupling equals the
production double-delta kernel of the ab initio route; the property steps reproduce the textbook
limits of the lecture notes (mass enhancement = lambda, rho ~ T, mobility ~ T^-1/2 in 1D, the two
onsets of the phonon-assisted absorption).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.models.electron_phonon import (
    MODEL_CRYSTALS,
    ElectronPhononModel,
    cubic_metal_model,
    indirect_semiconductor_model,
    light_atom_metal_model,
    properties,
    workflows,
)
from PAOFLOW.models.electron_phonon.model import RY_EV

pytestmark = pytest.mark.unit

MODELS = {
    'cubic_metal': cubic_metal_model,
    'light_atom_metal_2d': lambda: light_atom_metal_model(dim=2),
    'light_atom_metal_3d': lambda: light_atom_metal_model(dim=3),
    'indirect_2': lambda: indirect_semiconductor_model(2),
    'indirect_3': lambda: indirect_semiconductor_model(3),
    'indirect_4': lambda: indirect_semiconductor_model(4),
}


def _random_k(model: ElectronPhononModel, count: int = 5, seed: int = 3) -> np.ndarray:
    k = np.random.default_rng(seed).random((count, 3))
    k[:, [not p for p in model.periodic_axes]] = 0.0
    return k


@pytest.mark.parametrize('name', sorted(MODELS))
def test_phonons_and_vertex_obey_the_sum_rules(name):
    model = MODELS[name]()
    energies, _ = model.phonons(np.zeros((1, 3)))
    assert np.allclose(np.sort(np.abs(energies[0]))[:3], 0.0, atol=1e-7)  # acoustic sum rule
    k = _random_k(model)
    # a rigid translation of the crystal changes nothing: sum over the atoms of dV/du at q = 0
    vertex = model.vertex(k, np.zeros(3)).reshape(
        len(k), model.number_of_atoms, 3, *([model.number_of_orbitals] * 2)
    )
    assert np.abs(vertex.sum(axis=1)).max() < 1e-10
    # Hermiticity: <i k+q|dV_q|j k> = conj(<j k|dV_-q|i k+q>)
    q = np.array([0.25, 0.5, 0.125]) * np.array(model.periodic_axes)
    forward = model.vertex(k, q)
    backward = model.vertex(k + q, -q)
    assert np.allclose(forward, np.conj(np.swapaxes(backward, -1, -2)), atol=1e-12)
    # stable phonons on a grid
    grid = properties.crystal_grid(model, 6)
    assert model.phonons(grid)[0].min() > -1e-6


def test_hopping_gradient_is_that_of_the_power_law():
    model = cubic_metal_model(t=-1.0, t2=-0.15, eta=6.0)
    table = model.hopping_table()
    distance = np.linalg.norm(table.bonds, axis=1)
    analytic = (-6.0 * table.hoppings / distance**2)[:, None] * table.bonds
    assert np.allclose(table.gradients, analytic, atol=1e-7)


@pytest.mark.parametrize('n_bands', [2, 3, 4])
def test_semiconductor_is_the_parabolic_model_in_slater_koster_form(n_bands):
    from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
    from PAOFLOW.models.models import parabolic_model

    model = indirect_semiconductor_model(n_bands, indirect_gap=1.0, t_c=-0.25)
    reference = parabolic_model(
        n_bands=n_bands, gap=1.0 + 4 * (n_bands - 1) * 0.25, t_c=-0.25, asymmetry=0.0
    )
    k = _random_k(model, 9)
    bands = np.linalg.eigvalsh(model.hamiltonian(k))
    for source in (reference, model.to_hopping_model()):
        terms = bloch_terms_from_HRs(
            source.real_space_hamiltonian(),
            source.lattice_vectors,
            source.alat,
            source.orbital_positions,
        )
        assert np.abs(bands - np.linalg.eigvalsh(hamiltonian_block(terms, k))).max() < 1e-12
    states = properties.electron_states(model, {2: 200, 3: 40, 4: 16}[n_bands])
    vbm, cbm = properties.band_edges(states, model.nelec)
    assert abs(vbm) < 1e-12 and abs(cbm - vbm - 1.0) < 1e-12


def test_coupling_equals_the_production_double_delta_kernel():
    """lambda_qnu of the model = lambda_q_dense_ws_fast on the model's HRs and real-space vertex."""
    from PAOFLOW.elphon.eph_kq import THZ_TO_EV
    from PAOFLOW.elphon.elph_bloch import lambda_q_dense_ws_fast, precompute_dense_electrons

    model = cubic_metal_model()
    hopping_model = model.to_hopping_model()
    grid = tuple(hopping_model.real_space_grid)
    electrons = precompute_dense_electrons(
        hopping_model.real_space_hamiltonian(), model.lattice, 12, [0.10 / RY_EV], model.nelec, grid
    )
    q_points = np.array([[0.25, 0, 0], [0.25, 0.25, 0], [1 / 6, 1 / 3, 0], [1 / 3, 1 / 4, 1 / 12]])
    mine = properties.mode_resolved_coupling(model, 12, q_points, sigma=0.10)
    table = model.hopping_table()
    for iq, q in enumerate(q_points):
        vertex_real_space = np.zeros((1, 1, 3) + grid, complex)
        for bond, gradient in zip(table.bonds, table.gradients):
            cell = np.rint(bond / model.alat).astype(int)
            vertex_real_space[0, 0, :, cell[0] % grid[0], cell[1] % grid[1], cell[2] % grid[2]] += (
                gradient / RY_EV * (np.exp(2j * np.pi * q @ cell) - 1.0)
            )
        energy, eigenvectors = model.phonons(q[None])
        zmass = (eigenvectors[0] / np.sqrt(np.repeat(model.masses, 3) * 911.44424)[:, None]).T
        kernel = lambda_q_dense_ws_fast(
            vertex_real_space, electrons, q, zmass, energy[0] / THZ_TO_EV
        )
        # equal to 1e-9 in the coupled mode; the kernel leaves ~1e-7 in the modes that vanish
        scale = mine['lambda_qv'].max()
        assert np.allclose(np.ravel(kernel['lambda_qnu']), mine['lambda_qv'][iq], atol=1e-6 * scale)


def test_symmetry_and_irreducible_q():
    for model in (
        cubic_metal_model(),
        light_atom_metal_model(dim=3),
        indirect_semiconductor_model(4),
    ):
        assert len(properties.symmetry_operations(model)) == 48
        q_points, weights = properties.irreducible_q_points(model, 8)
        assert weights.sum() == 8**3 and len(q_points) == 35
    two_d = light_atom_metal_model(dim=2)
    q_points, weights = properties.irreducible_q_points(two_d, 6)
    assert len(q_points) == 36 and np.all(q_points[:, 2] == 0)


def test_hole_occupations_do_not_underflow():
    occupation, vacancy = properties.fermi_occupations(np.array([-0.5, 0.5]), 0.0, 100.0)
    assert vacancy[0] > 0 and occupation[1] > 0
    assert np.isclose(vacancy[0], occupation[1]) and np.isclose(occupation[0] + vacancy[0], 1.0)


def test_metal_steps_reproduce_the_textbook_limits(tmp_path):
    model = cubic_metal_model()
    out = workflows.superconductivity(model, str(tmp_path), nk=24, nq=4)
    assert 0.55 < out['lambda'] < 0.85 and out['Tc_mcmillan'] < out['Tc_allen_dynes']
    for name in ('alpha2F.dat', 'alpha2F_tr.dat', 'eliashberg.npz', 'lambda_summary.dat'):
        assert (tmp_path / name).is_file(), name
    solution = workflows.migdal_eliashberg(str(tmp_path), temperatures=np.arange(1.0, 16.0, 1.0))
    assert solution['result']['Tc_gap'] > out['Tc_allen_dynes']  # ME above the formulae
    assert (tmp_path / 'migdal_eliashberg' / 'gap_vs_T.dat').is_file()
    self_energy = workflows.self_energy(str(tmp_path), temperatures=(0.0,))
    assert abs(self_energy['mass_enhancement'][0] / out['lambda'] - 1.0) < 0.02
    resistivity = workflows.resistivity(str(tmp_path), temperatures=(300.0, 600.0))
    rho = resistivity['rows'][:, 1]
    assert abs(rho[1] / rho[0] - 2.0) < 0.1  # linear above the Debye scale


def test_semiconductor_steps_in_one_dimension(tmp_path):
    model = indirect_semiconductor_model(2)
    out = workflows.mobility(model, str(tmp_path), nk=600, temperatures=(150.0, 300.0, 450.0))
    slope = np.polyfit(np.log(out['temps_k']), np.log(out['mobility_e_serta']), 1)[0]
    assert -0.65 < slope < -0.35  # acoustic phonons in 1D: mu ~ T^-1/2
    assert np.all(np.isnan(out['mobility_h_serta']))  # electrons only by default
    assert np.all(out['mobility_e_mrta'] > 0)
    table = np.loadtxt(tmp_path / 'mobility.dat')
    assert table.shape == (3, 11)
    both = workflows.mobility(model, str(tmp_path), nk=600, temperatures=(300.0,), carriers='both')
    assert both['mobility_h_serta'][0] > 0
    absorption = workflows.phonon_assisted_absorption(model, str(tmp_path), nk=200)
    photon, gap, phonon = (
        absorption['photon_ev'],
        absorption['gap_ev'],
        absorption['phonon_edge_ev'],
    )
    cold, _room, hot = absorption['eps2_average']  # 10, 300 and 600 K
    below = photon < gap - phonon - 0.05
    above = photon > gap + phonon + 0.05
    assert cold[below].max() < 1e-3 * cold[above].max()  # no phonon to absorb at 10 K
    assert hot[below].max() > cold[below].max()
    for name in ('epsilon2_indabs_10.0K.dat', 'alpha_300.0K.dat', 'rates_300.0K.dat'):
        assert (tmp_path / name).is_file(), name


def test_model_crystals_registry_and_errors(tmp_path):
    assert set(MODEL_CRYSTALS) == {'cubic_metal', 'light_atom_metal', 'indirect_semiconductor'}
    with pytest.raises(ValueError):
        indirect_semiconductor_model(5)
    with pytest.raises(ValueError):
        workflows.mobility(cubic_metal_model(), str(tmp_path), nk=8)
    with pytest.raises(ValueError):
        workflows.superconductivity(cubic_metal_model(), str(tmp_path), nk=10, nq=4)
    with pytest.raises(FileNotFoundError):
        workflows.migdal_eliashberg(os.path.join(str(tmp_path), 'nothing'))
