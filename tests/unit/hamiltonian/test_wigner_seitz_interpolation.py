"""Wigner-Seitz interpolation of HRs (cell-based and pair-resolved): images, sums, fine grid, entry points."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from PAOFLOW.elphon.elph_bloch import _ws_lattice
from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
from PAOFLOW.hamiltonian.wigner_seitz import (
    check_fine_grid_holds_images,
    fine_grid_placement,
    wigner_seitz_fine_grid_array,
    wigner_seitz_hamiltonian_k,
    wigner_seitz_images,
    wigner_seitz_pair_hamiltonian_k,
    wigner_seitz_pair_images,
)

pytestmark = pytest.mark.unit

LATTICES = {
    'fcc': np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]]) / 2,
    'bcc': np.array([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -1.0, 1.0]]) / 2,
    'sc': np.eye(3),
    'hex': np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.6]]),
    'triclinic': np.array([[1.0, 0.1, 0.0], [0.4, 1.1, 0.0], [0.3, 0.2, 0.9]]),
}


def random_hrs(
    grid: tuple[int, int, int], nawf: int = 3, nspin: int = 1, seed: int = 0
) -> np.ndarray:
    """HRs = ifftn of a random Hermitian H(k) on the grid (PAOFLOW's convention)."""
    rng = np.random.default_rng(seed)
    hk = rng.normal(size=(nawf, nawf) + grid + (nspin,)) + 1j * rng.normal(
        size=(nawf, nawf) + grid + (nspin,)
    )
    hk = 0.5 * (hk + np.conj(np.swapaxes(hk, 0, 1)))
    return np.fft.ifftn(hk, axes=(2, 3, 4)), hk


@pytest.mark.parametrize('name', sorted(LATTICES))
@pytest.mark.parametrize('grid', [(4, 4, 4), (5, 6, 3), (8, 8, 8)])
def test_images_match_the_elphon_construction(name, grid):
    images = wigner_seitz_images(grid, LATTICES[name])
    cells, weights, _ = _ws_lattice(np.array(grid), LATTICES[name])
    ours = {tuple(c): w for c, w in zip(images.cells, images.weights)}
    theirs = {tuple(c): w for c, w in zip(cells, weights)}
    assert ours.keys() == theirs.keys()
    assert all(abs(ours[c] - theirs[c]) < 1e-12 for c in ours)
    assert abs(images.weights.sum() - np.prod(grid)) < 1e-9
    assert np.array_equal(np.mod(images.cells, grid), images.grid_index)
    assert {tuple(-c) for c in images.cells} == set(ours)  # inversion-symmetric set


@pytest.mark.parametrize('name', ['fcc', 'hex', 'triclinic'])
def test_fourier_sum_matches_bloch_terms_and_is_exact_on_the_mesh(name):
    grid = (4, 4, 3)
    hrs, hk = random_hrs(grid, nawf=3, nspin=2)
    images = wigner_seitz_images(grid, LATTICES[name])
    rng = np.random.default_rng(3)
    k_random = rng.uniform(-0.5, 0.5, (7, 3))
    ours = wigner_seitz_hamiltonian_k(hrs, images, k_random)
    for ispin in range(2):
        terms = bloch_terms_from_HRs(hrs, LATTICES[name], 1.0, None, ispin=ispin)
        assert np.allclose(
            np.moveaxis(ours[..., ispin], 2, 0), hamiltonian_block(terms, k_random), atol=1e-12
        )
    mesh = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(
        -1, 3
    )
    on_mesh = wigner_seitz_hamiltonian_k(hrs, images, mesh)
    assert np.allclose(on_mesh, hk.reshape(3, 3, -1, 2), atol=1e-12)


@pytest.mark.parametrize('fine', [(12, 12, 9), (6, 6, 5), (4, 4, 3)])
def test_fine_grid_fft_is_the_wigner_seitz_sum(fine):
    grid = (4, 4, 3)
    hrs, _ = random_hrs(grid, nawf=2)
    images = wigner_seitz_images(grid, LATTICES['fcc'])
    placement = fine_grid_placement(images, fine)
    if fine == (4, 4, 3):
        assert not placement.unique  # colliding images are added: still the exact sum on this grid
    fine_k = np.stack(np.meshgrid(*[np.arange(m) / m for m in fine], indexing='ij'), -1).reshape(
        -1, 3
    )
    direct = wigner_seitz_hamiltonian_k(hrs, images, fine_k)
    for mu in range(2):
        for nu in range(2):
            grid_values = np.fft.fftn(
                wigner_seitz_fine_grid_array(hrs[mu, nu, :, :, :, 0], images, placement)
            )
            assert np.allclose(grid_values.reshape(-1), direct[mu, nu, :, 0], atol=1e-12)


def test_fine_grid_check():
    images = wigner_seitz_images((8, 8, 8), LATTICES['fcc'])
    check_fine_grid_holds_images(images, (16, 16, 16))
    check_fine_grid_holds_images(images, (13, 13, 13))
    with pytest.raises(ValueError, match='nfft >= \\(13, 13, 13\\)'):
        check_fine_grid_holds_images(images, (12, 12, 12))


def _fcc_band_paoflow(tmp_path):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import single_band_model

    hoppings = {(1, 0, 0): -1.0, (0, 1, 0): -1.0, (0, 0, 1): -1.0, (1, -1, 0): -1.0, (1, 0, -1): -1.0,
                (0, 1, -1): -1.0, (1, 1, -1): -0.3, (2, -1, 0): 0.2, (1, 1, -2): 0.15, (2, -1, -1): 0.1}  # fmt: skip
    model = single_band_model(hoppings, LATTICES['fcc'], 7.6, 1.0, (5, 5, 5))
    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)


def test_interpolated_hamiltonian_wigner_seitz_and_its_velocities(tmp_path):
    p = _fcc_band_paoflow(tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    hrs = np.array(arrays['HRs'])
    a_vectors = np.asarray(arrays['a_vectors'], dtype=float)
    images = wigner_seitz_images(hrs.shape[2:5], a_vectors)
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=10, nfft2=10, nfft3=10, interpolation='wigner_seitz')
        p.pao_eigh()
        p.gradient_and_momenta(position_dipoles=False)
    k_cart = np.asarray(arrays['kgrid']).T
    k_cryst = k_cart @ a_vectors.T
    energies = np.asarray(arrays['E_k'])[:, :, 0]
    direct = np.linalg.eigvalsh(
        np.moveaxis(wigner_seitz_hamiltonian_k(hrs, images, k_cryst)[..., 0], 2, 0)
    )
    assert np.allclose(energies, direct, atol=1e-10)
    # velocities: pksp holds -dH/dk (eV bohr), the exact derivative of the Wigner-Seitz bands
    pksp = np.asarray(arrays['pksp'])
    step = 1e-5
    for ik in (3, 117, 555):
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            pair = np.array([k_cart[ik] + shift, k_cart[ik] - shift]) @ a_vectors.T
            e_pm = np.linalg.eigvalsh(
                np.moveaxis(wigner_seitz_hamiltonian_k(hrs, images, pair)[..., 0], 2, 0)
            )
            derivative = (e_pm[0, 0] - e_pm[1, 0]) / (2 * step) * attr['alat'] / (2 * np.pi)
            assert abs(np.real(pksp[ik, axis, 0, 0, 0]) + derivative) < 1e-6


def test_interpolated_hamiltonian_rejects_a_grid_too_small(tmp_path):
    p = _fcc_band_paoflow(tmp_path)
    p.data_controller.data_attributes['abort_on_exception'] = True
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError, match='Wigner-Seitz'):
        p.interpolated_hamiltonian(nfft1=6, nfft2=6, nfft3=6, interpolation='wigner_seitz')


def test_interpolated_hamiltonian_zero_padding_unchanged(tmp_path):
    from PAOFLOW.utils.zero_pad import zero_pad

    p = _fcc_band_paoflow(tmp_path)
    arrays, _ = p.data_controller.data_dicts()
    hrs = np.array(arrays['HRs'])
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=10, nfft2=10, nfft3=10)
    expected = np.fft.fftn(zero_pad(hrs[0, 0, :, :, :, 0], 5, 5, 5, 5, 5, 5))
    assert np.allclose(
        np.asarray(arrays['Hksp'])[:, 0, 0, 0].reshape(10, 10, 10), expected, atol=1e-12
    )


def test_bands_wigner_seitz_route(tmp_path):
    p = _fcc_band_paoflow(tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    hrs = np.array(arrays['HRs'])
    a_vectors = np.asarray(arrays['a_vectors'], dtype=float)
    path_cryst = np.linspace(0.0, 1.0, 37)[:, None] * np.array([0.5, 0.25, 0.75])
    arrays['kq'] = path_cryst.T.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        p.bands(fname='bands_ws', interpolation='wigner_seitz')
    images = wigner_seitz_images(hrs.shape[2:5], a_vectors)
    direct = np.linalg.eigvalsh(
        np.moveaxis(wigner_seitz_hamiltonian_k(hrs, images, path_cryst)[..., 0], 2, 0)
    )
    assert np.allclose(np.asarray(arrays['E_k'])[:, :, 0], direct, atol=1e-10)


def test_unknown_interpolation_is_rejected(tmp_path):
    from PAOFLOW.hamiltonian.do_double_grid import do_double_grid
    from PAOFLOW.spectrum.do_bands import bands_calc

    p = _fcc_band_paoflow(tmp_path)
    with pytest.raises(ValueError, match='interpolation'):
        do_double_grid(p.data_controller, interpolation='spline')
    p.data_controller.data_arrays['kq'] = np.zeros((3, 2))
    with pytest.raises(ValueError, match='interpolation'):
        bands_calc(p.data_controller, interpolation='spline')


# ----------------------------------------------------------------------------------------- pair-resolved images
DIAMOND_SITES = np.array(
    [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
)  # Cartesian, alat (fcc a_vectors above)
OPERATIONS = {
    'C3 [111]': np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
    'C4 z': np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]),  # swaps the two atoms of diamond
    'mirror x': np.diag([-1, 1, 1]),  # swaps them too
}


def diamond_s_model(grid=(4, 4, 4), decay=0.5, cutoff=4.0):
    """Aliased HRs of one s orbital per diamond site, hopping -exp(-d / decay) between any two sites.

    H_ij(k) = sum_R h(|R + tau_j - tau_i|) exp(2 pi i k.R) is summed exactly on the mesh and transformed back, so
    HRs is aliased like that of a DFT mesh and exactly symmetric under the space group of diamond (spherical cutoff).
    """
    lattice = LATTICES['fcc']
    positions = np.linalg.solve(lattice.T, DIAMOND_SITES.T).T
    bound = 8
    cells = np.stack(np.meshgrid(*[np.arange(-bound, bound + 1)] * 3, indexing='ij'), -1).reshape(
        -1, 3
    )
    mesh = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(
        -1, 3
    )
    phases = np.exp(2j * np.pi * mesh @ cells.T)
    hk = np.zeros((2, 2) + tuple(grid) + (1,), dtype=complex)
    for i in range(2):
        for j in range(2):
            distance = np.linalg.norm((cells + positions[j] - positions[i]) @ lattice, axis=1)
            amplitude = np.where(
                (distance > 1e-9) & (distance < cutoff), -np.exp(-distance / decay), 0.0
            )
            hk[i, j, ..., 0] = (phases @ amplitude).reshape(grid)
    return np.fft.ifftn(hk, axes=(2, 3, 4)), positions


def _symmetry_violation(hamiltonian_k_of, k_cart, lattice):
    """Largest |E(R k) - E(k)| over the operations, eV."""
    energies = np.linalg.eigvalsh(np.moveaxis(hamiltonian_k_of(k_cart @ lattice.T)[..., 0], 2, 0))
    worst = {}
    for name, rotation in OPERATIONS.items():
        rotated = np.linalg.eigvalsh(
            np.moveaxis(hamiltonian_k_of((k_cart @ rotation.T) @ lattice.T)[..., 0], 2, 0)
        )
        worst[name] = float(np.abs(rotated - energies).max())
    return worst


@pytest.mark.parametrize('name', ['fcc', 'hex', 'triclinic'])
def test_pair_images_of_a_single_site_are_the_cell_images(name):
    grid = (4, 4, 3)
    hrs, _ = random_hrs(grid, nawf=3, nspin=2)
    pairs = wigner_seitz_pair_images(grid, LATTICES[name], np.zeros((3, 3)))
    cells = wigner_seitz_images(grid, LATTICES[name])
    ours = {
        tuple(c): w
        for c, w in zip(pairs.images_of_pair(0, 0).cells, pairs.images_of_pair(0, 0).weights)
    }
    assert ours == {tuple(c): w for c, w in zip(cells.cells, cells.weights)}
    k_random = np.random.default_rng(5).uniform(-0.5, 0.5, (9, 3))
    assert np.allclose(
        wigner_seitz_pair_hamiltonian_k(hrs, pairs, k_random),
        wigner_seitz_hamiltonian_k(hrs, cells, k_random),
        atol=1e-12,
    )


def test_pair_images_of_diamond():
    grid = (4, 4, 4)
    hrs, positions = diamond_s_model(grid)
    pairs = wigner_seitz_pair_images(grid, LATTICES['fcc'], positions)
    for site_i in range(2):
        for site_j in range(2):
            assert abs(pairs.weights[:, site_i, site_j].sum() - np.prod(grid)) < 1e-9
    forward = {tuple(c) for c in pairs.images_of_pair(0, 1).cells}
    assert {tuple(-c) for c in pairs.images_of_pair(1, 0).cells} == forward  # H_ji(n) = H_ij(-n)^*
    assert forward != {tuple(c) for c in wigner_seitz_images(grid, LATTICES['fcc']).cells}
    mesh = np.stack(np.meshgrid(*[np.arange(n) / n for n in grid], indexing='ij'), -1).reshape(
        -1, 3
    )
    exact = np.fft.fftn(hrs, axes=(2, 3, 4)).reshape(2, 2, -1, 1)
    assert np.allclose(wigner_seitz_pair_hamiltonian_k(hrs, pairs, mesh), exact, atol=1e-12)


def test_pair_sum_keeps_the_space_group_of_diamond():
    """The cell-based images break the operations that swap the two atoms; the pair-resolved ones keep them."""
    grid = (4, 4, 4)
    hrs, positions = diamond_s_model(grid)
    lattice = LATTICES['fcc']
    pairs = wigner_seitz_pair_images(grid, lattice, positions)
    cells = wigner_seitz_images(grid, lattice)
    k_cart = np.random.default_rng(11).uniform(-1, 1, (60, 3))
    pair_violation = _symmetry_violation(
        lambda k: wigner_seitz_pair_hamiltonian_k(hrs, pairs, k), k_cart, lattice
    )
    cell_violation = _symmetry_violation(
        lambda k: wigner_seitz_hamiltonian_k(hrs, cells, k), k_cart, lattice
    )
    assert max(pair_violation.values()) < 1e-10
    assert cell_violation['C3 [111]'] < 1e-10  # fixes both atoms
    assert cell_violation['C4 z'] > 1e-4 and cell_violation['mirror x'] > 1e-4


@pytest.mark.parametrize('fine', [(8, 8, 8), (7, 7, 7)])
def test_pair_fine_grid_fft_is_the_pair_sum(fine):
    grid = (4, 4, 4)
    hrs, positions = diamond_s_model(grid)
    pairs = wigner_seitz_pair_images(grid, LATTICES['fcc'], positions)
    check_fine_grid_holds_images(pairs.all_cells(), fine)
    with pytest.raises(ValueError, match='nfft >= \\(7, 7, 7\\)'):
        check_fine_grid_holds_images(pairs.all_cells(), (6, 6, 6))  # the bond images reach |n| = 3
    fine_k = np.stack(np.meshgrid(*[np.arange(m) / m for m in fine], indexing='ij'), -1).reshape(
        -1, 3
    )
    direct = wigner_seitz_pair_hamiltonian_k(hrs, pairs, fine_k)
    site = (
        pairs.site_of_orbital
    )  # np.unique sorts the centres: the site order is not the orbital order
    for orbital_i in range(2):
        for orbital_j in range(2):
            images = pairs.images_of_pair(site[orbital_i], site[orbital_j])
            grid_values = np.fft.fftn(
                wigner_seitz_fine_grid_array(
                    hrs[orbital_i, orbital_j, :, :, :, 0], images, fine_grid_placement(images, fine)
                )
            )
            assert np.allclose(
                grid_values.reshape(-1), direct[orbital_i, orbital_j, :, 0], atol=1e-12
            )


def _diamond_paoflow(tmp_path):
    """PAOFLOW on a two-site HoppingModel (orbital centres in the data) carrying the aliased diamond HRs."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import HoppingModel

    lattice = LATTICES['fcc']
    hrs, positions = diamond_s_model((4, 4, 4))
    cells = np.stack(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing='ij'), -1).reshape(-1, 3)
    hoppings = []
    for i in range(2):
        for j in range(2):
            for cell in cells:
                if np.linalg.norm((cell + positions[j] - positions[i]) @ lattice) < 0.45:
                    hoppings.append((i, j, tuple(int(c) for c in cell), -1.0))
    model = HoppingModel(
        name='diamond s', hoppings=tuple(hoppings), number_of_orbitals=2, lattice_vectors=lattice, alat=10.0,
        orbital_positions=positions, nelec=2.0, real_space_grid=(4, 4, 4),
    )  # fmt: skip
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
    p.data_controller.data_arrays['HRs'] = hrs  # the aliased Hamiltonian of the long-range model
    return p, hrs, positions


def test_interpolated_hamiltonian_pairs_and_its_velocities(tmp_path):
    p, hrs, positions = _diamond_paoflow(tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    lattice = np.asarray(arrays['a_vectors'], dtype=float)
    pairs = wigner_seitz_pair_images(hrs.shape[2:5], lattice, positions)
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=8, nfft2=8, nfft3=8, interpolation='wigner_seitz_pairs')
        p.pao_eigh()
        p.gradient_and_momenta(position_dipoles=False)
    k_cart = np.asarray(arrays['kgrid']).T
    energies = np.asarray(arrays['E_k'])[:, :, 0]
    direct = np.linalg.eigvalsh(
        np.moveaxis(wigner_seitz_pair_hamiltonian_k(hrs, pairs, k_cart @ lattice.T)[..., 0], 2, 0)
    )
    assert np.allclose(energies, direct, atol=1e-10)
    pksp = np.asarray(
        arrays['pksp']
    )  # -dH/dk, eV bohr: the exact derivative of the pair-resolved bands
    step = 1e-5
    for ik in (5, 77, 300):
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            k_pair = np.array([k_cart[ik] + shift, k_cart[ik] - shift]) @ lattice.T
            e_pm = np.linalg.eigvalsh(
                np.moveaxis(wigner_seitz_pair_hamiltonian_k(hrs, pairs, k_pair)[..., 0], 2, 0)
            )
            derivative = (e_pm[0, 0] - e_pm[1, 0]) / (2 * step) * attr['alat'] / (2 * np.pi)
            assert abs(np.real(pksp[ik, axis, 0, 0, 0]) + derivative) < 1e-6


def test_bands_pairs_route(tmp_path):
    p, hrs, positions = _diamond_paoflow(tmp_path)
    arrays, _ = p.data_controller.data_dicts()
    lattice = np.asarray(arrays['a_vectors'], dtype=float)
    path_cryst = np.linspace(0.0, 1.0, 29)[:, None] * np.array([0.5, 0.25, 0.75])
    arrays['kq'] = path_cryst.T.copy()
    with contextlib.redirect_stdout(io.StringIO()):
        p.bands(fname='bands_pairs', interpolation='wigner_seitz_pairs')
    pairs = wigner_seitz_pair_images(hrs.shape[2:5], lattice, positions)
    direct = np.linalg.eigvalsh(
        np.moveaxis(wigner_seitz_pair_hamiltonian_k(hrs, pairs, path_cryst)[..., 0], 2, 0)
    )
    assert np.allclose(np.asarray(arrays['E_k'])[:, :, 0], direct, atol=1e-10)


def test_pairs_route_of_one_site_is_the_cell_route(tmp_path):
    p = _fcc_band_paoflow(tmp_path)
    arrays, _ = p.data_controller.data_dicts()
    hrs = np.array(arrays['HRs'])
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=10, nfft2=10, nfft3=10, interpolation='wigner_seitz_pairs')
    pairs_grid = np.array(arrays['Hksp'])
    arrays['HRs'] = hrs
    for key, value in (('nk1', 5), ('nk2', 5), ('nk3', 5)):
        p.data_controller.data_attributes[key] = value
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=10, nfft2=10, nfft3=10, interpolation='wigner_seitz')
    assert np.allclose(pairs_grid, np.asarray(arrays['Hksp']), atol=1e-12)


def test_pairs_route_needs_one_centre_per_orbital(tmp_path):
    from PAOFLOW.hamiltonian.do_double_grid import do_double_grid
    from PAOFLOW.spectrum.do_bands import bands_calc

    p, _, _ = _diamond_paoflow(tmp_path)
    arrays, attr = p.data_controller.data_dicts()
    attr['nfft1'] = attr['nfft2'] = attr['nfft3'] = 8
    arrays['naw'] = np.array([1])  # centres no longer match nawf = 2
    with pytest.raises(ValueError, match='one centre per orbital'):
        do_double_grid(p.data_controller, interpolation='wigner_seitz_pairs')
    arrays['kq'] = np.zeros((3, 2))
    with pytest.raises(ValueError, match='one centre per orbital'):
        bands_calc(p.data_controller, interpolation='wigner_seitz_pairs')
