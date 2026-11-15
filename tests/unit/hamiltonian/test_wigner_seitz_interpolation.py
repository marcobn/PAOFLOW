"""Wigner-Seitz interpolation of HRs: images, Fourier sums, fine-grid FFT and the PAOFLOW entry points."""

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
