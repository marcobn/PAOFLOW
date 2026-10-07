"""Unit tests for the anisotropic Migdal-Eliashberg solver and its Fermi-surface coupling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from PAOFLOW.elphon import anisotropic_eliashberg as mea
from PAOFLOW.elphon import migdal_eliashberg as me
from PAOFLOW.elphon.do_pao_eph_dense_q import irreducible_mesh, irreducible_qmesh
from PAOFLOW.elphon.elph_bloch import (
    RY_TO_EV,
    RY_TO_THZ,
    lambda_q_dense_ws_fast,
    precompute_dense_electrons,
)
from PAOFLOW.elphon.eph_kq import THZ_TO_EV
from PAOFLOW.elphon.fermi_surface_coupling import (
    FermiSurfacePairCoupling,
    read_fs_coupling,
    write_fs_coupling,
)


def _einstein(lam: float, w_e: float = 0.01, width: float = 1.0e-4) -> tuple[np.ndarray, ...]:
    """Narrow-Gaussian Einstein a2F with total coupling ``lam`` at ``w_e`` (eV)."""
    return me.a2f_from_modes([[lam]], [[w_e / THZ_TO_EV]], None, width, 500)


def _coupling_from_a2f(
    omega: np.ndarray, a2F: np.ndarray, pair_shares: np.ndarray, weights: np.ndarray
) -> dict[str, Any]:
    """Fermi-surface coupling ``Lambda_ab(w) = share_ab * 2 a2F(w) dw / w``."""
    nu, a2F_uniform = me._uniform_a2f(omega, a2F)
    lam_density = 2.0 * a2F_uniform * nu[0] / nu
    return {
        'coupling': np.asarray(pair_shares)[:, :, None] * lam_density[None, None, :],
        'freq_ev': nu,
        'weight': np.asarray(weights, dtype=float),
    }


def test_single_state_limit_is_the_isotropic_solution() -> None:
    omega, a2F = _einstein(1.0, w_e=0.008, width=2.0e-4)
    # Two identical states, each coupled half to itself and half to the other.
    coupling = _coupling_from_a2f(omega, a2F, np.full((2, 2), 0.5), [0.4, 0.4])
    assert mea.coupling_strength(coupling)['lambda'] == pytest.approx(1.0, rel=1.0e-3)
    for T in (1.0, 5.0):
        iso = me.solve_imag_iso(omega, a2F, T, 0.1, 0.1)
        aniso = mea.solve_imag_aniso(coupling, T, 0.1, 0.1)
        np.testing.assert_allclose(aniso['delta'], np.tile(iso['delta'], (2, 1)), atol=1.0e-10)
        np.testing.assert_allclose(aniso['Z'], np.tile(iso['Z'], (2, 1)), atol=1.0e-9)
    T = 6.0
    np.testing.assert_allclose(
        mea.linearized_max_eigenvalue_aniso(coupling, T, 0.1, 0.1),
        me.linearized_max_eigenvalue(omega, a2F, T, 0.1, 0.1),
        rtol=1.0e-7,
    )


def test_two_band_model_has_two_gaps_and_one_tc() -> None:
    omega, a2F = _einstein(1.0, w_e=0.01, width=2.0e-4)
    # Strong intraband coupling on state 0 (sigma-like), weak on state 1 (pi-like),
    # with weak interband coupling.
    shares = np.array([[1.2, 0.2], [0.2, 0.4]])
    coupling = _coupling_from_a2f(omega, a2F, shares, [0.3, 0.7])
    temps = np.array([1.0, 6.0, 30.0])
    res = mea.migdal_eliashberg_aniso(coupling, temps, mu_star=0.1, wscut=0.1, n_real=800)
    assert res['converged'].all()
    gap_sigma, gap_pi = res['gap0'][0]
    assert gap_sigma > 1.5 * gap_pi > 0.0
    assert np.all(res['gap0'][-1] == 0.0)  # normal state well above Tc
    # Pade gap edges close to Delta(i w_0) at low T, and a gapped quasiparticle DOS.
    np.testing.assert_allclose(res['gap_edge'][0], res['gap0'][0], rtol=0.1)
    w = res['w_real']
    assert res['qdos'][0][w < 0.8 * gap_pi].max() < 0.05
    # Swapping the two states permutes the solution.
    swapped = _coupling_from_a2f(omega, a2F, shares[::-1, ::-1], [0.7, 0.3])
    np.testing.assert_allclose(
        mea.solve_imag_aniso(swapped, 1.0, 0.1, 0.1)['delta'][::-1],
        mea.solve_imag_aniso(coupling, 1.0, 0.1, 0.1)['delta'],
        atol=1.0e-9,
    )


def test_pair_restricted_coulomb_decouples_sublattices() -> None:
    # Two sectors coupled only within themselves by phonons (k+q on a sublattice):
    # with mu* restricted to the same pairs each sector is the isotropic problem.
    omega, a2F = _einstein(1.0, w_e=0.008, width=2.0e-4)
    weights = np.array([0.4, 0.4])
    coupling = _coupling_from_a2f(omega, a2F, np.eye(2), weights)
    coupling['coulomb'] = np.diag(weights.sum() * np.ones(2))
    iso = me.solve_imag_iso(omega, a2F, 1.0, 0.1, 0.1)
    aniso = mea.solve_imag_aniso(coupling, 1.0, 0.1, 0.1)
    np.testing.assert_allclose(aniso['delta'], np.tile(iso['delta'], (2, 1)), atol=1.0e-10)
    np.testing.assert_allclose(mea.coulomb_weights(coupling).sum(axis=1), 1.0)


def test_batched_pade_matches_single_functions() -> None:
    rng = np.random.default_rng(3)
    z = 1j * np.linspace(0.01, 0.2, 12)
    values = rng.standard_normal((3, 12)) + 1.0
    w = np.linspace(0.0, 0.1, 7)
    batched = me.pade_eval(me.pade_coefficients(z, values), z, w)
    for row, out in zip(values, batched):
        np.testing.assert_allclose(out, me.pade_eval(me.pade_coefficients(z, row), z, w))


def test_irreducible_mesh_matches_qmesh() -> None:
    at = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 1.14]])
    bg = np.linalg.inv(at).T
    rot = np.array([[0, -1, 0], [1, -1, 0], [0, 0, 1]])  # C3 about z, crystal axes
    rots = [np.linalg.matrix_power(rot, n) for n in range(3)]
    mesh, reps, weights, star = irreducible_mesh(6, rots, at, bg)
    q_reps, q_weights = irreducible_qmesh(6, rots, at, bg)
    np.testing.assert_array_equal(mesh[reps], q_reps)
    np.testing.assert_array_equal(weights, q_weights)
    assert weights.sum() == 216
    np.testing.assert_array_equal(np.bincount(star), weights)
    np.testing.assert_array_equal(star[reps], np.arange(reps.size))


def test_pair_coupling_reproduces_isotropic_lambda(tmp_path: Path) -> None:
    rng = np.random.default_rng(21)
    nawf, ncart, Nk = 3, 3, 4
    HRs = rng.standard_normal((nawf, nawf, 2, 2, 2, 1)) + 1j * rng.standard_normal(
        (nawf, nawf, 2, 2, 2, 1)
    )
    gR = rng.standard_normal((nawf, nawf, ncart, 2, 2, 2)) + 1j * rng.standard_normal(
        (nawf, nawf, ncart, 2, 2, 2)
    )
    zmass = rng.standard_normal((3, ncart))
    freqs = np.array([2.0, 3.0, 4.0])
    sigma_ry = 0.3
    electrons = precompute_dense_electrons(HRs, np.eye(3), Nk, [sigma_ry], 3, (2, 2, 2))
    nkd = Nk**3
    # No symmetry: every k is its own star.
    accumulator = FermiSurfacePairCoupling(
        electrons, np.arange(nkd), np.arange(nkd), np.ones(nkd),
        fsthick_ev=8 * sigma_ry * RY_TO_EV, omega_max_ev=4.0 / RY_TO_THZ * RY_TO_EV, n_freq=7,
    )  # fmt: skip
    grid = np.stack(np.meshgrid(*[np.arange(Nk) / Nk] * 3, indexing='ij'), -1).reshape(-1, 3)
    lambda_iso = 0.0
    for q in grid:
        res = lambda_q_dense_ws_fast(
            gR, electrons, q, zmass, freqs, pair_coupling=accumulator,
            pair_weights=np.full(3, 1.0 / nkd), pair_q_weight=1.0 / nkd,
        )  # fmt: skip
        lambda_iso += res['lambda_qnu'][0].sum() / nkd
    coupling = accumulator.result(np.eye(3), np.eye(3), Nk)
    strength = mea.coupling_strength(coupling)
    assert strength['lambda'] == pytest.approx(lambda_iso, rel=1.0e-10)
    assert strength['dos_ef'] == pytest.approx(electrons['dos'][0] / RY_TO_EV, rel=1.0e-10)
    # Coulomb pairs: with q-grid = k-grid every row visits the whole Fermi surface.
    coulomb = coupling['coulomb']
    dos_ef = strength['dos_ef']
    assert coupling['weight'] @ coulomb.sum(axis=1) == pytest.approx(dos_ef**2, rel=1.0e-10)

    path = str(tmp_path / 'fs.npz')
    write_fs_coupling(path, coupling)
    loaded = read_fs_coupling(path)
    assert loaded['nk_dense'] == Nk and isinstance(loaded['fermi_ev'], float)
    np.testing.assert_array_equal(loaded['coupling'], coupling['coupling'])


def test_grid_split_conserves_weight() -> None:
    class _Grid:
        freq_step_ev = 0.01
        n_freq = 5

    lower, upper = FermiSurfacePairCoupling._grid_split(_Grid(), np.array([0.005, 0.025, 0.08]))
    np.testing.assert_array_equal(lower, [0, 1, 3])
    np.testing.assert_allclose(upper, [0.0, 0.5, 1.0])


def test_write_me_aniso_outputs(tmp_path: Path) -> None:
    omega, a2F = _einstein(1.0, w_e=0.01, width=2.0e-4)
    coupling = _coupling_from_a2f(omega, a2F, np.array([[1.2, 0.2], [0.2, 0.4]]), [0.3, 0.7])
    coupling.update(
        energy_ev=np.array([0.01, -0.02]), band=np.array([2, 3]),
        k_cryst=np.zeros((2, 3)), bg=np.eye(3),
    )  # fmt: skip
    res = mea.migdal_eliashberg_aniso(coupling, [1.0, 30.0], 0.1, 0.1, n_real=400)
    lin = mea.linearized_eigenvalues_aniso(coupling, [5.0, 30.0], 0.1, 0.1)
    assert lin['max_eigenvalue'][0] > 1.0 > lin['max_eigenvalue'][1]
    mea.write_me_aniso_outputs(res, coupling, str(tmp_path), 'x', lin)
    for name in (
        'x.lambda_FS', 'x.lambda_k_pairs', 'x.imag_aniso_001.00', 'x.imag_aniso_gap0_001.00',
        'x.pade_aniso_gap0_001.00', 'x.imag_aniso_gap_FS_001.00', 'x.pade_aniso_001.00',
        'x.qdos_001.00', 'gap_vs_T_aniso.dat', 'max_eigenvalue_aniso.dat',
        'migdal_eliashberg_aniso.npz',
    ):  # fmt: skip
        assert os.path.isfile(tmp_path / name), name
    imag = np.loadtxt(tmp_path / 'x.imag_aniso_001.00')
    assert imag.shape == (2 * res['imag'][0][0].size, 4)
    with np.load(tmp_path / 'migdal_eliashberg_aniso.npz') as data:
        distribution = data['gap0_distribution'][0]
        grid = data['gap_grid_mev']
        assert np.sum(distribution) * (grid[1] - grid[0]) == pytest.approx(1.0, rel=0.02)
