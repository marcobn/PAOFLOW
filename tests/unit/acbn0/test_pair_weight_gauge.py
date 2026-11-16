"""eACBN0 pair density matrices: independent of the basis chosen inside degenerate multiplets.

``eACBN0._pair_density_matrices`` weights each occupied state by its Mulliken projection on the two shells of the
pair. For a pair of different shells (s on I, p on J) that weight is not the same for every member of a degenerate
multiplet, so it is used as a matrix inside the multiplet: the result must not change when the degenerate Bloch
states are rotated among themselves, and must equal the per-band weight when nothing is degenerate.
"""

from __future__ import annotations

import itertools

import numpy as np

from PAOFLOW.ACBN0 import eACBN0
from PAOFLOW.models.sk_fitting import sk_element

ORBITALS = ['s', 'px', 'py', 'pz']


def _diamond_sp3(nk: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nearest-neighbour sp3 model of a diamond crystal (Harrison-like parameters, eV, bohr): H(k), S(k), k, weights."""
    a = 10.26
    lattice = a / 2 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    tau = [np.zeros(3), np.full(3, a / 4)]
    params = {'sss': -1.82, 'sps': 1.96, 'pps': 3.06, 'ppp': -0.87}
    onsite = [-13.55, -5.3, -5.3, -5.3]
    hoppings = {}
    for i, j in itertools.product(range(2), repeat=2):
        for R in itertools.product(range(-1, 2), repeat=3):
            d = tau[j] + np.asarray(R) @ lattice - tau[i]
            dist = np.linalg.norm(d)
            if dist < 1e-8 or dist > 0.45 * a:
                continue
            block = np.array(
                [[sk_element(oa, ob, *(d / dist), params) for ob in ORBITALS] for oa in ORBITALS]
            )
            hoppings.setdefault(R, np.zeros((8, 8)))[4 * i : 4 * i + 4, 4 * j : 4 * j + 4] += block
    hoppings.setdefault((0, 0, 0), np.zeros((8, 8)))[np.diag_indices(8)] += onsite * 2
    grid = np.arange(nk) / nk
    k_crystal = np.array(list(itertools.product(grid, grid, grid)))
    k_cart = k_crystal @ (2 * np.pi * np.linalg.inv(lattice).T)
    hks = np.zeros((8, 8, len(k_cart)), dtype=complex)
    for R, block in hoppings.items():
        phase = np.exp(1j * k_cart @ (np.asarray(R) @ lattice))
        hks += block[:, :, None] * phase[None, None, :]
    # four occupied bands: put the Fermi level in the gap
    energies = np.array([np.linalg.eigvalsh(hks[:, :, ik]) for ik in range(len(k_cart))])
    fermi = 0.5 * (energies[:, 3].max() + energies[:, 4].min())
    hks -= fermi * np.eye(8)[:, :, None]
    sks = np.repeat(np.eye(8, dtype=complex)[:, :, None], len(k_cart), axis=2)
    return hks, sks, k_cart, np.full(len(k_cart), 1.0 / len(k_cart))


def _rotate_degenerate(cache: list, rng: np.random.Generator, tol: float = 1e-8) -> list:
    rotated = []
    for energies, vectors in cache:
        vectors = vectors.copy()
        start = 0
        while start < len(energies):
            stop = start + 1
            while stop < len(energies) and abs(energies[stop] - energies[start]) < tol:
                stop += 1
            if stop - start > 1:
                z = rng.normal(size=(stop - start,) * 2) + 1j * rng.normal(size=(stop - start,) * 2)
                vectors[:, start:stop] = vectors[:, start:stop] @ np.linalg.qr(z)[0]
            start = stop
        rotated.append((energies, vectors))
    return rotated


def test_pair_matrices_do_not_depend_on_the_degenerate_basis():
    hks, sks, k_cart, weights = _diamond_sp3()
    driver = eACBN0.__new__(eACBN0)
    cache = driver._eigh_all_k(hks, sks)
    s_on_I, p_on_J = np.array([0]), np.array([5, 6, 7])
    reference = driver._pair_density_matrices(
        s_on_I, p_on_J, hks, sks, k_cart, weights, np.zeros(3), eigvec_cache=cache
    )
    rng = np.random.default_rng(11)
    degenerate = sum(np.any(np.abs(np.diff(e[e <= 0])) < 1e-8) for e, _ in cache)
    assert degenerate > 0  # the grid contains degenerate occupied multiplets (Gamma)
    for _ in range(3):
        rotated = _rotate_degenerate(cache, rng)
        result = driver._pair_density_matrices(
            s_on_I, p_on_J, hks, sks, k_cart, weights, np.zeros(3), eigvec_cache=rotated
        )
        for key in ('P_II', 'P_JJ', 'P_IJ', 'P_JI', 'n_IJ'):
            assert np.allclose(result[key], reference[key], atol=1e-12), key


def test_nondegenerate_states_keep_the_per_band_weight():
    rng = np.random.default_rng(5)
    nbasis, nk = 6, 3
    hks = np.empty((nbasis, nbasis, nk), dtype=complex)
    for ik in range(nk):
        a = rng.normal(size=(nbasis, nbasis)) + 1j * rng.normal(size=(nbasis, nbasis))
        hks[:, :, ik] = 0.5 * (a + a.conj().T)
    b = rng.normal(size=(nbasis, nbasis))
    overlap = np.eye(nbasis) + 0.05 * (b + b.T)
    sks = np.repeat(overlap[:, :, None].astype(complex), nk, axis=2)
    k_cart = rng.normal(size=(nk, 3))
    weights = np.full(nk, 1.0 / nk)
    R = np.array([0.3, -0.2, 0.5])
    basis_I, basis_J = np.array([0, 1]), np.array([3, 4, 5])
    driver = eACBN0.__new__(eACBN0)
    cache = driver._eigh_all_k(hks, sks)
    result = driver._pair_density_matrices(
        basis_I, basis_J, hks, sks, k_cart, weights, R, eigvec_cache=cache
    )

    # per-band weight N_m (the definition for non-degenerate states)
    expected = {key: 0 for key in ('P_IJ', 'P_II')}
    for ik, (energies, vectors) in enumerate(cache):
        occupied = vectors[:, energies <= 0.0]
        s_occupied = sks[:, :, ik] @ occupied
        n_band = (np.einsum('im,im->m', occupied[basis_I].conj(), s_occupied[basis_I]).real
                  + np.einsum('jm,jm->m', occupied[basis_J].conj(), s_occupied[basis_J]).real)  # fmt: skip
        phase = np.exp(1j * k_cart[ik] @ R)
        for key, a, b, ph in (('P_IJ', basis_I, basis_J, phase), ('P_II', basis_I, basis_I, 1.0)):
            ca, sa, cb, sb = occupied[a], s_occupied[a], occupied[b], s_occupied[b]
            expected[key] = expected[key] + 0.5 * weights[ik] * ph * (
                ca.conj() @ (sb * n_band).T + sa.conj() @ (cb * n_band).T
            )
    for key, value in expected.items():
        assert np.allclose(result[key], value, atol=1e-12), key
