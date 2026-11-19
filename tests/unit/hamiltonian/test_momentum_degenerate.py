"""Momentum matrix at degenerate k-points: one basis of each subspace for the three directions.

* products of two directions summed over a degenerate subspace equal the projector traces (they
  depended on the basis when every direction rotated the subspace on its own);
* the diagonal holds the band velocities of degenerate perturbation theory, per direction;
* the result does not depend on the eigenvectors eigh returns inside a subspace;
* get_degeneracies lists every subspace once (two Kramers pairs one rounding step apart were listed
  twice, and perturb_split rotated them twice).
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.hamiltonian.do_momentum import momentum_in_eigenbasis
from PAOFLOW.spectrum.do_eigh import get_degeneracies, merge_overlapping_groups
from PAOFLOW.utils.perturb_split import perturb_split

pytestmark = pytest.mark.unit


def _degenerate_system(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 8
    unitary = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
    energies = np.array([-2.0, -1.0, -1.0, -1.0, 0.5, 0.5, 1.5, 3.0])
    hamiltonian = unitary @ np.diag(energies) @ unitary.conj().T
    dh = rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
    dh = dh + dh.conj().transpose(0, 2, 1)
    _, vectors = np.linalg.eigh(hamiltonian)
    return dh, vectors, [np.array([1, 2, 3]), np.array([4, 5])]


def test_cross_products_are_the_projector_traces():
    dh, v, degen = _degenerate_system()
    p = momentum_in_eigenbasis(dh, v, degen)
    projector = lambda idx: v[:, idx] @ v[:, idx].conj().T  # noqa: E731
    first, second = [1, 2, 3], [4, 5]
    for a, b in [(0, 0), (0, 1), (1, 2), (2, 0)]:
        exact = np.trace(projector(first) @ dh[a] @ projector(second) @ dh[b])
        summed = np.sum(p[a][np.ix_(first, second)] * p[b][np.ix_(second, first)].T)
        assert summed == pytest.approx(exact, abs=1e-12)


def test_diagonal_holds_the_velocities_of_each_direction():
    dh, v, degen = _degenerate_system()
    p = momentum_in_eigenbasis(dh, v, degen)
    for block in degen:
        for direction in range(3):
            exact = np.linalg.eigvalsh(v[:, block].conj().T @ dh[direction] @ v[:, block])
            assert np.allclose(np.real(np.diag(p[direction][np.ix_(block, block)])), exact)
            off = p[direction][np.ix_(block, block)] - np.diag(
                np.diag(p[direction][np.ix_(block, block)])
            )
            assert np.abs(off).max() < 1e-12
    old = np.array([perturb_split(dh[l], dh[l], v, degen)[0] for l in range(3)])
    d = np.arange(8)
    assert np.allclose(p[:, d, d], old[:, d, d], atol=1e-12)  # the diagonal of the former rule


def test_independent_of_the_basis_inside_a_subspace():
    dh, v, degen = _degenerate_system()
    rng = np.random.default_rng(7)
    rotation = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
    w = v.copy()
    w[:, 1:4] = w[:, 1:4] @ rotation
    assert np.allclose(
        np.abs(momentum_in_eigenbasis(dh, w, degen)), np.abs(momentum_in_eigenbasis(dh, v, degen))
    )


def test_degenerate_groups_are_listed_once():
    # two Kramers pairs 2 micro-eV apart: rounded to -0.90094 and -0.90093, each value collected all
    # four bands, so the group appeared twice
    energies = np.array([-2.0, -0.900936, -0.900936, -0.900934, -0.900934, 0.5, 0.5])
    groups = get_degeneracies(energies[None, :, None], 7)[0][0]
    assert [g.tolist() for g in groups] == [[1, 2, 3, 4], [5, 6]]
    merged = merge_overlapping_groups([[2, 3], [3, 4], [7, 8], [2, 3]])
    assert [g.tolist() for g in merged] == [[2, 3, 4], [7, 8]]


def test_perturb_split_rotates_a_repeated_subspace_once():
    dh, v, degen = _degenerate_system()
    once = perturb_split(dh[0], dh[1], v, degen)
    twice = perturb_split(dh[0], dh[1], v, degen + [degen[0]])
    assert np.allclose(once[0], twice[0]) and np.allclose(once[1], twice[1])
