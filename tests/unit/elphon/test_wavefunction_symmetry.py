"""Rotation of plane-wave Bloch functions from the irreducible wedge to the full grid."""

import itertools

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_io import SpaceGroup
from PAOFLOW.elphon.dfpt_vertex import cartesian_rotation, rotate_q
from PAOFLOW.elphon.plane_waves import PlaneWaveBlock
from PAOFLOW.elphon.wavefunction_symmetry import (
    full_grid_map,
    irreducible_representatives,
    rotate_block,
    su2_rotation,
)

pytestmark = pytest.mark.unit

PAULI = [
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


def cubic_cartesian_ops():
    """The 48 signed permutation matrices."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            ops.append(m)
    return ops


def crystal_group(at, translations=None):
    """Cubic operations as integer crystal rotations for the lattice ``at`` (rows)."""
    rots = []
    for m in cubic_cartesian_ops():
        r = np.linalg.inv(at.T) @ m @ at.T  # x' = R x with r = at^T x
        ri = np.rint(r).astype(int)
        assert np.abs(r - ri).max() < 1e-12
        rots.append(ri)
    rots = np.array(rots)
    nsym = len(rots)
    t = np.zeros((nsym, 3)) if translations is None else np.asarray(translations, dtype=float)
    return SpaceGroup(
        rots, t, np.zeros((nsym, 1), dtype=int), np.zeros(nsym, dtype=bool), tuple([''] * nsym)
    )


FCC_AT = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])


def bg_of(at):
    return np.linalg.inv(at).T


def test_su2_is_a_projective_homomorphism_and_rotates_pauli_vectors():
    rng = np.random.default_rng(3)
    ops = cubic_cartesian_ops()
    for _ in range(20):
        a, b = ops[rng.integers(48)], ops[rng.integers(48)]
        ua, ub, uab = su2_rotation(a), su2_rotation(b), su2_rotation(a @ b)
        assert np.allclose(ua @ ua.conj().T, np.eye(2)) and abs(np.linalg.det(ua) - 1) < 1e-12
        prod = ua @ ub
        assert np.allclose(prod, uab) or np.allclose(prod, -uab)
        assert np.allclose(su2_rotation(-a), ua)  # improper operations use the proper part
        proper = a if np.linalg.det(a) > 0 else -a
        for j in range(3):
            lhs = ua @ PAULI[j] @ ua.conj().T
            rhs = sum(proper[i, j] * PAULI[i] for i in range(3))
            assert np.allclose(lhs, rhs)


def test_full_grid_map_covers_the_grid_from_qe_representatives():
    at = np.eye(3)
    group = crystal_group(at)
    grid = (4, 4, 4)
    ng = np.array(grid)
    full = np.stack(np.meshgrid(*[np.arange(n) for n in ng], indexing='ij'), -1).reshape(-1, 3) / ng
    reps = irreducible_representatives(full, group, grid, at, bg_of(at))
    assert len(reps) == 10  # simple cubic 4^3 with O_h: (000) (100) (110) (111) (200) ... (222)
    fmap = full_grid_map(full[reps], group, grid, at, bg_of(at))
    assert fmap.nk == 64 and np.allclose(fmap.kpts_cryst, full)
    for n in range(fmap.nk):
        k = rotate_q(group.rotations[fmap.isym[n]], full[reps][fmap.source[n]], at, bg_of(at))
        if fmap.time_reversed[n]:
            k = -k
        assert np.allclose(k + fmap.umklapp[n], fmap.kpts_cryst[n])
    # the representatives map onto themselves by the identity (first operation found)
    for i, r in enumerate(reps):
        n = int(np.flatnonzero((np.abs(fmap.kpts_cryst - full[r]).max(1) < 1e-9))[0])
        assert fmap.source[n] == i and fmap.isym[n] == 0 and not fmap.time_reversed[n]
        assert not fmap.umklapp[n].any()
    with pytest.raises(ValueError, match='cover'):
        full_grid_map(full[reps][:1], group, grid, at, bg_of(at))


def test_irreducible_representatives_are_first_of_their_star_in_grid_order():
    at = FCC_AT
    group = crystal_group(at)
    grid = (3, 3, 3)
    ng = np.array(grid)
    full = np.stack(np.meshgrid(*[np.arange(n) for n in ng], indexing='ij'), -1).reshape(-1, 3) / ng
    reps = irreducible_representatives(full, group, grid, at, bg_of(at))
    assert len(reps) == 4  # fcc 3^3 with O_h and time reversal
    for r in reps:
        star = set()
        for rot in group.rotations:
            for sign in (1, -1):
                k = sign * rotate_q(rot, full[r], at, bg_of(at))
                star.add(tuple(int(x) for x in np.rint(k * ng).astype(int) % ng))
        first = min(star)  # lexicographic = grid order, third index fastest
        assert first == tuple(int(x) for x in np.rint(full[r] * ng).astype(int))


def _psi_on_points(block, bg, x):
    """psi(x) = sum_G c(G) exp(2 pi i (kappa + m) . x) for crystal coordinates x (n, 3)."""
    kappa = np.linalg.solve(bg.T, block.xk_cart)
    phases = np.exp(2j * np.pi * (x @ (kappa[:, None] + block.miller)))  # (n, npw)
    npw = block.npw
    return np.stack(
        [block.coefficients[:, s * npw : (s + 1) * npw] @ phases.T for s in range(block.npol)], 0
    )  # (npol, nbnd, n)


@pytest.mark.parametrize('npol', [1, 2])
@pytest.mark.parametrize('time_reversed', [False, True])
def test_rotate_block_reproduces_psi_of_inverse_operation(npol, time_reversed):
    """(O_g psi)(x) = U psi(g^-1 x), then time reversal, on an fcc lattice with umklapp."""
    rng = np.random.default_rng(7)
    at, bg = FCC_AT, bg_of(FCC_AT)
    group = crystal_group(at, translations=rng.uniform(-0.5, 0.5, size=(48, 3)))
    mill = np.array([m for m in itertools.product(range(-2, 3), repeat=3) if np.dot(m, m) <= 5]).T
    npw = mill.shape[1]
    k_cryst = rng.uniform(-0.5, 0.5, 3)
    nbnd = 3
    coef = rng.normal(size=(nbnd, npol * npw)) + 1j * rng.normal(size=(nbnd, npol * npw))
    block = PlaneWaveBlock(k_cryst @ bg, mill, coef, npol)
    x = rng.uniform(0, 1, size=(5, 3))
    for isym in rng.choice(48, size=6, replace=False):
        rot, t = group.rotations[isym], group.translations[isym]
        umklapp = rng.integers(-1, 2, size=3)
        k_rot = rotate_q(rot, k_cryst, at, bg)
        target = (-k_rot if time_reversed else k_rot) + umklapp
        rotated = rotate_block(block, group, int(isym), time_reversed, target, at, bg)
        assert np.allclose(np.linalg.solve(bg.T, rotated.xk_cart), target)
        g_inv_x = (x + t) @ np.linalg.inv(rot).T  # R^-1 (x + t)
        ref = _psi_on_points(block, bg, g_inv_x)  # (npol, nbnd, n)
        if npol == 2:
            u = su2_rotation(cartesian_rotation(rot, at))
            ref = np.einsum('st,tbn->sbn', u, ref)
        if time_reversed:
            ref = np.conj(ref)
            if npol == 2:
                ref = np.stack([ref[1], -ref[0]], 0)  # i sigma_y K
        got = _psi_on_points(rotated, bg, x)
        assert np.allclose(got, ref, atol=1e-10)


def test_rotate_block_rejects_inconsistent_target():
    at, bg = np.eye(3), np.eye(3)
    group = crystal_group(at)
    mill = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    block = PlaneWaveBlock(np.array([0.25, 0.0, 0.0]), mill, np.ones((1, 3), dtype=complex), 1)
    with pytest.raises(ValueError, match='inconsistent'):
        rotate_block(block, group, 0, False, np.array([0.5, 0.0, 0.0]), at, bg)
