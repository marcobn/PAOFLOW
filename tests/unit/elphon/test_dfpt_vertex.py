"""Native DFPT vertex: exact identities on a toy plane-wave crystal and the star bookkeeping."""

import numpy as np
import pytest

from PAOFLOW.elphon.dfpt_io import SpaceGroup
from PAOFLOW.elphon.dfpt_vertex import (
    StarMember,
    cartesian_perturbation_potentials,
    rotate_periodic_field,
    star_members,
)
from PAOFLOW.elphon.plane_waves import periodic_part_on_grid, project_on_plane_waves
from PAOFLOW.elphon.pseudopotential_terms import bare_local_perturbation, nonlocal_perturbation

from .toy_plane_waves import ToyCrystal

pytestmark = pytest.mark.unit


def _bare_vertex(toy, blk_k, blk_kq, umklapp, q_cryst):
    """<psi_{m,k+q}| dV_bare/du_c |psi_{n,k}> (3, nb, nb) of the toy crystal."""
    pot = bare_local_perturbation(
        toy.vloc,
        toy.species,
        toy.tau_alat,
        q_cryst @ toy.bg,
        toy.g_miller,
        toy.g_cart,
        toy.tpiba,
        toy.fft_grid,
    )
    u_r = periodic_part_on_grid(blk_k.coefficients, blk_k.miller, toy.fft_grid)
    dvpsi = project_on_plane_waves(pot[:, None] * u_r[None], blk_kq.miller, toy.fft_grid, umklapp)
    kpg_k = blk_k.miller.T @ toy.bg + blk_k.xk_cart
    kpg_kq = blk_kq.miller.T @ toy.bg + blk_kq.xk_cart
    dvpsi += nonlocal_perturbation(
        toy.kb, blk_k.coefficients, kpg_k, kpg_kq, toy.tau_alat, toy.tpiba
    )
    return np.einsum('mG,cnG->cmn', np.conj(blk_kq.coefficients), dvpsi)


def test_translational_sum_rule_at_gamma():
    """sum_kappa g_mn,kappa alpha(k, 0) = (E_m - E_n) <m| d_alpha |n> for the exact eigenstates."""
    toy = ToyCrystal()
    k = np.array([0.1, -0.2, 0.15])
    blk, e, kpg = toy.block(k)
    g = _bare_vertex(toy, blk, blk, np.zeros(3, int), np.zeros(3))
    grad = np.einsum(
        'mG,aG,nG->amn', np.conj(blk.coefficients), 1j * toy.tpiba * kpg.T, blk.coefficients
    )
    rhs = (e[:, None] - e[None, :])[None] * grad
    assert np.abs(g - rhs).max() < 1e-10 * np.abs(rhs).max()


def test_hermiticity_between_q_and_minus_q_with_umklapp():
    """g_mn(k, q)^* = g_nm(k+q, -q) when k + q leaves the zone (G0 != 0)."""
    toy = ToyCrystal()
    k1 = np.array([0.4, 0.3, 0.0])
    q = np.array([0.3, 0.2, 0.1])
    g0 = np.array([1, 0, 0])
    k2 = k1 + q - g0  # the stored point of k1 + q
    b1, _, _ = toy.block(k1)
    b2, _, _ = toy.block(k2)
    g_plus = _bare_vertex(toy, b1, b2, g0, q)  # (c, m at k2, n at k1)
    g_minus = _bare_vertex(toy, b2, b1, -g0, -q)  # (c, m at k1, n at k2)
    assert np.abs(np.conj(g_plus) - g_minus.transpose(0, 2, 1)).max() < 1e-12
    assert np.abs(g_plus).max() > 1e-3  # the test is not vacuous


def test_umklapp_route_matches_shifted_potential():
    """Gathering at G'' - G0 equals shifting the potential by exp(i G0.r) and gathering at G''."""
    toy = ToyCrystal()
    k1 = np.array([0.4, 0.3, 0.0])
    q = np.array([0.3, 0.2, 0.1])
    g0 = np.array([1, 0, 0])
    b1, _, _ = toy.block(k1)
    b2, _, _ = toy.block(k1 + q - g0)
    pot = bare_local_perturbation(
        toy.vloc,
        toy.species,
        toy.tau_alat,
        q @ toy.bg,
        toy.g_miller,
        toy.g_cart,
        toy.tpiba,
        toy.fft_grid,
    )
    u_r = periodic_part_on_grid(b1.coefficients, b1.miller, toy.fft_grid)
    direct = project_on_plane_waves(pot[:, None] * u_r[None], b2.miller, toy.fft_grid, g0)
    x = np.stack(np.meshgrid(*[np.arange(n) / n for n in toy.fft_grid], indexing='ij'), axis=-1)
    shifted = pot * np.exp(2j * np.pi * (x @ g0))[None]
    other = project_on_plane_waves(shifted[:, None] * u_r[None], b2.miller, toy.fft_grid)
    assert np.abs(direct - other).max() < 1e-13 * np.abs(direct).max()


def test_rotate_periodic_field_against_analytic_rotation():
    nr = (6, 6, 4)
    x = np.stack(np.meshgrid(*[np.arange(n) / n for n in nr], indexing='ij'), axis=-1)
    rot = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])  # 90 degrees about z (crystal coords)
    t = np.array([0.5, 1.0 / 6.0, 0.0])
    f = lambda y: np.cos(2 * np.pi * (y @ [1, 0, 0])) + 0.5 * np.sin(2 * np.pi * (y @ [0, 2, 1]))
    field = f(x)
    rotated = rotate_periodic_field(field, rot, t)
    expected = f((x + t) @ np.linalg.inv(rot).T)  # f(g^-1 x), g: x -> R x - t
    assert np.abs(rotated - expected).max() < 1e-12
    with pytest.raises(ValueError):
        rotate_periodic_field(field, rot, np.array([0.3, 0.0, 0.0]))  # not on the grid


def _cubic_group():
    """O_h as signed permutation matrices (no fractional translations, one atom)."""
    rots = []
    for perm in ([0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]):
        for signs in np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3):
            m = np.zeros((3, 3), int)
            for i, p in enumerate(perm):
                m[i, p] = signs[i]
            rots.append(m)
    rots = np.array(rots)
    nsym = rots.shape[0]
    return SpaceGroup(
        rots, np.zeros((nsym, 3)), np.zeros((nsym, 1), int), np.zeros(nsym, bool), tuple('?' * nsym)
    )


def test_star_members_tile_the_grid():
    group = _cubic_group()
    assert group.nsym == 48
    q_irr = np.array([[0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0.5, 0.5, 0.5]])
    members = star_members(q_irr, group, (2, 2, 2), np.eye(3), np.eye(3))
    assert len(members) == 8
    labels = {tuple(np.rint(m.q_cryst * 2).astype(int)) for m in members}
    assert len(labels) == 8
    for m in members:
        sq = group.rotations[m.isym].astype(float) @ q_irr[m.iq_irr]  # cubic: R_cart = R
        if m.time_reversed:
            sq = -sq
        assert np.allclose(sq + m.umklapp, m.q_cryst)
    with pytest.raises(ValueError):
        star_members(q_irr[:2], group, (2, 2, 2), np.eye(3), np.eye(3))


def test_cartesian_perturbation_potentials_identity_time_reversal_and_umklapp():
    rng = np.random.default_rng(0)
    nr = (4, 4, 4)
    u = np.linalg.qr(rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3)))[0]
    dv_pat = rng.standard_normal((3,) + nr) + 1j * rng.standard_normal((3,) + nr)
    group = _cubic_group()
    tau = np.array([[0.0, 0.0, 0.0]])
    q0 = np.array([0.25, 0.0, 0.0])
    ident = StarMember(q0, 0, 0, False, np.zeros(3, int))
    out = cartesian_perturbation_potentials(dv_pat, u, q0, ident, group, tau, np.eye(3), np.eye(3))
    assert np.allclose(out, np.einsum('cv,v...->c...', np.conj(u), dv_pat))
    # time reversal: the representative of -q0 is (0.75, 0, 0) = -q0 + (1, 0, 0)
    trev = StarMember(np.array([0.75, 0.0, 0.0]), 0, 0, True, np.array([1, 0, 0]))
    out_tr = cartesian_perturbation_potentials(
        dv_pat, u, q0, trev, group, tau, np.eye(3), np.eye(3)
    )
    x = np.stack(np.meshgrid(*[np.arange(n) / n for n in nr], indexing='ij'), axis=-1)
    expected = np.conj(out) * np.exp(-2j * np.pi * x[..., 0])[None]
    assert np.allclose(out_tr, expected)
    # a rotated member must be reproduced by rotating the field directly
    isym = 3
    rot = group.rotations[isym]
    sq = rot.astype(float) @ q0
    rep = sq - np.floor(sq)
    member = StarMember(rep, 0, isym, False, np.rint(rep - sq).astype(int))
    out_rot = cartesian_perturbation_potentials(
        dv_pat, u, q0, member, group, tau, np.eye(3), np.eye(3)
    )
    direct = np.einsum(
        'ba,a...->b...', rot.astype(float), rotate_periodic_field(out, rot, np.zeros(3))
    )
    direct = direct * np.exp(-2j * np.pi * (x @ member.umklapp.astype(float)))[None]
    assert np.allclose(out_rot, direct)
