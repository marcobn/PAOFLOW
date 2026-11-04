"""Crystal coupled response: periodic Hartree on a cosine density, sphere-to-grid normalisation,
kernel-off epsilon against the sum over states, the long-range kernel against its closed form."""

import numpy as np
import pytest

from PAOFLOW.response import coupled_crystal as cc

pytestmark = pytest.mark.unit

HA = cc.HARTREE_EV


def test_periodic_hartree_of_a_cosine_density():
    L, nr = 10.0, 20
    k = cc.PeriodicKernel((nr, nr, nr), L * np.eye(3), L**3)
    x = np.arange(nr) * L / nr
    X = np.meshgrid(x, x, x, indexing='ij')[0]
    G = 2 * np.pi / L
    dn = np.cos(G * X).reshape(-1)
    V = k.apply_density(dn).real
    assert np.allclose(V, 4 * np.pi / G**2 * dn, atol=1e-10)
    assert abs(k.apply_density(np.ones(nr**3))).max() < 1e-12  # G = 0 dropped
    assert cc.PeriodicKernel((nr, nr, nr), L * np.eye(3), L**3, hartree=False).label == 'none'


def test_miller_sphere_and_grid_normalisation():
    bg = np.eye(3)  # cubic cell, alat units
    mill = cc.miller_sphere(np.zeros(3), bg, 4.0)
    assert mill.shape[0] == 3 and set(map(tuple, mill.T)) == {
        tuple(m)
        for m in np.stack(np.meshgrid(*[range(-2, 3)] * 3, indexing='ij'), -1).reshape(-1, 3)
        if np.dot(m, m) <= 4.0
    }
    shifted = cc.miller_sphere(np.array([0.5, 0.0, 0.0]), bg, 4.0)
    assert shifted.shape[1] != mill.shape[1]
    rng = np.random.default_rng(0)
    coeff = rng.standard_normal((3, mill.shape[1])) + 1j * rng.standard_normal((3, mill.shape[1]))
    coeff /= np.linalg.norm(coeff, axis=1)[:, None]
    nr = cc.grid_for_sphere(mill)
    omega = 7.0
    u = cc.sphere_to_grid(coeff, mill, nr, omega)
    norms = (np.abs(u) ** 2).sum(0) * omega / np.prod(nr)
    assert np.allclose(norms, 1.0)


def _synthetic_channel(nk=3, n=4, nocc=2, seed=1):
    """Random band structure on a nk^3 grid with random orthonormal periodic parts."""
    rng = np.random.default_rng(seed)
    nkd = nk**3
    e = np.sort(rng.uniform(-0.3, 0.0, (nkd, nocc)), axis=1)
    e = np.concatenate([e, np.sort(rng.uniform(0.2, 0.6, (nkd, n - nocc)), axis=1)], axis=1)
    U = np.stack(
        [
            np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))[0]
            for _ in range(nkd)
        ]
    )
    r = rng.standard_normal((nkd, 3, n, n)) + 1j * rng.standard_normal((nkd, 3, n, n))
    r = 0.5 * (r + np.conj(np.swapaxes(r, -1, -2)))
    for ik in range(nkd):
        for a in range(3):
            np.fill_diagonal(r[ik, a], 0.0)
    mill = cc.miller_sphere(np.zeros(3), np.eye(3), 2.0)
    coeff = [
        np.linalg.qr(
            (rng.standard_normal((mill.shape[1], n)) + 1j * rng.standard_normal((mill.shape[1], n)))
        )[0].T
        for _ in range(nkd)
    ]
    f = np.zeros(n)
    f[:nocc] = 2.0
    return cc.CrystalChannel(
        energies=e, eigenvectors=U, position=r, occupations=f, weights=np.full(nkd, 1.0 / nkd),
        coeff=coeff, mill=[mill] * nkd, nr=cc.grid_for_sphere(mill), omega=50.0, at_bohr=50.0 ** (1 / 3) * np.eye(3),
    )  # fmt: skip


def test_kernel_off_matches_the_sum_over_states():
    ch = _synthetic_channel()
    kernel = cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, hartree=False)
    w, eta = 0.05, 0.01
    out = cc.dielectric_function_crystal(ch, kernel, [w * HA], eta * HA)
    eps = out['epsilon'][0]
    f = ch.occupations
    chi = np.zeros((3, 3), dtype=complex)
    for ik in range(ch.nk_local):
        e = ch.energies[ik]
        c = (f[None, :] - f[:, None]) / (w - (e[:, None] - e[None, :]) + 1j * eta)
        for b in range(3):
            dP = c * ch.position[ik, b]
            for a in range(3):
                chi[a, b] -= ch.weights[ik] * np.einsum('ij,ji', ch.position[ik, a], dP) / ch.omega
    assert np.allclose(eps, np.eye(3) + 4 * np.pi * chi, rtol=1e-10, atol=1e-12)


def test_long_range_kernel_matches_its_closed_form():
    ch = _synthetic_channel()
    w, eta, alpha = 0.0, 0.01, 0.3
    ipa = cc.dielectric_function_crystal(
        ch, cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, hartree=False), [w], eta * HA
    )
    lrc = cc.dielectric_function_crystal(
        ch,
        cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, hartree=False, lrc_alpha=alpha),
        [w],
        eta * HA,
    )
    chi0 = (ipa['epsilon'][0] - np.eye(3)) / (4 * np.pi)
    # P = chi0 (E + alpha P)  ->  P = (1 - alpha chi0)^-1 chi0 E
    chi = np.linalg.solve(np.eye(3) - alpha * chi0, chi0)
    assert np.allclose(lrc['epsilon'][0], np.eye(3) + 4 * np.pi * chi, rtol=1e-7, atol=1e-9)
    assert cc.bootstrap_alpha(12.0, 11.0) == pytest.approx(4 * np.pi / (11.0 * 11.0))


def test_hartree_kernel_screens_and_keeps_the_density_neutral():
    ch = _synthetic_channel()
    hartree = cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega)
    out = cc.first_order_crystal(ch, hartree, 0, 0.0, 0.01 * HA)
    assert abs(out['density'].sum()) < 1e-8 * np.abs(out['density']).max()
    ipa = cc.first_order_crystal(
        ch, cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, hartree=False), 0, 0.0, 0.01 * HA
    )
    assert out['iterations'] > 0 and np.isfinite(out['polarization']).all()
    assert not np.allclose(out['polarization'], ipa['polarization'])
    n0 = ch.ground_state_density()
    assert n0.min() >= 0.0 and abs(n0.sum() * ch.omega / n0.size - ch.occupations.sum()) < 1e-8
    alda = cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, xc='lda', n0=n0)
    out2 = cc.first_order_crystal(ch, alda, 0, 0.0, 0.01 * HA)
    assert np.isfinite(out2['polarization']).all() and alda.label == 'hartree+alda'
