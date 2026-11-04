"""Coupled second-order response of a molecule: conventions against the sum-over-states code,
static limit against finite fields, Kleinman symmetry, origin independence, monopole screening."""

import numpy as np
import pytest

from PAOFLOW.response import coupled_response as cr
from PAOFLOW.response import nonlinear_optics as nlo
from PAOFLOW.response.molecular_nlo import hyperpolarizability

pytestmark = pytest.mark.unit

HA = cr.HARTREE_EV


def _chain(nsite=10, spacing=2.6, t0=-1.2, lam=1.3, ramp=2.0, bend=0.3, noise=0.05, seed=0):
    """Zigzag push-pull chain, one orbital per site: H (eV), tau (bohr)."""
    rng = np.random.default_rng(seed)
    x = np.arange(nsite) * spacing
    y = bend * spacing * (np.arange(nsite) % 2) + noise * rng.standard_normal(nsite)
    z = noise * rng.standard_normal(nsite)
    tau = np.stack([x, y, z], axis=1)
    tau -= tau.mean(0)
    d = np.linalg.norm(tau[:, None] - tau[None], axis=-1)
    H = t0 * np.exp(-(d - spacing) / lam)
    np.fill_diagonal(H, ramp * np.linspace(-1.0, 1.0, nsite) + 0.4 * (np.arange(nsite) % 2))
    return 0.5 * (H + H.T), tau


def _channel(H_ev, tau, shift=None):
    tau = tau if shift is None else tau + np.asarray(shift)[None, :]
    D = np.stack([np.diag(tau[:, a]) for a in range(3)])
    return cr.MolecularChannel(H_ev / HA, D, nocc=H_ev.shape[0] // 2)


def _monopole(tau, shift=None, U=0.4):
    tau = tau if shift is None else tau + np.asarray(shift)[None, :]
    n = tau.shape[0]
    return cr.MonopoleKernel(np.arange(n), tau, np.full(n, U))


def test_independent_particles_match_the_sum_over_states_code():
    """Kernel off: beta_T(-2w; w, w) equals -2 eps0 Omega chi of nonlinear_optical_spectra."""
    H, tau = _chain()
    n = H.shape[0]
    L = np.array([80.0, 50.0, 40.0])
    HRs = np.zeros((n, n, 1, 1, 1, 1), dtype=complex)
    HRs[:, :, 0, 0, 0, 0] = H
    energy = np.array([0.0, 0.15])  # 2w below the 0.5 eV gap: off resonance
    eta_ev = 0.02
    res = nlo.nonlinear_optical_spectra(
        HRs, np.diag(L / L[0]), float(L[0]), float(np.prod(L)), tau / L[None, :], float(n),
        (1, 1, 1), energy, components='all', eta=eta_ev, spin_factor=2.0, degeneracy_tol=1e-6,
        shg_bin=eta_ev / 50, want=('shg',),
    )  # fmt: skip
    code = hyperpolarizability(res['shg'], float(np.prod(L)))
    ci = {c: i for i, c in enumerate(res['components'])}
    out = cr.coupled_hyperpolarizability(
        _channel(H, tau), cr.CoulombKernel(), energy / HA, eta_ev / HA, static_step=None
    )
    # off resonance the real parts agree; the imaginary parts carry the damping convention
    # (eta in every denominator here, in the resonant ones of the sum over states) and differ
    # by a few per cent of the dominant component
    for comp in ('xxx', 'xxy', 'yxx', 'xyy', 'yyy', 'zxx', 'xyz'):
        a, b, c = ['xyz'.index(s) for s in comp]
        for ie in range(energy.size):
            ref = code[ci[comp], ie]
            mine = out['beta_T'][ie, a, b, c]
            scale = abs(code[ci['xxx'], ie])
            assert abs(mine.real - ref.real) < 0.01 * scale, (comp, ie, mine, ref)
            assert abs(mine.imag - ref.imag) < 0.05 * scale, (comp, ie, mine, ref)


@pytest.mark.parametrize('coupled', [False, True])
def test_static_limit_equals_the_finite_field_route(coupled):
    H, tau = _chain()
    ch = _channel(H, tau)
    kernel = _monopole(tau) if coupled else cr.CoulombKernel()
    out = cr.coupled_hyperpolarizability(ch, kernel, [0.0], 1.0e-6, static_step='auto')
    beta_w = out['beta_T'][0].real
    beta_ff = out['beta_T_static']
    alpha_w = out['alpha'][0].real
    scale = np.abs(beta_ff).max()
    assert np.abs(beta_w - beta_ff).max() < 2.0e-3 * scale
    assert np.abs(alpha_w - out['alpha_static']).max() < 1.0e-4 * np.abs(alpha_w).max()
    assert np.abs(out['beta_T'][0].imag).max() < 1.0e-6 * scale
    assert np.abs(alpha_w - alpha_w.T).max() < 1.0e-8 * np.abs(alpha_w).max()


def test_static_tensor_has_kleinman_symmetry():
    H, tau = _chain()
    ch = _channel(H, tau)
    assert 1.0e-6 < cr.auto_field_step(ch) < 1.0e-4  # 0.5 eV gap over a 23 bohr chain
    beta = cr.static_hyperpolarizability(ch, _monopole(tau))['beta_T']
    scale = np.abs(beta).max()
    for perm in ((1, 0, 2), (2, 1, 0), (0, 2, 1), (1, 2, 0)):
        assert np.abs(beta - np.transpose(beta, perm)).max() < 1.0e-4 * scale


def test_response_is_origin_independent():
    H, tau = _chain()
    shift = np.array([7.3, -2.1, 4.4])
    w = np.array([0.0, 0.4]) / HA
    ref = cr.coupled_hyperpolarizability(_channel(H, tau), _monopole(tau), w, 0.01 / HA, None)
    moved = cr.coupled_hyperpolarizability(
        _channel(H, tau, shift), _monopole(tau, shift), w, 0.01 / HA, None
    )
    assert np.abs(ref['beta_T'] - moved['beta_T']).max() < 1.0e-8 * np.abs(ref['beta_T']).max()
    assert np.abs(ref['alpha'] - moved['alpha']).max() < 1.0e-8 * np.abs(ref['alpha']).max()


def test_monopole_screening_reduces_the_response():
    H, tau = _chain()
    ch = _channel(H, tau)
    free = cr.coupled_hyperpolarizability(ch, cr.CoulombKernel(), [0.0], 1.0e-6, None)
    scr = cr.coupled_hyperpolarizability(ch, _monopole(tau), [0.0], 1.0e-6, None)
    assert 0.0 < scr['alpha'][0, 0, 0].real < free['alpha'][0, 0, 0].real
    assert abs(scr['beta_T'][0, 0, 0, 0]) < abs(free['beta_T'][0, 0, 0, 0])
    assert np.sign(scr['beta_T'][0, 0, 0, 0].real) == np.sign(free['beta_T'][0, 0, 0, 0].real)


def test_kernel_from_species_uses_the_table_and_the_fallback():
    tau = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    k = cr.kernel_from_species([0, 1, 1, 2], tau, ['C', 'o', 'Xx'], hubbard_u={'Xx': 0.25})
    assert np.allclose(np.diag(k.gamma), [cr.HUBBARD_U_HA['C'], cr.HUBBARD_U_HA['O'], 0.25])
    assert k.gamma[0, 1] < 0.5 and k.gamma[0, 2] < k.gamma[0, 1]
    dP = np.diag([0.1, 0.0, -0.1, 0.0]).astype(complex)
    assert np.allclose(k.charges(dP), [0.1, -0.1, 0.0])


def test_pw92_correlation_matches_the_tabulated_values():
    rs = np.array([1.0, 2.0, 5.0, 10.0])
    e_c, de_c = cr.pw92_correlation(rs)
    assert np.allclose(e_c, [-0.05975, -0.04479, -0.02819, -0.01857], atol=3.0e-4)
    h = 1.0e-5
    num = (cr.pw92_correlation(rs + h)[0] - cr.pw92_correlation(rs - h)[0]) / (2 * h)
    assert np.allclose(de_c, num, rtol=1.0e-6)
    n = 1.0e3  # high density: exchange dominates
    assert abs(cr.lda_potential(n) / (-((3.0 * n / np.pi) ** (1.0 / 3.0))) - 1.0) < 0.02
    f, g = cr.lda_kernels(np.array([1.0e-12, 1.0e-2, 1.0e-1]))
    assert f[0] == 0.0 and g[0] == 0.0 and f[1] < f[2] < 0.0 and g[1] > g[2] > 0.0


def test_grid_kernel_reproduces_the_analytic_gaussian_integrals():
    """(ab|cd) from the cutoff Poisson solve on the grid against pyints for s Gaussians."""
    from PAOFLOW.utils.pyints import coulomb_repulsion

    L, nr = 24.0, 64
    x = np.arange(nr) * L / nr
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    centres = np.array([[12.0, 12.0, 12.0], [13.5, 12.4, 11.2], [10.8, 13.1, 12.9]])
    alphas = np.array([0.8, 1.0, 0.6])
    phi = []
    for c, a in zip(centres, alphas):
        r2 = (X - c[0]) ** 2 + (Y - c[1]) ** 2 + (Z - c[2]) ** 2
        phi.append((2 * a / np.pi) ** 0.75 * np.exp(-a * r2))
    kernel = cr.GridKernel(np.array(phi), L * np.eye(3), L**3)
    assert abs(kernel.cutoff_radius - L / 2) < 1e-12
    quads = [(0, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 2, 2), (1, 2, 0, 1)]
    grid = kernel.coulomb_integrals(quads).real
    for (a, b, c, d), val in zip(quads, grid):
        args = []
        for i in (a, b, c, d):
            args += [centres[i], (2 * alphas[i] / np.pi) ** 0.75, (0, 0, 0), alphas[i]]
        ref = coulomb_repulsion(*args)
        assert abs(val - ref) < 2.0e-3 * abs(ref), ((a, b, c, d), val, ref)


def test_grid_kernel_density_and_potential_are_consistent_with_the_orbital_table():
    L, nr = 10.0, 24
    rng = np.random.default_rng(1)
    phi = rng.standard_normal((4, nr, nr, nr))
    kernel = cr.GridKernel(phi, L * np.eye(3), L**3)
    kernel.block_rows = 1000  # exercise the blocking
    dP = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    dn = kernel.density(dP)
    table = phi.reshape(4, -1).T
    ref = np.einsum('rm,nm,rn->r', table, dP, table)
    assert np.allclose(dn, ref)
    V = rng.standard_normal(nr**3) + 1j * rng.standard_normal(nr**3)
    assert np.allclose(kernel.potential_matrix(V), (table.T @ (V[:, None] * table)) * L**3 / nr**3)
    cplx = cr.GridKernel(phi * (1.0 + 0.5j), L * np.eye(3), L**3)
    cplx.block_rows = 1000
    tc = table * (1.0 + 0.5j)
    assert np.allclose(cplx.density(dP), np.einsum('rm,nm,rn->r', tc, dP, tc.conj()))
    assert np.allclose(cplx.potential_matrix(V), (tc.conj().T @ (V[:, None] * tc)) * L**3 / nr**3)


def test_active_space_restricts_the_response():
    H, tau = _chain()
    ch = _channel(H, tau)
    n = H.shape[0]
    full = cr.coupled_hyperpolarizability(ch, cr.CoulombKernel(), [0.0], 1e-6, None)
    small = cr.MolecularChannel(H / HA, ch.dipole, nocc=n // 2, active=n // 2 + 1)
    out = cr.coupled_hyperpolarizability(small, cr.CoulombKernel(), [0.0], 1e-6, None)
    # only the LUMO is empty in the active space: the sum over states of the occupied -> LUMO pairs
    e, C = np.linalg.eigh(H / HA)
    X = [C.conj().T @ ch.dipole[a] @ C for a in range(3)]
    l = n // 2
    alpha_2l = sum(4.0 * abs(X[0][h, l]) ** 2 / (e[l] - e[h]) for h in range(l))
    assert abs(out['alpha'][0, 0, 0].real - alpha_2l) < 1e-6 * alpha_2l
    assert abs(out['alpha'][0, 0, 0]) < abs(full['alpha'][0, 0, 0])
    with pytest.raises(ValueError):
        cr.MolecularChannel(H / HA, ch.dipole, nocc=n // 2, active=n // 2)


def test_distributed_solves_match_the_serial_result():
    MPI = pytest.importorskip('mpi4py.MPI')
    H, tau = _chain()
    ch = _channel(H, tau)
    k = _monopole(tau)
    w = np.array([0.0, 0.3]) / HA
    ref = cr.coupled_hyperpolarizability(ch, k, w, 0.01 / HA, static_step=1e-5)
    par = cr.coupled_hyperpolarizability(
        ch, k, w, 0.01 / HA, static_step=1e-5, comm=MPI.COMM_WORLD, n_threads=3
    )
    for key in ('alpha', 'beta_T', 'alpha_static', 'beta_T_static'):
        assert np.allclose(par[key], ref[key], rtol=1e-10, atol=1e-8 * np.abs(ref[key]).max())
