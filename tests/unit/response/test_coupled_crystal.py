"""Crystal coupled response: periodic Hartree on a cosine density, sphere-to-grid normalisation,
kernel-off epsilon against the sum over states, the long-range kernel against its closed form."""

import numpy as np
import pytest

from PAOFLOW.response import coupled_crystal as cc
from PAOFLOW.response import nonlinear_optics as nlo

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
    assert (
        all(cc.smooth_size(x) == x for x in nr)
        and cc.smooth_size(23) == 24
        and cc.smooth_size(25) == 25
    )
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
    assert ch.tables and len(ch.tables) == ch.nk_local
    uncached = cc.first_order_crystal(ch, hartree, 0, 0.0, 0.01 * HA, cache_gb=0)
    ch.tables = {}
    assert np.allclose(uncached['polarization'], out['polarization'])
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


def _polar_crystal(seed=2):
    """Two-orbital non-centrosymmetric tight-binding crystal on a simple cubic lattice: HRs (eV), cell."""
    n = 2
    HRs = np.zeros((n, n, 3, 3, 3, 1), dtype=complex)
    # on-site energies and intra-cell hopping (R = 0)
    H0 = np.array(
        [[-1.75, 0.6], [0.6, 2.15]]
    )  # indirect gap 0.97 eV: an insulator (band-index occupations)
    HRs[:, :, 0, 0, 0, 0] = H0
    # nearest-neighbour hoppings along +-x, +-y, +-z (indices 1 and 2 = R = +1 and -1)
    t = {0: (-0.35, -0.25, 0.15), 1: (-0.30, -0.20, 0.10), 2: (-0.33, -0.22, 0.12)}
    for axis, (t11, t22, t12) in t.items():
        for sgn, idx in ((1, 1), (-1, 2)):
            block = np.array([[t11, t12 * (1.0 + 0.3 * sgn)], [t12 * (1.0 - 0.3 * sgn), t22]])
            sl = [0, 0, 0]
            sl[axis] = idx
            HRs[:, :, sl[0], sl[1], sl[2], 0] = block
    # Hermiticity: H(-R) = H(R)^dagger
    for axis in range(3):
        sl1, sl2 = [0, 0, 0], [0, 0, 0]
        sl1[axis], sl2[axis] = 1, 2
        HRs[:, :, sl2[0], sl2[1], sl2[2], 0] = HRs[:, :, sl1[0], sl1[1], sl1[2], 0].conj().T
    at = np.eye(3)
    alat = 6.0
    positions = np.array([[0.0, 0.0, 0.0], [0.31, 0.22, 0.17]])  # off-centre second orbital
    return HRs, at, alat, alat**3, positions


def test_kernel_off_shg_matches_the_independent_particle_driver():
    HRs, at, alat, omega, pos = _polar_crystal()
    nk = (6, 6, 6)
    energy = np.array([0.3, 0.7])
    eta = 0.05
    comps = ['xxx', 'xyz', 'zxx', 'yyx']
    ref = nlo.nonlinear_optical_spectra(
        HRs, at, alat, omega, pos, 2.0, nk, energy, components=comps, eta=eta, spin_factor=2.0,
        degeneracy_tol=1e-6, shg_bin=eta / 50, want=('shg',),
    )  # fmt: skip
    ch = cc.channel_from_HRs(
        HRs, at, alat, omega, pos, 2.0, nk, scissor_ev=0.0, degeneracy_tol=1e-6
    )
    kernel = cc.PeriodicKernel((1, 1, 1), at * alat, omega, hartree=False)
    idx = [tuple('xyz'.index(ch_) for ch_ in cmp) for cmp in comps]
    out = cc.shg_crystal(ch, kernel, idx, energy, eta)
    # real parts: 1 % of the dominant component off resonance (0.3, 0.7 eV; gap 2.7 eV); the
    # imaginary parts carry the damping placement (eta in every denominator here) and differ by a
    # few per cent of the dominant component
    for ic, cmp in enumerate(comps):
        for ie in range(energy.size):
            r, m = ref['shg'][ic, ie], out['chi'][ic, ie]
            scale = np.abs(ref['shg'][:, ie]).max()
            assert abs(m.real - r.real) < 0.01 * scale, (cmp, ie, m, r)
            assert abs(m.imag - r.imag) < 0.15 * scale, (cmp, ie, m, r)


def test_long_range_kernel_second_order_closed_form():
    """LRC-only chi2 = F(2w) chi2_0 F(w) F(w), F = (1 - alpha chi1)^-1 (macroscopic field alpha P).

    The dressing of the first-order legs (potential, its covariant derivative and the derivative
    direction) is exact; the 2w feedback uses the current-based polarization and differs from the
    position-based chi1 by O(eta), hence the small eta."""
    HRs, at, alat, omega, pos = _polar_crystal()
    nk, w, eta, alpha = (4, 4, 4), 0.3, 0.005, 10.0
    ch = cc.channel_from_HRs(
        HRs, at, alat, omega, pos, 2.0, nk, scissor_ev=0.0, degeneracy_tol=1e-6
    )
    k0 = cc.PeriodicKernel((1, 1, 1), at * alat, omega, hartree=False)
    kl = cc.PeriodicKernel((1, 1, 1), at * alat, omega, hartree=False, lrc_alpha=alpha)
    assert not kl.grid_active and kl.label == 'lrc(10.000)'
    chi1 = (cc.dielectric_function_crystal(ch, k0, [w, 2 * w], eta)['epsilon'] - np.eye(3)) / (
        4 * np.pi
    )
    F1 = np.linalg.inv(np.eye(3) - alpha * chi1[0])
    F2 = np.linalg.inv(np.eye(3) - alpha * chi1[1])
    comps = [(a, b, c) for a in range(3) for b in range(3) for c in range(3)]
    c0 = cc.shg_crystal(ch, k0, comps, [w], eta)['chi_au'][:, 0].reshape(3, 3, 3)
    cl = cc.shg_crystal(ch, kl, comps, [w], eta)['chi_au'][:, 0].reshape(3, 3, 3)
    closed = np.einsum('aA,Abc,bB,cC->aBC', F2, c0, F1, F1)
    assert np.linalg.norm(cl - c0) > 0.5 * np.linalg.norm(c0)  # a strong enhancement (F up to 1.2)
    assert np.linalg.norm(cl - closed) < 2.0e-2 * np.linalg.norm(closed)
    # the dressing alone is exact: dressed first-order legs, negligible feedback
    wh, eh = w / cc.HARTREE_EV, eta / cc.HARTREE_EV
    tiny = cc.PeriodicKernel((1, 1, 1), at * alat, omega, hartree=False, lrc_alpha=1e-12)
    first = {b: cc.first_order_crystal(ch, kl, b, wh, eh) for b in range(3)}
    dressed = {
        b: dict(first[b], polarization=first[b]['polarization'] * (alpha / 1e-12)) for b in range(3)
    }
    cB = np.array(
        [
            cc.second_order_crystal(ch, tiny, 0, c, wh, eh, dressed[0], dressed[c])['polarization']
            for c in range(3)
        ]
    ).T
    closedB = np.einsum('abc,bB,cC->aBC', c0, F1, F1)[:, 0, :]
    assert np.allclose(cB, closedB, rtol=1e-8, atol=1e-10)


class _FakeController:
    """The DataController surface used by the writers: data_dicts, rank, comm, write_file_row_col."""

    def __init__(self, path):
        self.files, self.rank, self.attr = {}, 0, {'opath': str(path)}

        class _Comm:
            def Barrier(self):
                pass

        self.comm = _Comm()

    def data_dicts(self):
        return {}, self.attr

    def write_file_row_col(self, name, col1, col2):
        self.files[name] = (np.asarray(col1), np.asarray(col2))


def test_coupled_optical_spectra_nested_levels_and_bootstrap(tmp_path):
    ch = _synthetic_channel()
    energies, eta = [0.0, 0.1], 0.27
    out = cc.coupled_optical_spectra(ch, dielectric_energies_ev=energies, eta_ev=eta)
    assert out['levels'] == list(cc.CRYSTAL_KERNELS) and out['chi'] == {}
    # the independent-particle level is the direct call; every level has its static constant
    ipa = cc.PeriodicKernel(ch.nr, ch.at_bohr, ch.omega, hartree=False)
    direct = cc.dielectric_function_crystal(ch, ipa, energies, eta, directions=(0,))['epsilon']
    assert np.allclose(out['epsilon']['none'], direct)
    assert set(out['static_epsilon']) == set(cc.CRYSTAL_KERNELS)
    static = out['static_epsilon']
    assert out['alpha_boot'] == pytest.approx(cc.bootstrap_alpha(static['none'], static['hartree']))
    # the long-range kernels raise the static response above Hartree + ALDA
    assert static['hartree+alda+lrc'] > static['hartree+alda']
    assert static['hartree+alda+bootstrap'] > static['hartree+alda']
    # a given bootstrap alpha is used as is
    fixed = cc.coupled_optical_spectra(
        ch,
        levels=['hartree+alda+bootstrap'],
        dielectric_energies_ev=[0.0],
        eta_ev=eta,
        alpha_boot=0.5,
    )
    assert fixed['alpha_boot'] == 0.5 and 'none' not in fixed['static_epsilon']
    # writer: real and imaginary eps per level and direction, and the summary
    dc = _FakeController(tmp_path)
    names = cc.write_coupled_outputs(dc, out)
    for tag in cc.KERNEL_TAGS.values():
        assert 'cepsr_%s_xx.dat' % tag in names and 'cepsi_%s_xx.dat' % tag in names
    assert np.allclose(dc.files['cepsr_ipa_xx.dat'][1], direct[:, 0, 0].real)
    summary = (tmp_path / 'coupled_response.dat').read_text()
    assert 'hartree+alda+bootstrap' in summary and 'alpha_boot %.6f' % out['alpha_boot'] in summary


def test_coupled_optical_spectra_second_order_and_errors(tmp_path):
    HRs, at, alat, omega, pos = _polar_crystal()
    ch = cc.channel_from_HRs(HRs, at, alat, omega, pos, 2.0, (4, 4, 4), degeneracy_tol=1e-6)
    comps = [(0, 1, 2), (0, 0, 0)]
    out = cc.coupled_optical_spectra(
        ch, levels=['none'], shg_energies_ev=[0.3], components=comps, eta_ev=0.05
    )
    kernel = cc.PeriodicKernel((1, 1, 1), at * alat, omega, hartree=False)
    direct = cc.shg_crystal(ch, kernel, comps, [0.3], 0.05)['chi']
    assert np.allclose(out['chi']['none'], direct) and out['epsilon'] == {}
    dc = _FakeController(tmp_path)
    names = cc.write_coupled_outputs(dc, out)
    assert {'cshgr_ipa_xyz.dat', 'cshgi_ipa_xyz.dat', 'cshga_ipa_xxx.dat'} <= set(names)
    assert np.allclose(dc.files['cshga_ipa_xyz.dat'][1], np.abs(direct[0]))
    with pytest.raises(ValueError):
        cc.coupled_optical_spectra(ch, levels=['rpa'])
    with pytest.raises(ValueError):
        cc.kernel_for_level(ch, 'hartree+alda')  # no ground-state density
    with pytest.raises(ValueError):
        cc.kernel_for_level(ch, 'hartree+alda+bootstrap', n0=np.ones(1))  # no alpha


# --------------------------------------------------------------------------- metals (Drude channel)
def _models():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        '_nonlinear_models', Path(__file__).resolve().parent / '_nonlinear_models.py'
    )
    models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(models)
    return models


def test_metal_f_sum_rule_with_the_drude_channel():
    """int_0^inf w Im eps_aa dw = (2 pi^2/Omega) sum_k w_k sum_n f_n <n|d_a d_a H|n> (tight binding).

    Interband (k-dependent occupations) plus Drude: neither alone fulfils it in a metal."""
    from PAOFLOW.hamiltonian.bloch_derivatives import (
        bloch_terms_from_HRs,
        hamiltonian_derivatives_block,
        velocity_and_curvature_operators,
    )
    from PAOFLOW.response.density_matrix_response import fermi_dirac_derivatives

    model = _models().random_hopping_model(seed=2, gap=False)
    nk, mu, temperature, eta = (12, 12, 12), 0.3, 3000.0, 0.2
    ch = cc.channel_from_HRs(
        model.HRs, model.at, model.alat, model.volume, model.tau, 4.0, nk,
        fermi_energy=mu, temperature=temperature,
    )  # fmt: skip
    energies = np.concatenate([[1.0e-4], np.arange(0.02, 40.0, 0.02)])  # Drude: from ~0
    kernel = cc.kernel_for_level(ch, 'none', None, 0.0, None)
    eps = cc.dielectric_function_crystal(ch, kernel, energies, eta, directions=(2,))['epsilon']
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    K = nlo._k_grid(nk)
    H, dH, d2H = hamiltonian_derivatives_block(terms, K)
    _, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    f = fermi_dirac_derivatives(E, mu, temperature)[0]
    w_ha = energies / HA
    drude = np.zeros(1)
    for i, a in enumerate((2,)):
        waa = np.real(np.einsum('kim,kij,kjm->km', U.conj(), w[:, a, a], U)) / HA  # Ha bohr^2
        expected = 2.0 * np.pi**2 / model.volume * 2.0 * np.mean(np.sum(f * waa, axis=1))
        integral = np.trapezoid(w_ha * eps[:, a, a].imag, w_ha)
        assert abs(integral / expected - 1.0) < 0.01, (a, integral, expected)
        drude[i] = integral
    # the interband part alone (no Drude) misses the Fermi-surface weight
    insulator_like = cc.CrystalChannel(**{**ch.__dict__, 'occupation_derivative': None})
    eps_inter = cc.dielectric_function_crystal(insulator_like, kernel, energies, eta, (2,))
    inter = np.trapezoid(w_ha * eps_inter['epsilon'][:, 2, 2].imag, w_ha)
    assert inter < 0.9 * drude[0]


def test_drude_dielectric_function_of_one_band():
    """eps = 1 - w_p^2/(w (w + i gamma)), w_p^2 = (4 pi/Omega) g_s sum_k f d2 eps (one band)."""
    models = _models()
    hops = [(1, 1, (0, 0, 0), 12.0)]
    for R, h in (((1, 0, 0), -1.0), ((0, 1, 0), -0.8), ((0, 0, 1), -0.5), ((1, 1, 0), 0.2)):
        hops.append((0, 0, R, h))
        hops.append((0, 0, tuple(-np.array(R)), np.conj(h)))
    model = models.ExactModel(hops, 2, np.eye(3), 5.0, np.zeros((2, 3)), (5, 5, 5), nelec=0.8)
    nk, mu, temperature, tau = (24, 24, 24), -0.5, 1500.0, 20.0
    ch = cc.channel_from_HRs(
        model.HRs, model.at, model.alat, model.volume, model.tau, 0.8, nk,
        fermi_energy=mu, temperature=temperature, relaxation_time=tau,
    )  # fmt: skip
    energies = np.array([0.1, 0.5, 2.0])
    kernel = cc.kernel_for_level(ch, 'none', None, 0.0, None)
    eps = cc.dielectric_function_crystal(ch, kernel, energies, 0.05, directions=(0,))['epsilon']
    K = nlo._k_grid(nk) * 2 * np.pi / model.alat
    d2 = np.zeros(len(K))
    eps_k = np.zeros(len(K))
    for i, _, R, h in hops:
        if i == 0:
            d = model.alat * np.asarray(R, dtype=float)
            eps_k += np.real(h * np.exp(1j * K @ d))
            d2 += np.real(-(d[0] ** 2) * h * np.exp(1j * K @ d))
    f = 1.0 / (1.0 + np.exp((eps_k - mu) / (8.617333262e-5 * temperature)))
    wp2 = 4.0 * np.pi / model.volume * 2.0 * np.mean(f * d2) / HA  # Ha^2
    gamma = 0.6582119569 / tau / HA
    w = energies / HA
    assert np.allclose(eps[:, 0, 0], 1.0 - wp2 / (w * (w + 1j * gamma)), rtol=5e-3)
    with pytest.raises(ValueError, match='finite frequency'):
        ch.polarization_of(np.zeros((ch.nk_local, 2, 2), dtype=complex), 0.0)


def test_metal_kernel_levels_and_second_order_are_refused():
    model = _models().random_hopping_model(seed=2, gap=False)
    ch = cc.channel_from_HRs(
        model.HRs, model.at, model.alat, model.volume, model.tau, 4.0, (4, 4, 4),
        fermi_energy=0.3, temperature=1000.0,
    )  # fmt: skip
    with pytest.raises(ValueError, match='long-range'):
        cc.coupled_optical_spectra(ch, ('none', 'hartree+alda+lrc'), dielectric_energies_ev=[1.0])
    with pytest.raises(NotImplementedError, match='metals'):
        cc.coupled_optical_spectra(ch, ('none',), shg_energies_ev=[0.5])
    with pytest.raises(ValueError, match='scissor'):
        cc.channel_from_HRs(
            model.HRs, model.at, model.alat, model.volume, model.tau, 4.0, (4, 4, 4),
            fermi_energy=0.3, temperature=1000.0, scissor_ev=0.5,
        )  # fmt: skip
