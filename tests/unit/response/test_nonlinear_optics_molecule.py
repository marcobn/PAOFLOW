"""Molecules at Gamma (nk = 1, 1 x 1 x 1 HRs): SHG against finite-field beta, shift against the
dipole-shift formula, vanishing injection, wrapping invariance, selection rules, band-slab parity."""

import numpy as np
import pytest

from PAOFLOW.response import nonlinear_optics as nlo

pytestmark = pytest.mark.unit

HARTREE_EV = 27.211386245988
AU_BETA_PER_BOHR3_PMV = 8.8541878128e-12 * (0.529177210903e-10) ** 3 * 1e-12 / 3.2063613e-53


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


def _box(H, tau, L):
    n = H.shape[0]
    HRs = np.zeros((n, n, 1, 1, 1, 1), dtype=complex)
    HRs[:, :, 0, 0, 0, 0] = H
    L = np.asarray(L, dtype=float)
    return HRs, np.diag(L / L[0]), float(L[0]), float(np.prod(L)), tau / L[None, :]


def _spectra(H, tau, L, energy, nelec, **kw):
    HRs, at, alat, vol, pos = _box(H, tau, L)
    opts = dict(components='all', eta=0.02, spin_factor=2.0, degeneracy_tol=1e-6, shg_bin=0.02 / 50)
    opts.update(kw)
    return nlo.nonlinear_optical_spectra(
        HRs, at, alat, vol, pos, float(nelec), (1, 1, 1), energy, **opts
    )


def _finite_field_beta(H, tau, nocc, h=1.0e-5):
    """beta^T_abc = d^2 mu_a / dF_b dF_c (a.u.), mu = -2 sum_occ <x>, H1 = + x . F."""
    Hh = H / HARTREE_EV
    X = [np.diag(tau[:, a]) for a in range(3)]

    def mu(F):
        _, U = np.linalg.eigh(Hh + sum(F[a] * X[a] for a in range(3)))
        occ = U[:, :nocc]
        return np.array(
            [-2.0 * np.real(np.einsum('in,ij,jn->', occ.conj(), X[a], occ)) for a in range(3)]
        )

    beta = np.zeros((3, 3, 3))
    for b in range(3):
        for c in range(3):
            if b == c:
                vals = []
                for s in (-2, -1, 0, 1, 2):
                    F = np.zeros(3)
                    F[b] = s * h
                    vals.append(mu(F))
                beta[:, b, b] = (
                    -vals[4] + 16 * vals[3] - 30 * vals[2] + 16 * vals[1] - vals[0]
                ) / (12 * h * h)
            else:
                acc = np.zeros(3)
                for sb, sc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    F = np.zeros(3)
                    F[b], F[c] = sb * h, sc * h
                    acc += sb * sc * mu(F)
                beta[:, b, c] = acc / (4 * h * h)
    return beta


def test_static_shg_is_minus_half_the_finite_field_hyperpolarizability():
    """beta^T (Taylor, electrons) = -2 eps0 Omega chi(0): the B/T factor and the e^3 sign of the code."""
    H, tau = _chain()
    n = H.shape[0]
    L = (80.0, 50.0, 40.0)
    res = _spectra(H, tau, L, np.array([0.0]), n, want=('shg',))
    ci = {c: i for i, c in enumerate(res['components'])}
    beta_ff = _finite_field_beta(H, tau, n // 2)
    vol = float(np.prod(L))
    for comp in ('xxx', 'xxy', 'yxx', 'xyy', 'yyy', 'zxx'):
        a, b, c = ['xyz'.index(s) for s in comp]
        code = AU_BETA_PER_BOHR3_PMV * vol * res['shg'][ci[comp], 0]
        assert abs(code.imag) < 1e-6 * abs(code)
        assert abs(code.real / beta_ff[a, b, c] + 0.5) < 0.02


def test_shift_current_is_the_dipole_shift_formula_and_injection_vanishes():
    H, tau = _chain()
    n = H.shape[0]
    nocc = n // 2
    L = (80.0, 50.0, 40.0)
    energy = np.linspace(0.0, 3.0, 61)
    eta = 0.02
    res = _spectra(
        H, tau, L, energy, n, want=('shift', 'injection'), components=['xxx', 'xyy', 'yxx', 'xyz']
    )
    ev, U = np.linalg.eigh(H)
    X = np.stack([U.conj().T @ np.diag(tau[:, a]) @ U for a in range(3)])
    norm = 2.0 / float(np.prod(L))
    for ic, comp in enumerate(['xxx', 'xyy', 'yxx', 'xyz']):
        a, b, c = ['xyz'.index(s) for s in comp]
        ref = np.zeros(energy.size)
        for m in range(nocc):
            for k in range(nocc, n):
                kern = nlo.delta_kernel(ev[k] - ev[m] - energy, eta, 'gauss')
                ref += (
                    2.0
                    * np.real(X[b, k, m] * np.conj(X[c, k, m]))
                    * (X[a, k, k] - X[a, m, m]).real
                    * kern
                )
        ref *= nlo.C_SHIFT * norm
        assert np.allclose(res['shift'][ic], ref, atol=1e-8 * np.abs(ref).max())
    # no band velocities in a finite system: the injection currents are rounding noise
    scale = nlo.C_INJECT * norm * np.abs(X).max() ** 3 / eta
    assert np.abs(res['injectc']).max() < 1e-10 * scale
    assert np.abs(res['injectm']).max() < 1e-10 * scale
    assert np.abs(res['shiftm']).max() < 1e-10 * np.abs(res['shift']).max()


def test_outputs_are_invariant_under_wrapping_into_the_box():
    H, tau = _chain()
    n = H.shape[0]
    L = np.array([80.0, 50.0, 40.0])
    energy = np.linspace(0.0, 2.0, 21)
    ref = _spectra(H, tau, L, energy, n, components=['xxx', 'xyy'])
    wrapped = np.mod(tau + np.array([0.49, 0.48, 0.47]) * L, L) - 0.5 * L  # straddles every face
    res = _spectra(H, wrapped, L, energy, n, components=['xxx', 'xyy'])
    for key in ('shift', 'shg'):
        assert np.allclose(res[key], ref[key], rtol=1e-12, atol=1e-12 * np.abs(ref[key]).max())
    # Omega chi is independent of the box
    big = _spectra(H, tau, 1.5 * L, energy[:1], n, components=['xxx'], want=('shg',))
    assert np.isclose(
        np.prod(1.5 * L) * big['shg'][0, 0], np.prod(L) * ref['shg'][0, 0], rtol=1e-10
    )


def test_selection_rules_at_gamma():
    L = (80.0, 50.0, 40.0)
    # centrosymmetric chain: odd site count, alternating on-site energies, closed shell
    H, tau = _chain(nsite=11, ramp=0.0, bend=0.0, noise=0.0)
    np.fill_diagonal(H, 0.4 * (np.arange(11) % 2))
    assert np.allclose(H, H[::-1, ::-1])
    res = _spectra(H, tau, L, np.linspace(0.0, 2.0, 21), 10)
    assert np.abs(res['shg']).max() < 1e-8
    assert np.abs(res['shift']).max() < 1e-14
    # D3h: triangle + centre, closed shell of 6 electrons
    ang = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
    taut = np.array([[2.6 * np.cos(t), 2.6 * np.sin(t), 0.0] for t in ang] + [[0.0, 0.0, 0.0]])
    Ht = np.full((4, 4), -1.0)
    Ht[0:3, 0:3] = -0.8
    np.fill_diagonal(Ht, [0.0, 0.0, 0.0, 1.5])
    res = _spectra(Ht, taut, L, np.array([0.0, 0.4]), 6, want=('shg',))
    ci = {c: i for i, c in enumerate(res['components'])}
    chi = {c: res['shg'][ci[c], 0] for c in ('xxx', 'xyy', 'yxy', 'yyx', 'yyy', 'xxy', 'zzz')}
    scale = abs(chi['xxx'])
    assert scale > 1e-3
    for c in ('xyy', 'yxy', 'yyx'):
        assert abs(chi[c] + chi['xxx']) < 1e-10 * scale
    for c in ('yyy', 'xxy', 'zzz'):
        assert abs(chi[c]) < 1e-10 * scale


def test_band_slabs_and_work_split_reproduce_the_shg_histograms():
    H, tau = _chain(nsite=12)
    n = H.shape[0]
    L = (80.0, 50.0, 40.0)
    energy = np.linspace(0.0, 3.0, 31)
    ref = _spectra(H, tau, L, energy, n, want=('shg',), components=['xxx', 'xyy', 'xyz'])
    slab = _spectra(
        H, tau, L, energy, n, want=('shg',), components=['xxx', 'xyy', 'xyz'], shg_band_slab=1
    )
    assert np.allclose(slab['shg'], ref['shg'], rtol=1e-12, atol=1e-12 * np.abs(ref['shg']).max())
    slab5 = _spectra(
        H, tau, L, energy, n, want=('shg',), components=['xxx', 'xyy', 'xyz'], shg_band_slab=5
    )
    assert np.allclose(slab5['shg'], ref['shg'], rtol=1e-12, atol=1e-12 * np.abs(ref['shg']).max())
    # the rank split: the sum over the work units of three "ranks" equals the single-rank result
    HRs, at, alat, vol, pos = _box(H, tau, L)
    kb = nlo._k_grid((1, 1, 1))
    from PAOFLOW.hamiltonian.bloch_derivatives import (
        bloch_terms_from_HRs,
        hamiltonian_derivatives_block,
        velocity_and_curvature_operators,
    )

    terms = bloch_terms_from_HRs(HRs, at, alat, pos)
    Hk, dH, d2H = hamiltonian_derivatives_block(terms, kb)
    v_orb, w_orb = velocity_and_curvature_operators(terms, Hk, dH, d2H)
    E, U = np.linalg.eigh(Hk)
    geometry = nlo.band_geometry(E, U, v_orb, w_orb, 1e-6)
    f = nlo.occupations(E, n // 2)
    edges = np.linspace(-1.0, 8.0, 901)
    comps = [(0, 0, 0), (0, 1, 1)]
    single = {'hist1': np.zeros((2, 900), complex), 'hist2': np.zeros((2, 900), complex)}
    nlo.accumulate_shg_histograms(geometry, f, E, edges, comps, single, 1e-6, band_slab=3)
    total = {'hist1': np.zeros((2, 900), complex), 'hist2': np.zeros((2, 900), complex)}
    for rank in range(3):
        nlo.accumulate_shg_histograms(
            geometry, f, E, edges, comps, total, 1e-6, band_slab=3, work=(rank, 3)
        )
    for key in single:
        assert np.allclose(
            total[key], single[key], rtol=1e-12, atol=1e-14 * np.abs(single[key]).max()
        )
    out1 = {k: np.zeros((2, energy.size)) for k in ('shift', 'shiftm', 'injectc', 'injectm')}
    out3 = {k: np.zeros((2, energy.size)) for k in ('shift', 'shiftm', 'injectc', 'injectm')}
    widths = np.full(geometry.omega.shape, 0.02)
    nlo.accumulate_shift_and_injection(
        geometry, f, geometry.omega, energy, widths, comps, 'gauss', out1
    )
    for rank in range(3):
        nlo.accumulate_shift_and_injection(
            geometry, f, geometry.omega, energy, widths, comps, 'gauss', out3, work=(rank, 3)
        )
    assert np.allclose(
        out3['shift'], out1['shift'], rtol=1e-12, atol=1e-14 * np.abs(out1['shift']).max()
    )
