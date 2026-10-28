"""Non-linear optics kernels: sum rule, gauge invariance, symmetry selection rules, units."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import (
    bloch_terms_from_HRs,
    hamiltonian_derivatives_block,
    velocity_and_curvature_operators,
)
from PAOFLOW.response import nonlinear_optics as nlo

_spec = importlib.util.spec_from_file_location(
    '_nonlinear_models', Path(__file__).resolve().parent / '_nonlinear_models.py'
)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)
honeycomb_model = _models.honeycomb_model
random_hopping_model = _models.random_hopping_model
rice_mele_chain = _models.rice_mele_chain


def geometry_at(model, kpts, tol=1e-4):
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    H, dH, d2H = hamiltonian_derivatives_block(terms, kpts)
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    return nlo.band_geometry(E, U, v, w, tol)


def spectra(model, nk, energy, **kwargs):
    defaults = dict(components='all', eta=0.05, smearing='gauss', degeneracy_tol=1e-4, kblock=64)
    defaults.update(kwargs)
    return nlo.nonlinear_optical_spectra(
        model.HRs,
        model.at,
        model.alat,
        model.volume,
        model.tau,
        model.nelec,
        nk,
        energy,
        **defaults,
    )


def index(results, label):
    return results['components'].index(label)


def natural_scale(model, nk, key):
    """Prefactor x normalisation: the size of a response built from O(1) band quantities."""
    norm = 2.0 / (np.prod(nk) * model.volume)
    return {'shift': nlo.C_SHIFT, 'shiftm': nlo.C_SHIFT, 'injectc': nlo.C_INJECT,
            'injectm': nlo.C_INJECT, 'shg': nlo.C_SHG}[key] * norm  # fmt: skip


# --------------------------------------------------------------------------- geometry
def test_generalized_derivative_matches_covariant_finite_differences():
    """The sum-rule r^b_{nm;a} equals the k-derivative of r^b_nm in a parallel-transport gauge."""
    model = random_hopping_model(seed=5, time_reversal=False)
    k0 = np.array([0.17, -0.23, 0.29])
    geometry = geometry_at(model, k0[None])
    r0, rd0, U0 = geometry.position[0], geometry.position_derivative[0], geometry.eigenvectors[0]
    E0 = geometry.energies[0]
    assert np.all(np.abs(E0[1:] - E0[:-1]) > 0.1)  # no degeneracies
    UA0 = model.phases(k0).conj().T @ U0  # the engine's eigenvectors in the atomic gauge
    bg_cart = np.linalg.inv(model.at).T

    def r_at(k):
        HA, dHA, _ = model.atomic_gauge(k)
        E, U = np.linalg.eigh(HA)
        overlap = np.diag(UA0.conj().T @ U)
        U = U * np.conj(overlap / np.abs(overlap))[None, :]  # parallel transport from k0
        v = np.einsum('im,aij,jn->amn', U.conj(), dHA, U)
        omega = E[:, None] - E[None, :]
        inv = np.where(np.abs(omega) > 1e-4, 1.0 / np.where(np.abs(omega) > 1e-4, omega, 1.0), 0.0)
        return -1j * v * inv[None]

    for a in range(3):
        unit = np.zeros(3)
        unit[a] = 1.0
        rows = []
        for step in (2e-3, 1e-3):
            dk_cryst = (unit * step * model.alat / (2 * np.pi)) @ np.linalg.inv(bg_cart)
            rows.append((r_at(k0 + dk_cryst) - r_at(k0 - dk_cryst)) / (2 * step))
        richardson = (4.0 * rows[1] - rows[0]) / 3.0  # O(step^4)
        off = ~np.eye(model.nawf, dtype=bool)
        for b in range(3):
            assert np.allclose(rd0[a, b][off], richardson[b][off], rtol=1e-5, atol=1e-7)
    assert np.allclose(np.diagonal(r0, axis1=1, axis2=2), 0.0)


def test_gauge_invariance_under_orbital_unitaries():
    model = random_hopping_model(seed=6, time_reversal=False)
    rng = np.random.default_rng(1)
    energy = np.linspace(0.5, 8.0, 31)
    base = spectra(model, (4, 4, 4), energy, components=['xyz', 'yyy', 'zxx'])

    # a unitary mixing orbitals that share a centre (0,1) and (2,3), and random phases
    def block(size):
        m = rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))
        q, _ = np.linalg.qr(m)
        return q

    P = np.zeros((4, 4), dtype=complex)
    P[:2, :2] = block(2)
    P[2:, 2:] = block(2)
    HRs = np.einsum('ij,jkabcs,lk->ilabcs', P, model.HRs, P.conj())
    rotated = nlo.nonlinear_optical_spectra(
        HRs, model.at, model.alat, model.volume, model.tau, model.nelec, (4, 4, 4), energy,
        components=['xyz', 'yyy', 'zxx'], eta=0.05, kblock=64,
    )  # fmt: skip
    for key in ('shift', 'shiftm', 'injectc', 'injectm', 'shg'):
        scale = np.max(np.abs(base[key])) + 1e-30
        assert np.allclose(base[key], rotated[key], atol=1e-10 * scale)


def test_haldane_chern_number_from_two_band_berry_curvature():
    """Omega_v = -2 Im sum_c r^x_vc r^y_cv integrates to the Chern number of the Haldane model."""
    model = honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2)
    n = 48
    axes = [np.arange(n) / n, np.arange(n) / n, [0.0]]
    kpts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    geometry = geometry_at(model, kpts)
    r = geometry.position
    omega_v = -2.0 * np.imag(r[:, 0, 0, 1] * r[:, 1, 1, 0])  # valence band 0, conduction 1
    area_cell = model.alat**2 * abs(np.linalg.det(model.at[:2, :2]))
    chern = np.sum(omega_v) * (2 * np.pi) ** 2 / area_cell / kpts.shape[0] / (2 * np.pi)
    assert abs(abs(chern) - 1.0) < 2e-2
    trivial = honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=0.0)
    geometry = geometry_at(trivial, kpts)
    r = geometry.position
    omega_v = -2.0 * np.imag(r[:, 0, 0, 1] * r[:, 1, 1, 0])
    assert abs(np.sum(omega_v) * (2 * np.pi) ** 2 / area_cell / kpts.shape[0] / (2 * np.pi)) < 2e-2


# --------------------------------------------------------------------------- symmetry
def test_inversion_symmetric_chain_gives_no_response_and_odd_parity():
    energy = np.linspace(0.2, 5.0, 25)
    ssh = rice_mele_chain(t=1.0, dt=0.3, delta=0.0)
    res = spectra(ssh, (24, 1, 1), energy, components=['xxx'])
    for key in ('shift', 'shiftm', 'injectc', 'injectm', 'shg'):
        assert np.max(np.abs(res[key])) < 1e-9 * natural_scale(ssh, (24, 1, 1), key)
    plus = spectra(rice_mele_chain(1.0, 0.3, 0.4), (24, 1, 1), energy, components=['xxx'])
    # inversion about the strong bond: (delta, dt) -> (-delta, dt) flips every response
    minus = spectra(rice_mele_chain(1.0, 0.3, -0.4), (24, 1, 1), energy, components=['xxx'])
    # translation by half a cell: (delta, dt) -> (-delta, -dt) leaves them unchanged
    moved = spectra(rice_mele_chain(1.0, -0.3, -0.4), (24, 1, 1), energy, components=['xxx'])
    assert np.max(np.abs(plus['shift'])) > 1e-3 * natural_scale(ssh, (24, 1, 1), 'shift')
    assert np.allclose(plus['shift'], -minus['shift'], atol=1e-10 * np.max(np.abs(plus['shift'])))
    assert np.allclose(plus['shg'], -minus['shg'], atol=1e-10 * np.max(np.abs(plus['shg'])))
    assert np.allclose(plus['shift'], moved['shift'], atol=1e-10 * np.max(np.abs(plus['shift'])))
    assert np.allclose(plus['shg'], moved['shg'], atol=1e-10 * np.max(np.abs(plus['shg'])))
    # time-reversal symmetric: the T-odd responses vanish
    for key in ('shiftm', 'injectm'):
        assert np.max(np.abs(plus[key])) < 1e-9 * natural_scale(ssh, (24, 1, 1), key)


def test_gapped_graphene_d3h_selection_rules():
    model = honeycomb_model(delta=0.6, t=-1.0)
    energy = np.linspace(0.3, 4.0, 38)
    res = spectra(model, (18, 18, 1), energy, components='all', eta=0.08)
    sig = res['shift']
    yyy, yxx, xxy, xyx, xxx = (sig[index(res, c)] for c in ('yyy', 'yxx', 'xxy', 'xyx', 'xxx'))
    scale = np.max(np.abs(yyy))
    assert scale > 1e-8
    assert np.allclose(yyy, -yxx, atol=1e-8 * scale)
    assert np.allclose(yyy, -xxy, atol=1e-8 * scale)
    assert np.allclose(yyy, -xyx, atol=1e-8 * scale)
    assert np.max(np.abs(xxx)) < 1e-8 * scale
    for label in res['components']:
        if 'z' in label:
            assert np.max(np.abs(sig[index(res, label)])) < 1e-12 * scale
    chi = res['shg']
    cscale = np.max(np.abs(chi[index(res, 'yyy')]))
    assert np.allclose(chi[index(res, 'yyy')], -chi[index(res, 'yxx')], atol=1e-8 * cscale)
    assert np.allclose(chi[index(res, 'yyy')], -chi[index(res, 'xxy')], atol=1e-8 * cscale)
    assert np.max(np.abs(chi[index(res, 'xxx')])) < 1e-8 * cscale
    # time reversal + mirror: no circular injection, no magnetic responses
    for key in ('injectc', 'injectm', 'shiftm'):
        assert np.max(np.abs(res[key])) < 1e-9 * natural_scale(model, (18, 18, 1), key)


def test_haldane_breaks_time_reversal():
    """T-odd responses appear with the Haldane flux (strain removes the C3 axis that forbids MSC)."""
    energy = np.linspace(0.3, 4.0, 20)
    nk = (12, 12, 1)
    magnetic = spectra(
        honeycomb_model(0.6, -1.0, 0.15, np.pi / 2, strain=0.2),
        nk,
        energy,
        components=['yyy', 'xxy'],
    )
    model = honeycomb_model(0.6, -1.0, 0.15, 0.0, strain=0.2)
    assert np.max(np.abs(magnetic['shiftm'])) > 1e-4 * natural_scale(model, nk, 'shiftm')
    assert np.max(np.abs(magnetic['injectm'])) > 1e-4 * natural_scale(model, nk, 'injectm')
    plain = spectra(model, nk, energy, components=['yyy', 'xxy'])
    assert np.max(np.abs(plain['shiftm'])) < 1e-9 * natural_scale(model, nk, 'shiftm')
    assert np.max(np.abs(plain['injectm'])) < 1e-9 * natural_scale(model, nk, 'injectm')
    assert np.max(np.abs(plain['shift'])) > 1e-4 * natural_scale(model, nk, 'shift')


# --------------------------------------------------------------------------- numerics
def test_scissor_shifts_the_delta_spectra_rigidly():
    model = honeycomb_model(delta=0.6, t=-1.0)
    energy = np.linspace(0.0, 5.0, 101)  # step 0.05 = the scissor
    plain = spectra(model, (12, 12, 1), energy, components=['yyy'], eta=0.1)
    shifted = spectra(model, (12, 12, 1), energy, components=['yyy'], eta=0.1, scissor=0.5)
    assert shifted['gap_ev'] == plain['gap_ev']
    assert np.allclose(shifted['shift'][0, 10:], plain['shift'][0, :-10], rtol=1e-10, atol=1e-14)
    assert np.allclose(shifted['injectc'][0, 10:], plain['injectc'][0, :-10], rtol=1e-10, atol=1e-6)


def direct_shg(model, nk, energy, components, eta, tol=1e-4):
    """Slow reference: Eqs. (C2)-(C3) evaluated with explicit denominators on the grid."""
    axes = [np.arange(n) / n for n in nk]
    kpts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    g = geometry_at(model, kpts, tol)
    nocc = model.nelec // 2
    f = np.zeros(model.nawf)
    f[:nocc] = 1.0
    w = energy + 1j * eta  # resonance at w_mn - w - i eta
    w2 = 2.0 * energy + 1j * eta  # resonance at w_mn - 2w - i eta (one common i eta)
    out = np.zeros((len(components), energy.size), dtype=complex)
    n = model.nawf
    for ik in range(kpts.shape[0]):
        E, r, rd, dv = (
            g.energies[ik],
            g.position[ik],
            g.position_derivative[ik],
            g.delta_velocity[ik],
        )
        for ic, (a, b, c) in enumerate(components):
            for i in range(n):
                for j in range(n):
                    if i == j or abs(E[i] - E[j]) < tol:
                        continue
                    wmn = E[j] - E[i]  # omega_mn with n = i, m = j
                    fnm = f[i] - f[j]
                    # two-band terms (n = i, m = j)
                    t1 = 2 * r[a, i, j] * (rd[c, b, j, i] + rd[b, c, j, i]) / (wmn * (wmn - w2))
                    t2 = (rd[c, a, i, j] * r[b, j, i] + rd[b, a, i, j] * r[c, j, i]) / (
                        wmn * (wmn - w)
                    )
                    t3 = (
                        r[a, i, j]
                        * (r[b, j, i] * dv[c, j, i] + r[c, j, i] * dv[b, j, i])
                        / wmn**2
                        * (1 / (wmn - w) - 4 / (wmn - w2))
                    )
                    t4 = -(r[b, j, i] * rd[a, c, i, j] + r[c, j, i] * rd[a, b, i, j]) / (
                        2 * wmn * (wmn - w)
                    )
                    t5 = -dv[a, j, i] * np.real(r[b, i, j] * r[c, j, i]) / (2 * wmn**2 * (wmn - w))
                    out[ic] += 1j * fnm * (t1 + t2 + t3 + t4 + t5)
                    for l in range(n):
                        if l == i or l == j:
                            continue
                        den = (E[l] - E[i]) - (E[j] - E[l])
                        if abs(den) < tol:
                            continue
                        T = r[a, i, j] * (r[b, j, l] * r[c, l, i] + r[c, j, l] * r[b, l, i]) / den
                        out[ic] += T * (
                            2 * fnm / (wmn - w2)
                            + (f[l] - f[i]) / ((E[l] - E[i]) - w)
                            + (f[j] - f[l]) / ((E[j] - E[l]) - w)
                        )
    return nlo.C_SHG * 2.0 / (kpts.shape[0] * model.volume) * out


def test_shg_histograms_match_direct_evaluation():
    model = random_hopping_model(seed=8, time_reversal=False)
    energy = np.linspace(0.1, 6.0, 40)
    components = [(0, 1, 2), (1, 1, 1), (2, 0, 0)]
    reference = direct_shg(model, (3, 3, 3), energy, components, eta=0.15)
    fast = spectra(
        model, (3, 3, 3), energy, components=components, eta=0.15, shg_bin=0.15 / 100, want=('shg',)
    )
    scale = np.max(np.abs(reference))
    assert scale > 1e-3
    assert np.allclose(fast['shg'], reference, atol=2e-3 * scale)


def test_kleinman_symmetry_of_the_static_limit():
    """chi(0; 0, 0) is real and symmetric under all index permutations (strained honeycomb, T + M_x)."""
    model = honeycomb_model(delta=0.8, t=-1.0, strain=0.3)
    energy = np.array([0.0])
    comps = ['xxy', 'xyx', 'yxx', 'yyy', 'xxx']
    res = spectra(model, (48, 48, 1), energy, components=comps, eta=0.01, want=('shg',))
    chi = res['shg'][:, 0]
    scale = np.max(np.abs(chi))
    assert scale > 10.0  # pm^2/V-scale numbers per bohr of slab: the sheet response is large
    assert np.max(np.abs(chi.imag)) < 1e-4 * scale  # purely real static susceptibility
    assert abs(chi[0] - chi[1]) < 1e-10 * scale  # intrinsic permutation symmetry (exact)
    assert abs(chi[0] - chi[2]) < 2e-3 * scale  # Kleinman: xxy = yxx up to k-convergence
    assert abs(chi[4]) > 0.1 * scale  # strain breaks the threefold axis: xxx no longer vanishes


def test_prefactors_components_and_writer(tmp_path):
    assert nlo.C_SHIFT == pytest.approx(7.647e-4, rel=1e-3)  # A/V^2 = pi e^2 / hbar
    assert nlo.C_INJECT == pytest.approx(2.3236e12, rel=1e-3)  # A V^-2 s^-1 = 2 pi e^3 / hbar^2
    assert nlo.C_SHG == pytest.approx(9047.6, rel=1e-3)  # pm/V = 1e12 e / (2 eps0)
    assert nlo.parse_components('diag') == [(0, 0, 0), (1, 1, 1), (2, 2, 2)]
    assert len(nlo.parse_components('all')) == 27
    assert nlo.parse_components(['xyz', (1, 0, 0)]) == [(0, 1, 2), (1, 0, 0)]
    with pytest.raises(ValueError):
        nlo.parse_components(['xq'])
    assert nlo.component_label((2, 1, 0)) == 'zyx'

    class DC:
        def __init__(self):
            self.files = {}

        def write_file_row_col(self, name, col1, col2):
            self.files[name] = (np.asarray(col1), np.asarray(col2))

    energy = np.linspace(0.5, 3.0, 6)
    results = {
        'energy_ev': energy, 'components': ['xyz', 'yyy'],
        'shift': np.ones((2, 6)) * 1e-6, 'shiftm': np.zeros((2, 6)),
        'injectc': np.ones((2, 6)), 'injectm': np.zeros((2, 6)),
        'shg': (1.0 + 2.0j) * np.ones((2, 6)),
    }  # fmt: skip
    dc = DC()
    names = nlo.write_nonlinear_outputs(dc, results)
    assert len(names) == 14 and 'shift_xyz.dat' in names and 'shga_yyy.dat' in names
    assert np.allclose(dc.files['shift_xyz.dat'][1], 1.0)  # uA/V^2
    assert np.allclose(dc.files['shga_yyy.dat'][1], np.sqrt(5.0))
    dc2 = DC()
    nlo.write_nonlinear_outputs(dc2, results, slab_height_bohr=10.0, suffix='_0')
    assert np.allclose(dc2.files['shift_xyz_0.dat'][1], 10.0 * 0.0529177)  # nm uA/V^2
    assert np.allclose(dc2.files['shgr_yyy_0.dat'][1], 10.0 * 52.9177)  # pm^2/V


def test_occupations_and_delta_kernels():
    e = np.array([[-1.0, 0.0, 1.0]])
    assert np.allclose(nlo.occupations(e, 2), [[1, 1, 0]])
    assert np.allclose(nlo.occupations(e, 2, fermi_energy=0.5), [[1, 1, 0]])
    fd = nlo.occupations(e, 2, temperature_k=300.0, fermi_energy=0.0)
    assert fd[0, 1] == pytest.approx(0.5) and fd[0, 0] > 0.99 and fd[0, 2] < 0.01
    with pytest.raises(ValueError):
        nlo.occupations(e, 2, temperature_k=300.0)
    x = np.linspace(-50, 50, 200001)
    for kind in ('gauss', 'lorentz'):
        assert np.trapezoid(nlo.delta_kernel(x, 0.1, kind), x) == pytest.approx(1.0, abs=2e-3)
    with pytest.raises(ValueError):
        nlo.delta_kernel(x, 0.1, 'box')
