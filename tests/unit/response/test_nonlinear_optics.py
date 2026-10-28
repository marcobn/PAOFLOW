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


# --------------------------------------------------------------------------- degenerate clusters
def supercell_x(model):
    """The model in a cell doubled along x: ``(HRs, at, tau, nelec, volume)``.

    Band folding makes every band of a time-reversal-symmetric model an exact doublet on the
    zone-boundary plane ``k_x = 1/2`` of the supercell (primitive ``k'_x = +-1/4``), the two
    members having opposite x velocities: the intra-doublet velocity blocks are not scalar and
    the basis returned by ``eigh`` inside a doublet is arbitrary."""
    n = model.nawf
    hops = []
    for i, j, R, h in model.hops:
        for s in range(2):
            x = R[0] + s
            hops.append((i + n * s, j + n * (x % 2), (x // 2, R[1], R[2]), h))
    HRs = _models.hopping_list_to_HRs(hops, 2 * n, (4, 4, 4))
    at = model.at.copy()
    at[0] *= 2.0
    tau = np.concatenate([model.tau * [0.5, 1, 1], model.tau * [0.5, 1, 1] + [0.5, 0, 0]])
    return HRs, at, tau, 2 * model.nelec, 2.0 * model.volume


def test_degenerate_clusters_are_transitive():
    e = np.array([[0.0, 0.5e-4, 1.0e-4, 1.5e-4, 1.0], [0.0, 2.0e-4, 1.0, 1.0 + 0.9e-4, 2.0]])
    assert nlo.degenerate_clusters(e, 1.0e-4).tolist() == [[0, 0, 0, 0, 1], [0, 1, 2, 2, 3]]


def test_kernels_are_invariant_under_rotations_inside_exact_doublets():
    """Shift, injection and SHG weights of a k-point do not depend on the basis chosen inside a
    degenerate cluster (the U(N) covariance that the diagonal Delta^a_nm form lacks)."""
    model = random_hopping_model(seed=7, time_reversal=True)
    HRs, at, tau, nelec, _ = supercell_x(model)
    terms = bloch_terms_from_HRs(HRs, at, model.alat, tau)
    kpts = np.array(
        [[0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
    )  # folded plane, k_y, k_z in {0, 1/2}: exact doublets
    H, dH, d2H = hamiltonian_derivatives_block(terms, kpts)
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    tol = 1e-6
    cluster = nlo.degenerate_clusters(E, tol)
    assert np.all(np.bincount(cluster[0]) == 2) and np.all(np.bincount(cluster[1]) == 2)
    rng = np.random.default_rng(3)
    V = np.zeros_like(U)
    for ik in range(kpts.shape[0]):
        for c in np.unique(cluster[ik]):
            idx = np.nonzero(cluster[ik] == c)[0]
            q, _ = np.linalg.qr(rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2)))
            V[ik][np.ix_(idx, idx)] = q
    geometries = [nlo.band_geometry(E, U, v, w, tol), nlo.band_geometry(E, U @ V, v, w, tol)]
    # the intra-doublet velocity blocks are not proportional to the identity: non-trivial test
    idx = np.nonzero(cluster[0] == cluster[0][0])[0]
    block = geometries[0].velocity_intra[0, 0][np.ix_(idx, idx)]
    assert np.abs(block - 0.5 * np.trace(block) * np.eye(2)).max() > 1e-3
    nocc = nelec // 2
    comps = nlo.parse_components(['xyz', 'yzx', 'xxy', 'zzz'])
    energy = np.linspace(0.2, 8.0, 40)
    edges = -20.0 + 0.02 * np.arange(2001)
    outs = []
    for geo in geometries:
        f = nlo.occupations(E, nocc)
        out = {key: np.zeros((4, energy.size)) for key in ('shift', 'shiftm', 'injectc', 'injectm')}
        nlo.accumulate_shift_and_injection(
            geo, f, geo.omega, energy, np.full(geo.omega.shape, 0.1), comps, 'gauss', out
        )
        hist = {key: np.zeros((4, 2000), dtype=complex) for key in ('hist1', 'hist2')}
        nlo.accumulate_shg_histograms(geo, f, E, edges, comps, hist, tol)
        outs.append({**out, **hist})
    ref = max(np.max(np.abs(outs[0]['shift'])), np.max(np.abs(outs[0]['hist1'])))
    for key, a in outs[0].items():
        b = outs[1][key]
        assert np.allclose(a, b, atol=1e-10 * np.max(np.abs(a)) + 1e-13 * ref), key
    # the diagonal velocities themselves do depend on the basis
    assert not np.allclose(geometries[0].delta_velocity, geometries[1].delta_velocity, atol=1e-6)


def test_supercell_folding_is_exact_with_degenerate_doublets():
    """The x-supercell on a 4^3 grid equals the primitive cell on an (8, 4, 4) grid: the exact
    doublets of the folded plane (arbitrary eigensolver bases) must give the same sums as the
    non-degenerate primitive k-points they come from. Also checks two k-block sizes."""
    model = random_hopping_model(seed=8, time_reversal=True)
    HRs, at, tau, nelec, volume = supercell_x(model)
    energy = np.linspace(0.5, 8.0, 31)
    kwargs = dict(components=['xyz', 'zxx', 'yyy'], eta=0.05, degeneracy_tol=1e-6)
    primitive = spectra(model, (8, 4, 4), energy, **kwargs)
    folded = {
        kb: nlo.nonlinear_optical_spectra(
            HRs, at, model.alat, volume, tau, nelec, (4, 4, 4), energy, kblock=kb, **kwargs
        )
        for kb in (64, 1)
    }
    for key in ('shift', 'shiftm', 'injectc', 'injectm', 'shg'):
        scale = max(np.max(np.abs(primitive[key])), 1e-6 * natural_scale(model, (8, 4, 4), key))
        assert np.allclose(primitive[key], folded[64][key], atol=1e-8 * scale), key
        assert np.allclose(folded[64][key], folded[1][key], atol=1e-9 * scale), key


# --------------------------------------------------------------------------- position dipoles
def synthetic_dipoles(model, seed=11):
    """A smooth Hermitian periodic dipole field d(K) and its analytic Cartesian gradient."""
    rng = np.random.default_rng(seed)
    n = model.nawf

    def herm(scale):
        m = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        return scale * 0.5 * (m + m.conj().T)

    d0, d1, d2 = herm(0.3), herm(0.2), herm(0.15)
    at = model.at

    def dipoles(K):
        K = np.asarray(K, dtype=float)
        c1 = np.cos(2 * np.pi * K[:, 0])[:, None, None]
        s2 = np.sin(2 * np.pi * K[:, 1])[:, None, None]
        d = d0[None] + d1[None] * c1 + d2[None] * s2
        return np.repeat(d[:, None], 3, axis=1) * np.array([1.0, 0.7, -0.4])[None, :, None, None]

    def gradient(K):
        K = np.asarray(K, dtype=float)
        s1 = np.sin(2 * np.pi * K[:, 0])[:, None, None]
        c2 = np.cos(2 * np.pi * K[:, 1])[:, None, None]
        # d K_j / d k_a = (alat / 2 pi) at[j, a]
        g = np.empty((K.shape[0], 3, 3, n, n), dtype=complex)
        for a in range(3):
            dK1, dK2 = model.alat / (2 * np.pi) * at[0, a], model.alat / (2 * np.pi) * at[1, a]
            dd = -2 * np.pi * dK1 * d1[None] * s1 + 2 * np.pi * dK2 * d2[None] * c2
            g[:, a] = dd[:, None] * np.array([1.0, 0.7, -0.4])[None, :, None, None]
        return g

    return dipoles, gradient


def test_dipole_gradient_by_differences_matches_the_analytic_one():
    model = random_hopping_model(seed=5, time_reversal=False)
    dipoles, gradient = synthetic_dipoles(model)
    K = np.array([[0.17, -0.23, 0.29], [0.4, 0.1, 0.05]])
    fd = nlo.dipole_gradient_by_differences(dipoles, K, model.at, model.alat, step=1e-4)
    assert np.allclose(fd, gradient(K), atol=1e-7)


def test_generalized_derivative_with_dipoles_matches_covariant_finite_differences():
    """r_{;a} with position dipoles equals the gauge-fixed k-derivative of r = r^TB + D minus
    the shift i (D^a_nn - D^a_mm) r^b_nm of the Berry connection by the diagonal dipoles."""
    model = random_hopping_model(seed=5, time_reversal=False)
    dipoles, gradient = synthetic_dipoles(model)
    k0 = np.array([0.17, -0.23, 0.29])
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    H, dH, d2H = hamiltonian_derivatives_block(terms, k0[None])
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    # the engine takes the atomic-gauge derivative in the lattice basis: d_a d + i (tau_j - tau_i)_a d
    grad = (
        gradient(k0[None]) + 1j * terms.separation_bohr[None, :, None] * dipoles(k0[None])[:, None]
    )
    geometry = nlo.band_geometry(E, U, v, w, 1e-4, dipoles(k0[None]), grad)
    r0, rd0, U0 = geometry.position[0], geometry.position_derivative[0], geometry.eigenvectors[0]
    assert np.all(np.abs(E[0, 1:] - E[0, :-1]) > 0.1)
    UA0 = model.phases(k0).conj().T @ U0
    bg_cart = np.linalg.inv(model.at).T
    D0 = np.einsum('im,aij,jn->amn', U0.conj(), dipoles(k0[None])[0], U0)

    def r_at(k):
        HA, dHA, _ = model.atomic_gauge(k)
        E, U = np.linalg.eigh(HA)
        overlap = np.diag(UA0.conj().T @ U)
        U = U * np.conj(overlap / np.abs(overlap))[None, :]  # parallel transport from k0
        v = np.einsum('im,aij,jn->amn', U.conj(), dHA, U)
        omega = E[:, None] - E[None, :]
        inv = np.where(np.abs(omega) > 1e-4, 1.0 / np.where(np.abs(omega) > 1e-4, omega, 1.0), 0.0)
        # the lattice-gauge dipoles rotated to the atomic gauge: d^A = P^+ d P
        P = model.phases(k)
        dA = np.einsum('ij,ajk,kl->ail', P.conj().T, dipoles(k[None])[0], P)
        D = np.einsum('im,aij,jn->amn', U.conj(), dA, U)
        return -1j * v * inv[None] + D * (np.abs(omega) > 1e-4)[None]

    off = ~np.eye(model.nawf, dtype=bool)
    for a in range(3):
        unit = np.zeros(3)
        unit[a] = 1.0
        rows = []
        for step in (2e-3, 1e-3):
            dk_cryst = (unit * step * model.alat / (2 * np.pi)) @ np.linalg.inv(bg_cart)
            rows.append((r_at(k0 + dk_cryst) - r_at(k0 - dk_cryst)) / (2 * step))
        richardson = (4.0 * rows[1] - rows[0]) / 3.0
        shift = 1j * (np.diagonal(D0[a])[:, None] - np.diagonal(D0[a])[None, :])
        for b in range(3):
            expected = richardson[b] - shift * r0[b]
            assert np.allclose(rd0[a, b][off], expected[off], rtol=1e-5, atol=1e-7), (a, b)


def test_dipole_kernels_are_invariant_under_rotations_inside_exact_doublets():
    """The dipole terms of r and r_{;a} keep the U(N) covariance inside degenerate clusters."""
    model = random_hopping_model(seed=7, time_reversal=True)
    HRs, at, tau, nelec, _ = supercell_x(model)
    n = HRs.shape[0]
    rng = np.random.default_rng(5)
    m = rng.standard_normal((3, n, n)) + 1j * rng.standard_normal((3, n, n))
    d0 = 0.2 * (m + np.conj(np.transpose(m, (0, 2, 1))))  # constant Hermitian dipoles
    terms = bloch_terms_from_HRs(HRs, at, model.alat, tau)
    kpts = np.array([[0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    H, dH, d2H = hamiltonian_derivatives_block(terms, kpts)
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    tol = 1e-6
    cluster = nlo.degenerate_clusters(E, tol)
    rng2 = np.random.default_rng(3)
    V = np.zeros_like(U)
    for ik in range(kpts.shape[0]):
        for c in np.unique(cluster[ik]):
            idx = np.nonzero(cluster[ik] == c)[0]
            q, _ = np.linalg.qr(rng2.standard_normal((2, 2)) + 1j * rng2.standard_normal((2, 2)))
            V[ik][np.ix_(idx, idx)] = q
    dip = np.repeat(d0[None], 2, axis=0)
    grad = np.zeros((2, 3, 3, n, n), dtype=complex)
    geometries = [
        nlo.band_geometry(E, U, v, w, tol, dip, grad),
        nlo.band_geometry(E, U @ V, v, w, tol, dip, grad),
    ]
    nocc = nelec // 2
    comps = nlo.parse_components(['xyz', 'yzx', 'xxy', 'zzz'])
    energy = np.linspace(0.2, 8.0, 40)
    edges = -20.0 + 0.02 * np.arange(2001)
    outs = []
    for geo in geometries:
        f = nlo.occupations(E, nocc)
        out = {key: np.zeros((4, energy.size)) for key in ('shift', 'shiftm', 'injectc', 'injectm')}
        nlo.accumulate_shift_and_injection(
            geo, f, geo.omega, energy, np.full(geo.omega.shape, 0.1), comps, 'gauss', out
        )
        hist = {key: np.zeros((4, 2000), dtype=complex) for key in ('hist1', 'hist2')}
        nlo.accumulate_shg_histograms(geo, f, E, edges, comps, hist, tol)
        outs.append({**out, **hist})
    ref = max(np.max(np.abs(outs[0]['shift'])), np.max(np.abs(outs[0]['hist1'])))
    for key, a in outs[0].items():
        assert np.allclose(a, outs[1][key], atol=1e-10 * np.max(np.abs(a)) + 1e-13 * ref), key


def test_spectra_with_dipoles_are_gauge_invariant_and_differ_from_the_tight_binding_ones():
    """The driver's dipole path (finite-difference gradient, gauge term) is invariant under a
    unitary mixing of the orbitals applied to the Hamiltonian and to the dipoles together."""
    model = random_hopping_model(seed=6, time_reversal=False)
    dipoles, _ = synthetic_dipoles(model, seed=12)
    energy = np.linspace(0.5, 8.0, 31)
    comps = ['xyz', 'yyy', 'zxx']
    plain = spectra(model, (4, 4, 4), energy, components=comps)
    base = spectra(model, (4, 4, 4), energy, components=comps, position_dipoles=dipoles)
    assert not np.allclose(plain['shift'], base['shift'], rtol=1e-3)
    rng = np.random.default_rng(2)
    P = np.zeros((4, 4), dtype=complex)
    for lo, hi in ((0, 2), (2, 4)):  # orbitals sharing a centre
        m = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
        P[lo:hi, lo:hi], _ = np.linalg.qr(m)
    HRs = np.einsum('ij,jkabcs,lk->ilabcs', P, model.HRs, P.conj())

    def rotated_dipoles(K):
        return np.einsum('ij,kajl,ml->kaim', P, dipoles(K), P.conj())

    rotated = nlo.nonlinear_optical_spectra(
        HRs, model.at, model.alat, model.volume, model.tau, model.nelec, (4, 4, 4), energy,
        components=comps, eta=0.05, kblock=64, degeneracy_tol=1e-4, position_dipoles=rotated_dipoles,
    )  # fmt: skip
    assert rotated['position_dipoles'] and base['position_dipoles']
    for key in ('shift', 'shiftm', 'injectc', 'injectm', 'shg'):
        scale = max(np.max(np.abs(base[key])), 1e-6 * natural_scale(model, (4, 4, 4), key))
        assert np.allclose(base[key], rotated[key], atol=1e-9 * scale), key


# --------------------------------------------------------------------------- symmetry reduction
def test_irreducible_wedge_reproduces_the_full_grid():
    """Gapped graphene (D3h): the wedge run with the closure of the requested components
    equals the full-grid run for every kind, including the forbidden component."""
    from PAOFLOW.utils import point_groups as pg

    model = honeycomb_model(delta=0.5)
    ops_cart = pg.point_group_operations('-6m2_x')
    bg = np.linalg.inv(model.at).T
    ops_frac = pg.fractional_k_from_cartesian(ops_cart, bg)
    assert np.allclose(ops_frac, np.rint(ops_frac), atol=1e-8)
    energy = np.linspace(0.2, 6.0, 41)
    comps = ['yyy', 'yxx', 'xxy', 'xxx']
    kwargs = dict(components=comps, eta=0.08, degeneracy_tol=1e-4)
    full = spectra(model, (18, 18, 1), energy, **kwargs)
    wedge = spectra(model, (18, 18, 1), energy, symmetry_ops_cryst=np.rint(ops_frac), **kwargs)
    assert wedge['symmetry']['operations'] == 12
    assert (
        wedge['symmetry']['k_irreducible'] <= 324 / 4
    )  # 6 in-plane operations, small orbits on the lines
    assert wedge['components'] == comps
    for key in ('shift', 'shiftm', 'injectc', 'injectm', 'shg'):
        # the T-odd kinds and the injection current of D3h vanish: roundoff of the natural scale
        atol = 1e-9 * np.max(np.abs(full[key])) + 1e-11 * natural_scale(model, (18, 18, 1), key)
        assert np.allclose(full[key], wedge[key], atol=atol), key
    # the identity alone changes nothing
    same = spectra(model, (18, 18, 1), energy, symmetry_ops_cryst=np.eye(3)[None], **kwargs)
    for key in ('shift', 'shg'):
        assert np.allclose(full[key], same[key], atol=1e-12 * np.max(np.abs(full[key])))
    # an operation that is not a symmetry of the Hamiltonian is rejected
    bad = pg.fractional_k_from_cartesian(pg._rot('z', 90.0), bg)
    with pytest.raises(ValueError):
        spectra(model, (18, 18, 1), energy, symmetry_ops_cryst=np.rint(bad), **kwargs)
