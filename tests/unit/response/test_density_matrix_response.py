"""Density-matrix (equation-of-motion) second-order response: insulator limits, metals, conventions."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import (
    bloch_terms_from_HRs,
    hamiltonian_derivatives_block,
    velocity_and_curvature_operators,
)
from PAOFLOW.response import density_matrix_response as dmr
from PAOFLOW.response import nonlinear_optics as nlo
from PAOFLOW.utils.constants import ELECTRON_CHARGE_SI, EPSILON0_SI, HBAR

_spec = importlib.util.spec_from_file_location(
    '_nonlinear_models', Path(__file__).resolve().parent / '_nonlinear_models.py'
)
_models = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_models)

KB = 8.617333262e-5


def eom(model, nk, energy, response, **kwargs):
    options = dict(components='all', eta=0.02, degeneracy_tol=1e-4, kblock=4096)
    options.update(kwargs)
    nelec = options.pop('nelec', model.nelec)
    return dmr.density_matrix_spectra(
        model.HRs, model.at, model.alat, model.volume, model.tau, nelec, nk, energy,
        response=response, **options,
    )  # fmt: skip


def reference(model, nk, energy, **kwargs):
    options = dict(components='all', eta=0.02, degeneracy_tol=1e-4, kblock=4096)
    options.update(kwargs)
    return nlo.nonlinear_optical_spectra(
        model.HRs, model.at, model.alat, model.volume, model.tau, model.nelec, nk, energy, **options
    )


def molecule_in_a_box():
    """Five orbitals of a T-even molecule in a 30 bohr box: no hopping between boxes."""
    rng = np.random.default_rng(3)
    n = 5
    tau = np.array(
        [[0, 0, 0], [0.04, 0.01, 0.0], [0.01, 0.05, 0.02], [0.06, 0.04, 0.05], [0.02, 0.07, 0.06]]
    )
    h = rng.standard_normal((n, n))
    h = 0.5 * (h + h.T) + np.diag([-3.0, -2.0, 0.5, 2.0, 3.5])
    hops = [(i, j, np.zeros(3, dtype=int), h[i, j]) for i in range(n) for j in range(n)]
    return _models.ExactModel(hops, n, np.eye(3), 30.0, tau, (3, 3, 3), nelec=4), h


def one_band_metal():
    """A T-broken band (odd second harmonics) and a decoupled empty band: only the Drude term."""
    hops = [(1, 1, (0, 0, 0), 12.0)]
    for R, h in (
        ((1, 0, 0), -1.0),
        ((0, 1, 0), -0.8),
        ((2, 0, 0), 0.25j),
        ((1, 1, 0), 0.1 + 0.15j),
        ((1, -1, 0), -0.1j),
    ):
        hops.append((0, 0, R, h))
        hops.append((0, 0, tuple(-np.array(R)), np.conj(h)))
    at = np.diag([1.0, 1.0, 10.0])
    return _models.ExactModel(hops, 2, at, 5.0, np.zeros((2, 3)), (5, 5, 1), nelec=0.8), hops


def k_grid_2d(n):
    axis = np.arange(n) / n
    return np.stack(np.meshgrid(axis, axis, [0.0], indexing='ij'), -1).reshape(-1, 3)


# --------------------------------------------------------------------------- helpers
def test_fermi_dirac_derivatives_and_relaxation_rates():
    e = np.linspace(-0.3, 0.3, 61)
    f, fp, fpp = dmr.fermi_dirac_derivatives(e, 0.05, 400.0)
    h = 1e-5
    fph = dmr.fermi_dirac_derivatives(e + h, 0.05, 400.0)[0]
    fmh = dmr.fermi_dirac_derivatives(e - h, 0.05, 400.0)[0]
    assert np.allclose(fp, (fph - fmh) / (2 * h), atol=1e-6)
    assert np.allclose(fpp, (fph - 2 * f + fmh) / h**2, atol=1e-2 * np.abs(fpp).max())
    step, d1, d2 = dmr.fermi_dirac_derivatives(e, 0.05, 0.0)
    assert np.array_equal(step, (e < 0.05).astype(float)) and not d1.any() and not d2.any()
    # tau(e) = a + b e (fs): gamma = hbar/tau and its slope -hbar b/tau^2
    gamma, dgamma = dmr.intraband_rates(e, lambda x: 20.0 + 10.0 * x, 0.05)
    assert np.allclose(gamma, dmr.HBAR_EV_FS / (20.0 + 10.0 * e))
    assert np.allclose(dgamma, -dmr.HBAR_EV_FS * 10.0 / (20.0 + 10.0 * e) ** 2, rtol=1e-6)
    const, slope = dmr.intraband_rates(e, None, 0.05)
    assert np.allclose(const, 0.05) and not slope.any()


# --------------------------------------------------------------------------- insulators
def test_static_shg_of_a_molecule_matches_the_finite_field():
    """chi(w -> 0) = (1/2 eps0) d2P/dE2 of the field-dressed ground state (charge e > 0, as the code)."""
    model, h = molecule_in_a_box()
    X = model.alat * (model.tau @ model.at)

    def mean_position(field):
        _, U = np.linalg.eigh(h + np.diag(X @ field))
        return 2.0 * (np.sum(np.abs(U[:, :2]) ** 2, axis=1) @ X)

    step = 2e-3
    comps = [(0, 1, 2), (0, 0, 0), (1, 0, 0), (2, 1, 1)]
    finite_field = []
    for a, b, c in comps:
        eb, ec = np.eye(3)[b] * step, np.eye(3)[c] * step
        d2 = (
            mean_position(eb + ec)
            - mean_position(eb - ec)
            - mean_position(-eb + ec)
            + mean_position(-eb - ec)
        ) / (4 * step**2)
        # electrons: P = -e <x>/V; the code's convention (charge +e) flips it
        finite_field.append(1e12 * 0.5 * ELECTRON_CHARGE_SI * d2[a] / model.volume / EPSILON0_SI)
    finite_field = np.array(finite_field)
    energy = np.array([0.01])
    out = eom(model, (1, 1, 1), energy, 'shg', components=comps, eta=1e-4)
    chen = reference(model, (1, 1, 1), energy, components=comps, eta=1e-4, want=('shg',))
    assert np.allclose(out['shg'][:, 0].real, finite_field, rtol=2e-3)
    assert np.allclose(chen['shg'][:, 0].real, finite_field, rtol=2e-3)
    assert np.all(np.abs(out['shg'][:, 0].imag) < 0.05 * np.abs(finite_field))


def test_insulator_shg_below_the_gap_matches_the_chen_form():
    model = _models.random_hopping_model(seed=8, time_reversal=False)  # gap 1.37 eV
    energy = np.array([0.5, 0.65])
    comps = [(0, 1, 2), (1, 1, 1), (2, 0, 0), (0, 0, 1)]
    out = eom(model, (10, 10, 10), energy, 'shg', components=comps, eta=0.002, kblock=1000)
    chen = reference(model, (10, 10, 10), energy, components=comps, eta=0.002, want=('shg',))
    scale = np.abs(chen['shg']).max()
    assert np.all(np.abs(out['shg'] - chen['shg']) < 0.01 * scale)
    assert not np.any(out['metallic'])
    for ch in ('fermi_surface_mixed', 'fermi_surface_drude'):
        assert not np.any(out['channels'][ch])  # step occupations: no Fermi surface


def test_zero_frequency_currents_of_an_insulator_are_the_shift_and_injection_currents():
    """sigma = shift + tau eta (linear and circular) with the same signs as nonlinear_optics.

    The real populations carry the relaxation time (here 300 fs, unrelated to eta) and the Gaussian
    delta of the injection formulas."""
    energy = np.array([1.2, 2.0, 2.8])
    comps = [(1, 1, 1), (0, 0, 1), (1, 0, 0)]
    eta, tau_fs = 0.02, 300.0
    tau = tau_fs * 1e-15
    graphene = _models.honeycomb_model(0.6, -1.0, 0.0, 0.0, strain=0.2)  # T-even
    nk = (400, 400, 1)
    out = eom(graphene, nk, energy, 'dc', components=comps, eta=eta, relaxation_time=tau_fs)
    shift = reference(graphene, nk, energy, components=comps, eta=eta, smearing='lorentz')
    inject = reference(graphene, nk, energy, components=comps, eta=eta, smearing='gauss')
    inter, intra = out['channels']['fermi_sea_inter'], out['channels']['fermi_sea_intra']
    shift_scale = np.abs(shift['shift']).max()
    assert np.all(np.abs(inter.real - shift['shift']) < 0.03 * shift_scale)
    inject_scale = tau * np.abs(inject['injectc']).max()
    assert inject_scale > 0.0
    assert np.all(np.abs(out['dc'].imag + tau * inject['injectc']) < 0.005 * inject_scale)
    # T-odd populations (Haldane flux): the magnetic injection
    haldane = _models.honeycomb_model(0.6, -1.0, 0.15, np.pi / 2, strain=0.2)
    nk = (200, 200, 1)
    out = eom(haldane, nk, energy, 'dc', components=comps, eta=0.05, relaxation_time=tau_fs)
    inject = reference(haldane, nk, energy, components=comps, eta=0.05, smearing='gauss')
    intra = out['channels']['fermi_sea_intra']
    scale = tau * np.abs(inject['injectm']).max()
    assert scale > 0.0
    assert np.all(np.abs(intra.real - tau * inject['injectm']) < 0.01 * scale)


def test_no_dc_current_below_the_gap_and_the_gyration_current():
    """Floquet: a clean insulator driven below its gap carries no DC current, whatever tau.

    The coherences carry an off-resonant principal-value term that the virtual (dressing)
    populations cancel only with the adiabatic weight ('hybrid'); relaxing all populations with
    gamma leaves a current that grows with tau. Above the gap the T-odd circular current (the
    phi-odd part, free of the T-even injection) is the delta-function gyration current shiftm."""
    models = {
        s: _models.honeycomb_model(0.6, -1.0, 0.15, s * np.pi / 2, strain=0.2) for s in (1, -1)
    }
    below = np.array([0.46, 0.7])  # the direct gap is 0.93 eV
    comps = [(0, 0, 1)]
    totals = {}
    for n, eta in ((150, 0.04), (300, 0.02)):
        for pops in ('hybrid', 'relaxation'):
            out = eom(models[1], (n, n, 1), below, 'dc', components=comps, eta=eta,
                      relaxation_time=300.0, populations=pops)  # fmt: skip
            totals[(eta, pops)] = out['dc'][0].imag
    # hybrid: vanishes linearly with eta; relaxation: finite and growing as 1/gamma
    assert np.allclose(totals[(0.02, 'hybrid')], 0.5 * totals[(0.04, 'hybrid')], rtol=0.15)
    assert np.all(np.abs(totals[(0.02, 'relaxation')]) > 50 * np.abs(totals[(0.02, 'hybrid')]))
    above = np.array([1.0, 2.2])
    eom_odd, ref_odd = 0.0, 0.0
    for s, model in models.items():
        out = eom(model, (400, 400, 1), above, 'dc', components=comps, eta=0.02)
        ref = reference(model, (400, 400, 1), above, components=comps, eta=0.02,
                        smearing='lorentz', want=('shift',))  # fmt: skip
        eom_odd = eom_odd - 0.5 * s * out['dc'][0].imag
        ref_odd = ref_odd + 0.5 * s * ref['shiftm'][0]
    assert np.allclose(eom_odd, ref_odd, rtol=0.02)


def test_quantized_circular_photogalvanic_trace_with_adaptive_widths():
    """de Juan et al.: one quantum 2 pi e^3/h^2 per Weyl node inside the Pauli window."""
    model = _models.weyl_model(t=1.0, b=0.3)
    quantum = 2.0 * np.pi * ELECTRON_CHARGE_SI**3 / 6.62607015e-34**2
    perms = [((0, 1, 2), 1), ((1, 2, 0), 1), ((2, 0, 1), 1), ((0, 2, 1), -1), ((1, 0, 2), -1),
             ((2, 1, 0), -1)]  # fmt: skip
    tau_fs = 2000.0
    out = eom(
        model, (60, 60, 60), np.array([0.5, 0.7]), 'dc', components=[p for p, _ in perms],
        eta=0.01, spin_factor=1.0, degeneracy_tol=1e-6, fermi_energy=-0.3, temperature=50.0,
        relaxation_time=tau_fs, adaptive_smearing=1.0, nelec=1.0, kblock=20000,
    )  # fmt: skip
    pops = out['channels']['fermi_sea_intra']
    trace = sum(sign * (-pops[i].imag) for i, (_, sign) in enumerate(perms))
    assert np.allclose(np.abs(trace / (tau_fs * 1e-15) / quantum), 1.0, atol=0.03)


# --------------------------------------------------------------------------- metals
def test_nonlinear_drude_of_one_band_matches_the_closed_form():
    """sigma = 2 e^3/(w^2 + gamma^2) int f d_a d_b d_c eps (hbar = 1), proportional to tau^2."""
    model, hops = one_band_metal()
    n, temperature, mu, tau_fs = 120, 600.0, -0.5, 30.0
    energy = np.array([0.002, 0.1])
    comps = [(0, 0, 0), (1, 0, 0), (0, 1, 1), (1, 1, 1)]
    with pytest.raises(ValueError, match='metallic channel'):
        eom(model, (n, n, 1), energy, 'dc', components=comps)
    out = eom(
        model, (n, n, 1), energy, 'dc', components=comps, relaxation_time=tau_fs,
        fermi_energy=mu, temperature=temperature,
    )  # fmt: skip
    assert out['metallic']
    kc = k_grid_2d(n) @ np.linalg.inv(model.at).T * 2 * np.pi / model.alat
    eps = np.zeros(len(kc), dtype=complex)
    d3 = np.zeros((len(kc), 3, 3, 3), dtype=complex)
    for i, _, R, h in hops:
        if i == 0:
            d = model.alat * (np.asarray(R, dtype=float) @ model.at)
            phase = h * np.exp(1j * kc @ d)
            eps += phase
            d3 += -1j * phase[:, None, None, None] * np.einsum('a,b,c->abc', d, d, d)[None]
    f = 1.0 / (1.0 + np.exp((eps.real - mu) / (KB * temperature)))
    gamma = dmr.HBAR_EV_FS / tau_fs
    norm = 2.0 / (n * n * model.volume)
    drude = out['channels']['fermi_surface_drude']
    scale = np.abs(drude).max()
    for ic, (a, b, c) in enumerate(comps):
        closed = ELECTRON_CHARGE_SI / HBAR * norm * 2.0 * np.sum(f * d3[:, a, b, c].real)
        closed = closed / (energy**2 + gamma**2)
        assert np.allclose(drude[ic], closed, rtol=5e-3, atol=1e-4 * scale)
    for ch in ('fermi_sea_inter', 'fermi_sea_intra', 'fermi_surface_mixed'):
        assert not np.any(out['channels'][ch])  # no interband matrix elements


def test_berry_curvature_dipole_of_a_doped_gapped_graphene():
    """Mixed channel = i D^{b,ac}/(w + i gamma) + i D^{c,ab}/(-w + i gamma), D = int df Omega."""
    model = _models.honeycomb_model(0.6, -1.0, 0.0, 0.0, strain=0.3)  # T-even, no C3
    n, temperature, mu, eta = 300, 800.0, 0.45, 0.02
    energy = np.array([1e-5, 0.1])
    comps = [(0, 0, 1), (1, 0, 1), (1, 1, 0)]
    results = {}
    for tau_fs in (20.0, 60.0):
        results[tau_fs] = eom(
            model, (n, n, 1), energy, 'dc', components=comps, eta=eta, nelec=1.0,
            relaxation_time=tau_fs, fermi_energy=mu, temperature=temperature, kblock=10000,
        )  # fmt: skip
    terms = bloch_terms_from_HRs(model.HRs, model.at, model.alat, model.tau)
    H, dH, d2H = hamiltonian_derivatives_block(terms, k_grid_2d(n))
    v, w = velocity_and_curvature_operators(terms, H, dH, d2H)
    E, U = np.linalg.eigh(H)
    geometry = nlo.band_geometry(E, U, v, w, 1e-4)
    fp = dmr.fermi_dirac_derivatives(E, mu, temperature)[1]
    velocity = np.real(np.diagonal(geometry.velocity_intra, axis1=2, axis2=3))
    r = geometry.position
    omega = -2.0 * np.imag(np.einsum('kanm,kcmn->kacn', r, r))  # Omega^{ac}_n
    D = np.einsum('kn,kbn,kacn->bac', fp, velocity, omega) * 2.0 / (n * n * model.volume)
    for tau_fs, out in results.items():
        gamma = dmr.HBAR_EV_FS / tau_fs
        mixed = out['channels']['fermi_surface_mixed']
        for ic, (a, b, c) in enumerate(comps):
            formula = 1j * D[b, a, c] / (energy + 1j * gamma) + 1j * D[c, a, b] / (
                -energy + 1j * gamma
            )
            formula = ELECTRON_CHARGE_SI / HBAR * formula
            assert np.allclose(mixed[ic], formula, rtol=2e-3, atol=1e-3 * np.abs(formula).max())
        assert np.abs(out['channels']['fermi_surface_drude']).max() < 1e-9 * np.abs(mixed).max()
    # zero-frequency limit proportional to tau
    ratio = (
        results[60.0]['channels']['fermi_surface_mixed'][:, 0]
        / (results[20.0]['channels']['fermi_surface_mixed'][:, 0])
    )
    assert np.allclose(ratio.real, 3.0, rtol=0.03)


def test_relaxation_time_callable_and_guards():
    model, _ = one_band_metal()
    energy = np.array([0.05])
    options = dict(
        components=[(0, 0, 0), (1, 0, 0)], fermi_energy=-0.5, temperature=600.0
    )  # fmt: skip
    constant = eom(model, (24, 24, 1), energy, 'dc', relaxation_time=30.0, **options)
    function = eom(
        model, (24, 24, 1), energy, 'dc', relaxation_time=lambda e: np.full_like(e, 30.0), **options
    )  # fmt: skip
    assert np.allclose(constant['dc'], function['dc'], rtol=1e-12, atol=0.0)
    sloped = eom(
        model, (24, 24, 1), energy, 'dc', relaxation_time=lambda e: 30.0 + 5.0 * e, **options
    )
    assert not np.allclose(sloped['dc'], constant['dc'], rtol=1e-3)
    with pytest.raises(ValueError, match='scissor'):
        eom(model, (6, 6, 1), energy, 'dc', scissor=0.5, **options)
    with pytest.raises(ValueError, match='response'):
        eom(model, (6, 6, 1), energy, 'optical_rectification', **options)


# --------------------------------------------------------------------------- orchestrator
def test_orchestrator_switches_to_the_density_matrix_engine_for_a_metal(tmp_path):
    """PAOFLOW.nonlinear_optics on doped gapped graphene: photocurrent and SHG channels, plots."""
    import matplotlib

    matplotlib.use('Agg')
    from PAOFLOW import GPAO, PAOFLOW

    model = {'label': 'graphene', 't': 2.7, 'delta': 1.0}
    pao = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path / 'out'), verbose=False)
    options = dict(nk=(24, 24, 1), emin=0.5, emax=3.0, ne=6, eta=0.1, components=['yyy', 'xxy'])
    with pytest.raises(ValueError, match='temperature'):
        pao.nonlinear_optics(nelec=2.5, fermi_energy=0.8, **options)
    res = pao.nonlinear_optics(
        nelec=2.5, fermi_energy=0.8, temperature=600.0, relaxation_time=25.0, twoD=True, **options
    )
    attr = pao.data_controller.data_dicts()[1]
    meta = attr['nonlinear_optics']
    assert meta['fermi_surface'] and meta['metallic'] and meta['relaxation_time_fs'] == 25.0
    for key, channels in (('shg', 'shg_terms'), ('photocurrent', 'photocurrent_channels')):
        assert set(res[channels]) == set(dmr.CHANNELS)
        assert np.allclose(
            sum(res[channels].values()), res[key], atol=1e-12 * np.abs(res[key]).max()
        )
    assert np.abs(res['shg_terms']['fermi_surface_drude']).max() > 0.0  # a Fermi surface
    opath = Path(attr['opath'])
    for name in (
        'pcr_yyy.dat', 'pci_xxy.dat', 'pcr_fermi_surface_mixed_yyy.dat', 'shgr_yyy.dat',
        'shgtr_fermi_sea_inter_yyy.dat', 'shift_yyy.dat', 'injectc_xxy.dat',
    ):  # fmt: skip
        assert (opath / name).exists(), name
    plots = GPAO.GPAO()
    plots.plot_photocurrent_channels(
        path=str(opath), component='yyy', two_d=True, filename=str(tmp_path / 'pc.png')
    )
    plots.plot_shg_terms(path=str(opath), component='yyy', filename=str(tmp_path / 'shg.png'))
    assert (tmp_path / 'pc.png').exists() and (tmp_path / 'shg.png').exists()
    # an insulator keeps the convergent SHG form
    res = pao.nonlinear_optics(nelec=2, **options)
    assert 'photocurrent' not in res and not attr['nonlinear_optics']['fermi_surface']


def test_velocity_gauge_kubo_reference_agrees_with_the_engine():
    """Peierls (velocity gauge) perturbation theory with adiabatic switching, a different gauge and code.

    With every second-order element damped by 2 eta (populations 'relaxation' with gamma = 2 eta,
    coherence_damping = 2) the engine is the length-gauge form of the same adiabatic theory: the shift,
    gyration and both injection currents of the strained Haldane model agree on the same grid."""
    _vspec = importlib.util.spec_from_file_location(
        '_velocity_gauge', Path(__file__).resolve().parent / '_velocity_gauge.py'
    )
    vg = importlib.util.module_from_spec(_vspec)
    _vspec.loader.exec_module(vg)
    model = _models.honeycomb_model(0.6, -1.0, 0.15, np.pi / 2, strain=0.2)
    comps = [(0, 0, 1), (1, 1, 1), (0, 0, 0), (1, 0, 0)]
    nk, eta = (150, 150, 1), 0.04
    for w in (1.2, 2.2):
        kubo = vg.sigma_si(model, nk, vg.kubo_second_order(model, nk, (w, -w), comps, eta, nelec=2))
        out = eom(
            model, nk, np.array([w]), 'dc', components=comps, eta=eta, nelec=2,
            relaxation_time=HBAR * 1e15 / (2 * eta), populations='relaxation', coherence_damping=2.0,
        )  # fmt: skip
        assert np.allclose(out['dc'][:, 0], kubo, rtol=0, atol=1e-6 * np.abs(kubo).max())
