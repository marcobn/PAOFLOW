"""Frequency-dependent Kubo-Greenwood response, PAOFLOW.kubo_optical_response (14 November 2026).

- the kernel: its Hall part is the closed form of Yao et al. 2004, Eq. 7, with the oscillator damping;
  its omega -> 0 limit is the Berry-curvature sea at any eta; the absorptive part is positive and vanishes
  as omega^2 in an insulator (no sub-gap tail); a vanishing gap turns a transition into a Drude term;
- the static limits on models: interband Hall = the intrinsic term of kubo_response (Haldane Chern
  number), intraband = the Boltzmann (tutorial03 Eq. 4) conductivity;
- the output file.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from PAOFLOW.response import kubo_kernels as kk

from . import _kubo_cases as kc

pytestmark = pytest.mark.unit


def _system(seed: int = 5, nb: int = 6):
    rng = np.random.default_rng(seed)

    def hermitian():
        m = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        return (m + m.conj().T) / 2

    energies = np.sort(rng.normal(size=nb) * 1.5)
    occupations = (energies < 0.1).astype(float)
    return energies, hermitian(), hermitian(), occupations


def test_hall_part_is_the_closed_form_of_yao():
    energies, a, b, f = _system()
    omegas, eta = np.array([0.0, 0.3, 1.1, 2.5]), 0.07
    pair_ab, pair_ba = a * b.T, b * a.T
    ab, _ = kk.optical_kernel(energies, pair_ab, f, np.zeros_like(f), omegas, eta)
    ba, _ = kk.optical_kernel(energies, pair_ba, f, np.zeros_like(f), omegas, eta)
    hall = 0.5 * (ab - ba)
    gap = energies[None, :] - energies[:, None]  # [n, m] = e_m - e_n
    df = f[:, None] - f[None, :]
    off = gap != 0
    closed = [
        np.sum(df[off] * np.imag(pair_ab)[off] / (gap[off] ** 2 - w**2 - 2j * eta * w))
        for w in omegas
    ]
    assert np.allclose(hall, closed, rtol=1e-12, atol=1e-12)


def test_static_limit_is_the_berry_curvature_sea():
    """At omega = 0 the oscillator Hall part is the clean-limit sea for any broadening."""
    energies, a, b, f = _system()
    omega = kk.berry_curvature(energies, a, b, 0.0, cutoff=1e-20)
    for eta in (1e-7, 0.05, 0.3):
        inter, _ = kk.optical_kernel(energies, a * b.T, f, np.zeros_like(f), np.array([0.0]), eta)
        inter_ba, _ = kk.optical_kernel(
            energies, b * a.T, f, np.zeros_like(f), np.array([0.0]), eta
        )
        assert 0.5 * (inter - inter_ba)[0].real == pytest.approx(-(f @ omega), rel=1e-12)


def test_insulator_does_not_absorb_below_the_gap():
    """Re sigma_aa ~ omega^2 at omega -> 0 (the omega + i eta substitution gave Re sigma(0) ~ eta)."""
    energies, a, _, f = _system()
    omegas = np.array([0.0, 1e-3, 2e-3])
    inter, _ = kk.optical_kernel(energies, a * a.T, f, np.zeros_like(f), omegas, 0.1)
    assert abs(inter[0]) < 1e-12
    assert inter.real[2] / inter.real[1] == pytest.approx(4.0, rel=1e-3)


def test_closing_gap_turns_a_transition_into_a_drude_term():
    """Two levels 2 delta apart, half filled at the Fermi level: as delta -> 0 the transition
    tends to i (-f') P / (omega + 2 i eta) with (f_n - f_m) / (e_m - e_n) -> -f', at omega > 0 (at
    omega = 0 exactly a finite gap does not conduct)."""
    eta, omegas = 0.05, np.array([0.02, 0.1, 0.4])
    pair = np.array([[0.0, 0.7], [0.7, 0.0]], dtype=complex)
    width, delta = 0.02, 1e-6
    energies = np.array([-delta, delta])
    occupations = 0.5 * (1.0 - np.tanh(energies / (2 * width)))
    window = occupations * (1 - occupations) / width
    inter, _ = kk.optical_kernel(energies, pair, occupations, window, omegas, eta, 1e-9)
    _, drude = kk.optical_kernel(energies, pair, occupations, window, omegas, eta, 1e-3)
    assert np.allclose(inter, drude, rtol=1e-6)


def test_absorption_is_positive():
    energies, a, _, f = _system()
    omegas = np.linspace(0.0, 6.0, 61)
    inter, _ = kk.optical_kernel(energies, a * a.T, f, np.zeros_like(f), omegas, 0.05)
    assert np.all(inter.real[1:] > -1e-12)


def test_static_hall_conductivity_is_the_intrinsic_term():
    """Haldane model in its gap: Re sigma_xy(omega -> 0) = intrinsic sigma_xy, 2 pi sigma_xy = 2 C."""
    p = kc.prepared('haldane', tempfile.mkdtemp())
    o = p.kubo_optical_response('charge_current', ['xy', 'xx'], eta=1e-5, omega_min=0.0, omega_max=6.0, nomega=31, twoD=True, write=False)  # fmt: skip
    d = p.kubo_response('charge_current', ['xy'], terms=('intrinsic',), emin=0.0, emax=0.0, ne=1, twoD=True, write=False)  # fmt: skip
    assert o['xy']['interband'][0].real == pytest.approx(d['xy']['intrinsic'][0], rel=1e-8)
    assert 2 * np.pi * o['xy']['total'][0].real == pytest.approx(2.0, abs=5e-3)
    broad = p.kubo_optical_response('charge_current', ['xy'], eta=0.1, omega_min=0.0, omega_max=6.0, nomega=31, twoD=True, write=False)  # fmt: skip
    assert broad['xy']['interband'][0].real == pytest.approx(d['xy']['intrinsic'][0], rel=1e-8)
    assert np.all(o['xx']['interband'].real[1:] > 0.0)  # absorption


def test_static_drude_term_is_the_boltzmann_conductivity(tmp_path):
    """Intraband term at omega = 0: sum_n (-f'_n) v_n^2 / (2 eta), tutorial03 Eq. 4 (whose 1e-4
    regulariser is undone here)."""
    p = kc.prepared('rashba', str(tmp_path))
    eta = 0.05
    o = p.kubo_optical_response('charge_current', ['xx'], eta=eta, omega_min=0.0, omega_max=1.0, nomega=3, mu=-2.0, write=False)  # fmt: skip
    p.linear_response(response='cond', intraband=True, gamma=eta, a_tensor=[[0, 0]], eminH=-2.0, emaxH=-2.0, esize=1)  # fmt: skip
    legacy = np.loadtxt(os.path.join(str(tmp_path), 'Cond_xx.dat'))[1]
    assert o['xx']['intraband'][0].real == pytest.approx(
        legacy * (2 * eta + 1e-4) / (2 * eta), rel=1e-10
    )
    # Drude: sigma(omega) = sigma(0) * 2 eta / (2 eta - i omega)
    drude = o['xx']['intraband']
    assert np.allclose(drude, drude[0] * 2 * eta / (2 * eta - 1j * o['xx']['omega']), rtol=1e-12)


def test_spin_hall_static_limit_and_charge_sign(tmp_path):
    """Kane-Mele: the AC spin Hall response at omega -> 0 is the intrinsic spin Hall term (same
    electron-charge sign)."""
    p = kc.prepared('kane_mele', str(tmp_path))
    o = p.kubo_optical_response('spin_current', ['zxy'], eta=1e-5, omega_min=0.0, omega_max=1.0, nomega=3, write=False)  # fmt: skip
    d = p.kubo_response('spin_current', ['zxy'], terms=('intrinsic',), emin=0.0, emax=0.0, ne=1, write=False)  # fmt: skip
    assert o['z_xy']['total'][0].real == pytest.approx(d['z_xy']['intrinsic'][0], rel=1e-6)


def test_output_file(tmp_path):
    p = kc.prepared('rashba', str(tmp_path))
    o = p.kubo_optical_response('charge_current', ['xy'], eta=0.05, nomega=11, mu=-2.0)
    data = np.loadtxt(os.path.join(str(tmp_path), 'kubo_optical_charge_current_xy.dat'))
    assert data.shape == (11, 7)  # omega, Re/Im of total, interband, intraband
    assert np.allclose(data[:, 1] + 1j * data[:, 2], o['xy']['total'], rtol=1e-9)
    with open(os.path.join(str(tmp_path), 'kubo_optical_charge_current_xy.dat')) as f:
        header = f.read(600)
    assert 'electron charge' in header and 'computed spin configuration' in header


def test_flat_band_does_not_poison_the_occupations(tmp_path):
    """A decoupled band with zero velocity has zero adaptive width; the Drude term must stay finite."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import single_band_model

    model = single_band_model(
        {(1, 0, 0): -1.0, (0, 1, 0): -1.0}, np.diag([1.0, 1.0, 10.0]), 6.0, 0.6, (3, 3, 1)
    )
    p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), smearing='gauss', verbose=False)
    p.interpolated_hamiltonian(nfft1=48, nfft2=48, nfft3=1)
    p.pao_eigh()
    p.gradient_and_momenta()
    p.adaptive_smearing()
    from PAOFLOW.spectrum.do_dos import ZERO_WIDTH

    # the remote band: zero velocity, stored as the finite ZERO_WIDTH / 2
    assert np.min(p.data_controller.data_arrays['deltakp']) == 0.5 * ZERO_WIDTH
    o = p.kubo_optical_response(
        'charge_current', ['xx'], eta=0.05, omega_max=1.0, nomega=5, mu=-1.0, write=False
    )
    assert np.all(np.isfinite(o['xx']['total'])) and o['xx']['intraband'][0].real > 0
    d = p.kubo_response(
        'charge_current', ['xx'], terms=('intrinsic',), emin=-1.0, emax=-1.0, ne=1, write=False
    )
    assert np.all(np.isfinite(d['xx']['intrinsic']))
