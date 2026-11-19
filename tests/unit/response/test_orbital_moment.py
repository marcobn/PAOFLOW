"""Orbital moment of Bloch states and the orbital magnetization (modern theory).

* gapped graphene: m(K) = (hbar v_F)^2 / Delta, equal in the two bands, reversed in the other valley;
* a six-site ring in a large cell: m_n = -dE_n/dB from the Peierls phases (Hellmann-Feynman), every state;
* Haldane model: dM/dmu in the gap equals dN/dB counted in a Peierls magnetic supercell (Streda);
* MoS2 three-band model: the atomic (-4) and Bloch parts of the exciton valley g-factor.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
from PAOFLOW.hamiltonian.magnetic_field import BOHR_MAGNETON_EV_PER_TESLA, FLUX_QUANTUM_TESLA_BOHR2
from PAOFLOW.models import models as M
from PAOFLOW.models.models import HoppingModel
from PAOFLOW.response.orbital_moment import RYDBERG_EV

pytestmark = pytest.mark.unit


def _paoflow(model, tmp_path):
    from PAOFLOW import PAOFLOW

    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)


def _grid_moments(p, n1: int, n2: int, n3: int = 1):
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=n1, nfft2=n2, nfft3=n3)
        p.pao_eigh()
        p.gradient_and_momenta()
        p.orbital_moment()
    return p.data_controller.data_arrays


def _index(n: int, k1: float, k2: float) -> int:
    return round(k1 * n) * n + round(k2 * n)


def test_gapped_graphene_valley_moment(tmp_path):
    gap = 0.4
    p = _paoflow(M.honeycomb_model(delta=gap, t=-1.0), tmp_path)
    arrays = _grid_moments(p, 12, 12)
    alat = p.data_controller.data_attributes['alat']
    closed = (np.sqrt(3) / 2 * alat) ** 2 / (
        gap * RYDBERG_EV
    )  # (hbar v_F)^2 / Delta in mu_B, |t| = 1
    moment_k = arrays['orbital_moment'][_index(12, 2 / 3, 1 / 3), :, 2, 0]
    moment_kp = arrays['orbital_moment'][_index(12, 1 / 3, 2 / 3), :, 2, 0]
    np.testing.assert_allclose(np.abs(moment_k), closed, rtol=1e-10)
    np.testing.assert_allclose(moment_kp, -moment_k, atol=1e-10)
    assert moment_k[0] == pytest.approx(moment_k[1])


def _ring() -> HoppingModel:
    hoppings = tuple(
        h
        for j in range(6)
        for h in ((j, (j + 1) % 6, (0, 0, 0), -1.0), ((j + 1) % 6, j, (0, 0, 0), -1.0))
    )
    angles = 2 * np.pi * np.arange(6) / 6
    positions = np.c_[
        0.5 + 1.5 * np.cos(angles) / 20, 0.5 + 1.5 * np.sin(angles) / 20, np.full(6, 0.5)
    ]
    return HoppingModel(
        'ring', hoppings, 6, np.diag([20.0, 20.0, 20.0]), 1.0, positions, 1.0, (1, 1, 1)
    )


def test_moment_is_minus_the_field_derivative_of_the_energy(tmp_path):
    def levels(flux: float, with_moments: bool = False):
        p = _paoflow(_ring(), tmp_path)
        with contextlib.redirect_stdout(io.StringIO()):
            p.peierls_field(flux_quanta=flux)
        if not with_moments:
            return np.linalg.eigvalsh(
                np.asarray(p.data_controller.data_arrays['HRs'])[:, :, 0, 0, 0, 0]
            )
        arrays = _grid_moments(p, 3, 3, 3)
        return arrays['E_k'][0, :, 0], arrays['orbital_moment'][0, :, 2, 0]

    flux, step = 2.0, 1.0e-3
    _, moments = levels(flux, with_moments=True)
    slope = (levels(flux + step) - levels(flux - step)) / (2 * step)  # eV per flux quantum per cell
    tesla_per_quantum = FLUX_QUANTUM_TESLA_BOHR2 / 400.0  # cell 20 x 20 bohr
    np.testing.assert_allclose(
        moments, -slope / tesla_per_quantum / BOHR_MAGNETON_EV_PER_TESLA, atol=1e-6
    )


def test_streda_relation_in_a_chern_insulator(tmp_path):
    haldane = M.honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2, nelec=1.0)
    p = _paoflow(haldane, tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.interpolated_hamiltonian(nfft1=36, nfft2=36, nfft3=1)
        p.pao_eigh()
        p.gradient_and_momenta()
        mu, magnetization = p.orbital_magnetization(emin=-0.3, emax=0.3, ne=7)
    slope = np.polyfit(mu, magnetization[:, 2], 1)[0]
    # dN/df from a magnetic supercell of 8 cells holding one flux quantum
    s = _paoflow(haldane, tmp_path / 'sc')
    with contextlib.redirect_stdout(io.StringIO()):
        s.doubling_Hamiltonian(3, 0, 0)
        s.peierls_field(flux_quanta=1)
    arrays, attributes = s.data_controller.data_dicts()
    terms = bloch_terms_from_HRs(
        np.asarray(arrays['HRs']), np.asarray(arrays['a_vectors']), float(attributes['alat']), None
    )
    ks = np.array([[i / 4, j / 6, 0.0] for i in range(4) for j in range(6)])
    counts = (np.linalg.eigvalsh(hamiltonian_block(terms, ks)) < 0.0).sum(axis=1)
    assert len(set(counts.tolist())) == 1
    states_per_quantum = int(counts[0]) - 8
    area = (np.sqrt(3) / 2) * attributes['alat'] ** 2
    spin_factor = 2.0  # spin-degenerate model bands
    expected = (
        spin_factor
        * states_per_quantum
        * area
        / (FLUX_QUANTUM_TESLA_BOHR2 * BOHR_MAGNETON_EV_PER_TESLA)
    )
    assert abs(states_per_quantum) == 1
    assert slope == pytest.approx(expected, rel=1e-3)


def test_mos2_valley_moments(tmp_path):
    p = _paoflow(M.tmd_model('MoS2', 'GGA', soc=False), tmp_path)
    arrays = _grid_moments(p, 12, 12)
    l_z = np.zeros((3, 3), complex)
    l_z[1, 2], l_z[2, 1] = 2j, -2j  # <xy|L_z|x2-y2> = 2i
    g = {}
    for name, (k1, k2) in (('K', (2 / 3, 1 / 3)), ("K'", (1 / 3, 2 / 3))):
        ik = _index(12, k1, k2)
        vectors = arrays['v_k'][ik, :, :, 0]
        atomic = np.real(np.diag(vectors.conj().T @ l_z @ vectors))
        bloch = arrays['orbital_moment'][ik, :, 2, 0]
        assert atomic[0] == pytest.approx(2.0 if name == 'K' else -2.0) and atomic[
            1
        ] == pytest.approx(0.0, abs=1e-12)
        total = bloch - atomic  # the atomic moment of an electron is -mu_B L
        g[name] = -2.0 * (total[1] - total[0])
    assert g['K'] == pytest.approx(-g["K'"])
    assert g['K'] == pytest.approx(-3.194, abs=2e-3)  # atomic -4, Bloch +0.81 (measured: about -4)
