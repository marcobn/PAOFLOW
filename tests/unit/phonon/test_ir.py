"""Unit tests for the Stage 3 infrared-spectrum machinery.

Exercises the mode effective-charge contraction
(:func:`PAOFLOW.phonon.do_ir_raman.mode_effective_charges`) with analytic
inputs, and the end-to-end :func:`compute_ir_spectrum` on a two-atom cell whose
harmonic force constants are set explicitly, without any Quantum ESPRESSO or
finite-displacement runtime.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

_trapz = getattr(np, 'trapezoid', np.trapz)

from PAOFLOW.phonon.do_ir_raman import (
    _lorentzian,
    compute_ir_spectrum,
    mode_effective_charges,
)
from PAOFLOW.phonon.do_phonopy import init_phonopy


class _StubController:
    """Minimal ``DataController`` stand-in exposing ``data_dicts()``."""

    def __init__(self, arry, attr, rank=0):
        self._arry = arry
        self._attr = attr
        self.rank = rank

    def data_dicts(self):
        return self._arry, self._attr


def _diatomic_controller(tmp_path, supercell_matrix=1, masses=None):
    """Two-atom fcc controller (one primitive cell when supercell_matrix=1)."""
    alat = 10.6829  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    cell_bohr = a_vectors * alat
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=float) @ cell_bohr
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': ['Ga', 'As'],
        'species': [('Ga', 'Ga_ONCV_sr.upf'), ('As', 'As_ONCV_sr.upf')],
    }
    attr = {
        'alat': alat,
        'natoms': 2,
        'omega': omega,
        'opath': str(tmp_path),
        'savedir': 'GaAs.save',
        'fpath': str(tmp_path),
        'nspin': 1,
        'insulator': True,
        'nk1': 4,
        'nk2': 4,
        'nk3': 4,
        'phonon_supercell_matrix': supercell_matrix,
        'phonon_displacement_distance': 0.06,
        'verbose': False,
    }
    dc = _StubController(arry, attr)
    if masses is not None:
        arry['_force_masses'] = masses
    return dc


# ---------------------------------------------------------------------------
# Mode effective-charge contraction (pure physics, no phonopy dynamics).
# ---------------------------------------------------------------------------


def test_mode_effective_charges_optical_diatomic():
    """Out-of-phase optical mode of a +q/-q diatomic is IR-active."""
    q = 1.7
    born = np.array([np.eye(3) * q, np.eye(3) * (-q)])
    masses = np.array([1.0, 1.0])

    # Optical eigenvector along x: atoms move out of phase, unit-normalised.
    e_opt = np.zeros((6, 1))
    e_opt[0, 0] = 1.0 / np.sqrt(2.0)
    e_opt[3, 0] = -1.0 / np.sqrt(2.0)

    zbar, intensity = mode_effective_charges(born, e_opt, masses)
    # Zbar_x = q/sqrt2 - (-q)(-1/sqrt2) ... = q*sqrt2 ; |Zbar|^2 = 2 q^2.
    np.testing.assert_allclose(zbar[0, 0].real, q * np.sqrt(2.0), atol=1e-12)
    assert intensity[0] == pytest.approx(2.0 * q**2, abs=1e-10)


def test_mode_effective_charges_acoustic_is_silent():
    """In-phase acoustic translation carries no dipole (zero intensity)."""
    q = 2.0
    born = np.array([np.eye(3) * q, np.eye(3) * (-q)])
    masses = np.array([1.0, 1.0])

    e_ac = np.zeros((6, 1))
    e_ac[0, 0] = 1.0 / np.sqrt(2.0)
    e_ac[3, 0] = 1.0 / np.sqrt(2.0)

    _, intensity = mode_effective_charges(born, e_ac, masses)
    assert intensity[0] == pytest.approx(0.0, abs=1e-12)


def test_mode_effective_charges_zero_born_is_inactive():
    """A nonpolar crystal (Z* = 0, e.g. Si) has no infrared activity."""
    born = np.zeros((2, 3, 3))
    masses = np.array([1.0, 1.0])
    rng = np.random.default_rng(0)
    eig = rng.standard_normal((6, 6))

    _, intensity = mode_effective_charges(born, eig, masses)
    np.testing.assert_allclose(intensity, 0.0, atol=1e-14)


def test_mode_effective_charges_mass_weighting():
    """Heavier atoms contribute less to the dipole (1/sqrt(M) weighting)."""
    q = 1.0
    born = np.array([np.eye(3) * q, np.eye(3) * (-q)])
    masses = np.array([4.0, 1.0])

    e = np.zeros((6, 1))
    e[0, 0] = 1.0  # only the heavy atom moves along x
    zbar, _ = mode_effective_charges(born, e, masses)
    # Zbar_x = q * 1 / sqrt(4) = q / 2.
    assert zbar[0, 0].real == pytest.approx(q / 2.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Lorentzian broadening.
# ---------------------------------------------------------------------------


def test_lorentzian_is_normalised():
    grid = np.linspace(-5000.0, 5000.0, 400001)
    y = _lorentzian(grid, 0.0, 4.0)
    integral = _trapz(y, grid)
    assert integral == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# End-to-end compute_ir_spectrum with explicit force constants.
# ---------------------------------------------------------------------------


def _set_diatomic_force_constants(phonon, k=1.0):
    """Attach a simple two-atom spring model satisfying the acoustic sum rule."""
    natom = len(phonon.supercell)
    fc = np.zeros((natom, natom, 3, 3))
    eye = np.eye(3)
    for i in range(natom):
        for j in range(natom):
            fc[i, j] = (k * eye) if i == j else (-k * eye / (natom - 1))
    phonon.force_constants = fc


def test_compute_ir_spectrum_end_to_end(tmp_path):
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    q = 2.0
    born = np.array([np.eye(3) * q, np.eye(3) * (-q)])
    arry, _ = dc.data_dicts()
    arry['born_charges'] = born

    res = compute_ir_spectrum(dc, freq_min=0.0, freq_max=None, gamma=4.0, fname='t')

    freqs = res['frequencies']
    intens = res['intensities']
    active = res['active']

    # Three acoustic branches at ~0 (silent) + three optical branches (active).
    order = np.argsort(freqs)
    assert np.allclose(freqs[order][:3], 0.0, atol=1e-4)
    np.testing.assert_allclose(intens[order][:3], 0.0, atol=1e-8)
    assert active[order][3:].all()
    assert not active[order][:3].any()

    # Output files were written with the expected names.
    assert os.path.isfile(os.path.join(str(tmp_path), 't_ir_modes.dat'))
    spath = os.path.join(str(tmp_path), 't_ir_spectrum.dat')
    assert os.path.isfile(spath)

    # The broadened spectrum integrates to the total active intensity.
    grid, spectrum = res['spectrum']
    integral = _trapz(spectrum, grid)
    optical_total = intens[order][3:].sum()
    assert integral == pytest.approx(optical_total, rel=0.02)


def test_compute_ir_spectrum_requires_born(tmp_path):
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    with pytest.raises(ValueError, match='Born effective charges'):
        compute_ir_spectrum(dc, write=False)
