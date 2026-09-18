"""Unit tests for the Stage 4 Raman-spectrum machinery.

Exercises the rotational invariants and finite-difference Raman-tensor
construction (:mod:`PAOFLOW.phonon.do_ir_raman`) with analytic inputs, and the
end-to-end :func:`compute_raman_spectrum` on a two-atom cell whose harmonic
force constants are set explicitly and whose displaced-cell dielectric tensors
are supplied synthetically -- no Quantum ESPRESSO or PAOFLOW optical runtime.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

_trapz = getattr(np, 'trapezoid', np.trapz)

from PAOFLOW.phonon.do_ir_raman import (
    _bose_factor,
    compute_raman_spectrum,
    mode_displacement_vectors,
    raman_invariants,
    raman_powder_activity,
    raman_tensors_from_epsilon,
)
from PAOFLOW.phonon.do_phonopy import init_phonopy

sys.path.insert(0, os.path.dirname(__file__))
from test_ir import _diatomic_controller, _set_diatomic_force_constants

# ---------------------------------------------------------------------------
# Rotational invariants (pure algebra).
# ---------------------------------------------------------------------------


def test_invariants_isotropic_tensor():
    """An isotropic tensor a*I has anisotropy gamma2 = 0."""
    a0 = 1.7
    a, gamma2 = raman_invariants(np.eye(3) * a0)
    assert a == pytest.approx(a0, abs=1e-12)
    assert gamma2 == pytest.approx(0.0, abs=1e-12)
    assert raman_powder_activity(np.eye(3) * a0) == pytest.approx(45.0 * a0 * a0, abs=1e-10)


def test_invariants_diagonal_anisotropic():
    """diag(1,2,3): a = 2, gamma2 = 3, activity = 45*4 + 7*3 = 201."""
    r = np.diag([1.0, 2.0, 3.0])
    a, gamma2 = raman_invariants(r)
    assert a == pytest.approx(2.0, abs=1e-12)
    assert gamma2 == pytest.approx(3.0, abs=1e-12)
    assert raman_powder_activity(r) == pytest.approx(201.0, abs=1e-10)


def test_invariants_pure_offdiagonal():
    """A symmetric off-diagonal tensor has zero mean and gamma2 = 3*Rxy^2."""
    r = np.zeros((3, 3))
    r[0, 1] = r[1, 0] = 1.3
    a, gamma2 = raman_invariants(r)
    assert a == pytest.approx(0.0, abs=1e-12)
    assert gamma2 == pytest.approx(3.0 * 1.3**2, abs=1e-10)


def test_invariants_symmetrise_input():
    """The non-symmetric part of the input is averaged away."""
    r = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a, gamma2 = raman_invariants(r)
    # Symmetrised Rxy = 1.0, mean a = 1.
    assert a == pytest.approx(1.0, abs=1e-12)
    assert gamma2 == pytest.approx(3.0 * 1.0**2, abs=1e-10)


# ---------------------------------------------------------------------------
# Finite-difference Raman tensor.
# ---------------------------------------------------------------------------


def test_raman_tensor_central_difference():
    delta = 0.05
    eps_plus = np.zeros((2, 3, 3))
    eps_minus = np.zeros((2, 3, 3))
    eps_plus[0] = np.eye(3) * 0.1
    eps_minus[0] = -np.eye(3) * 0.1
    r = raman_tensors_from_epsilon(eps_plus, eps_minus, delta)
    # (0.1 - (-0.1)) / (2*0.05) = 2 along the diagonal.
    np.testing.assert_allclose(r[0], np.eye(3) * 2.0, atol=1e-12)
    np.testing.assert_allclose(r[1], 0.0, atol=1e-12)


def test_raman_tensor_symmetrised():
    delta = 1.0
    eps_plus = np.zeros((1, 3, 3))
    eps_minus = np.zeros((1, 3, 3))
    eps_plus[0, 0, 1] = 2.0  # asymmetric input -> R[0,1]=1 before symmetrising
    r = raman_tensors_from_epsilon(eps_plus, eps_minus, delta)
    # 0.5*(R + R^T) splits the single off-diagonal entry symmetrically.
    assert r[0, 0, 1] == pytest.approx(0.5, abs=1e-12)
    assert r[0, 1, 0] == pytest.approx(0.5, abs=1e-12)


# ---------------------------------------------------------------------------
# Mass-weighted displacement vectors.
# ---------------------------------------------------------------------------


def test_mode_displacement_mass_weighting():
    """u_k = delta * e_k / sqrt(M_k) reshaped to (natom, 3)."""
    masses = np.array([4.0, 1.0])
    e = np.zeros((6, 1))
    e[0, 0] = 1.0  # heavy atom moves along x
    delta = 0.1
    disp = mode_displacement_vectors(e, masses, delta)
    assert disp.shape == (1, 2, 3)
    assert disp[0, 0, 0] == pytest.approx(delta / np.sqrt(4.0), abs=1e-12)
    np.testing.assert_allclose(disp[0, 1], 0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# Bose-Einstein Stokes factor.
# ---------------------------------------------------------------------------


def test_bose_factor_limits():
    assert _bose_factor(100.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    # High-frequency / low-T mode is essentially unoccupied.
    assert _bose_factor(2000.0, 10.0) == pytest.approx(0.0, abs=1e-6)
    # n increases with temperature.
    assert _bose_factor(100.0, 600.0) > _bose_factor(100.0, 300.0)


# ---------------------------------------------------------------------------
# End-to-end compute_raman_spectrum with synthetic dielectric data.
# ---------------------------------------------------------------------------


def test_compute_raman_spectrum_end_to_end(tmp_path):
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    qpts = phonon.run_qpoints([[0.0, 0.0, 0.0]], with_eigenvectors=True)
    freqs = np.asarray(qpts.frequencies)[0]
    order = np.argsort(freqs)
    nmodes = freqs.shape[0]

    delta = 0.05
    eps_plus = np.zeros((nmodes, 3, 3))
    eps_minus = np.zeros((nmodes, 3, 3))
    computed = np.zeros(nmodes, dtype=bool)
    # Only the three optical branches are displaced; isotropic +/- response.
    for v in order[3:]:
        eps_plus[v] = np.eye(3) * 0.1
        eps_minus[v] = -np.eye(3) * 0.1
        computed[v] = True

    res = compute_raman_spectrum(
        dc, eps_plus, eps_minus, delta, computed=computed, gamma=4.0, fname='t'
    )

    activities = res['activities']
    intens = res['intensities']
    active = res['active']

    # Optical branches active, acoustic branches silent.
    assert active[order][3:].all()
    assert not active[order][:3].any()
    np.testing.assert_allclose(activities[order][:3], 0.0, atol=1e-12)

    # Isotropic Raman tensor R = 2*I -> a = 2, gamma2 = 0, activity = 180.
    np.testing.assert_allclose(activities[order][3:], 180.0, rtol=1e-6)

    # Output files were written with the expected names.
    assert os.path.isfile(os.path.join(str(tmp_path), 't_raman_modes.dat'))
    spath = os.path.join(str(tmp_path), 't_raman_spectrum.dat')
    assert os.path.isfile(spath)

    # The broadened spectrum integrates to the total active intensity.
    grid, spectrum = res['spectrum']
    integral = _trapz(spectrum, grid)
    optical_total = intens[order][3:].sum()
    assert integral == pytest.approx(optical_total, rel=0.03)


def test_compute_raman_spectrum_non_computed_is_silent(tmp_path):
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    nmodes = 3 * len(phonon.primitive)
    eps_plus = np.zeros((nmodes, 3, 3))
    eps_minus = np.zeros((nmodes, 3, 3))
    # Non-zero dielectric difference, but no mode flagged as computed.
    eps_plus[:] = np.eye(3) * 0.1
    eps_minus[:] = -np.eye(3) * 0.1
    computed = np.zeros(nmodes, dtype=bool)

    res = compute_raman_spectrum(dc, eps_plus, eps_minus, 0.05, computed=computed, write=False)
    np.testing.assert_allclose(res['activities'], 0.0, atol=1e-12)
    assert not res['active'].any()


def test_compute_raman_spectrum_shape_validation(tmp_path):
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    with pytest.raises(ValueError, match='modes are expected'):
        compute_raman_spectrum(dc, np.zeros((2, 3, 3)), np.zeros((2, 3, 3)), 0.05, write=False)


# ---------------------------------------------------------------------------
# Resonance Raman: complex dielectric tensors.
# ---------------------------------------------------------------------------


def test_invariants_complex_reduce_to_real():
    """A complex tensor with zero imaginary part reproduces the real result."""
    r = np.diag([1.0, 2.0, 3.0]).astype(complex)
    a, gamma2 = raman_invariants(r)
    assert isinstance(a, complex)
    assert a.real == pytest.approx(2.0, abs=1e-12)
    assert a.imag == pytest.approx(0.0, abs=1e-12)
    assert gamma2 == pytest.approx(3.0, abs=1e-12)
    # Activity uses |a|^2 and is identical to the real case.
    assert raman_powder_activity(r) == pytest.approx(201.0, abs=1e-10)


def test_invariants_complex_uses_modulus():
    """For a complex isotropic tensor the activity scales as |a|^2."""
    a0 = 1.0 + 1.0j
    r = np.eye(3) * a0
    a, gamma2 = raman_invariants(r)
    assert a == pytest.approx(a0, abs=1e-12)
    assert gamma2 == pytest.approx(0.0, abs=1e-12)
    # |a|^2 = 2 -> activity = 45 * 2 = 90.
    assert raman_powder_activity(r) == pytest.approx(90.0, abs=1e-10)


def test_raman_tensor_complex_passthrough():
    """Complex dielectric differences yield a complex Raman tensor."""
    delta = 0.5
    eps_plus = np.zeros((1, 3, 3), dtype=complex)
    eps_minus = np.zeros((1, 3, 3), dtype=complex)
    eps_plus[0] = np.eye(3) * (0.2 + 0.4j)
    eps_minus[0] = -np.eye(3) * (0.2 + 0.4j)
    r = raman_tensors_from_epsilon(eps_plus, eps_minus, delta)
    assert np.iscomplexobj(r)
    # (0.2+0.4j - (-(0.2+0.4j))) / (2*0.5) = 0.4 + 0.8j on the diagonal.
    np.testing.assert_allclose(np.diag(r[0]), 0.4 + 0.8j, atol=1e-12)


def test_resonance_reduces_to_static_for_real_eps(tmp_path):
    """Real complex-typed dielectric data reproduces the static activities."""
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    nmodes = 3 * len(phonon.primitive)
    delta = 0.05
    eps_plus = np.zeros((nmodes, 3, 3), dtype=complex)
    eps_minus = np.zeros((nmodes, 3, 3), dtype=complex)
    computed = np.zeros(nmodes, dtype=bool)
    eps_plus[:] = np.eye(3) * 0.1
    eps_minus[:] = -np.eye(3) * 0.1
    computed[:] = True

    res = compute_raman_spectrum(dc, eps_plus, eps_minus, delta, computed=computed, write=False)
    # R = 2*I -> a = 2 (real), gamma2 = 0, activity = 180 for the optical modes
    # (acoustic branches stay silent) -- same as the static case.
    active = res['active']
    np.testing.assert_allclose(res['activities'][active], 180.0, rtol=1e-6)


def test_resonance_enhancement_near_pole(tmp_path):
    """A larger |eps| derivative (closer to resonance) raises the activity."""
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)

    nmodes = 3 * len(phonon.primitive)
    delta = 0.05

    def _activity(amp):
        eps_plus = np.zeros((nmodes, 3, 3), dtype=complex)
        eps_minus = np.zeros((nmodes, 3, 3), dtype=complex)
        computed = np.zeros(nmodes, dtype=bool)
        eps_plus[:] = np.eye(3) * amp
        eps_minus[:] = -np.eye(3) * amp
        computed[:] = True
        res = compute_raman_spectrum(dc, eps_plus, eps_minus, delta, computed=computed, write=False)
        return res['activities'][computed].max()

    off = _activity(0.1 + 0.05j)
    on = _activity(0.5 + 0.6j)
    assert on > off


# ---------------------------------------------------------------------------
# read_epsilon_at: complex harvest with interpolation.
# ---------------------------------------------------------------------------


def _write_eps_grid(eps_dir, energies, real_of, imag_of):
    os.makedirs(eps_dir, exist_ok=True)
    for comp, i, j in (
        ('xx', 0, 0),
        ('yy', 1, 1),
        ('zz', 2, 2),
        ('xy', 0, 1),
        ('xz', 0, 2),
        ('yz', 1, 2),
    ):
        np.savetxt(
            os.path.join(eps_dir, 'epsr_%s.dat' % comp),
            np.column_stack([energies, real_of(comp, energies)]),
        )
        np.savetxt(
            os.path.join(eps_dir, 'epsi_%s.dat' % comp),
            np.column_stack([energies, imag_of(comp, energies)]),
        )


def test_read_epsilon_at_interpolates(tmp_path):
    from PAOFLOW.phonon.io import read_epsilon_at, read_static_epsilon

    eps_dir = os.path.join(str(tmp_path), 'eps')
    energies = np.linspace(0.0, 4.0, 9)  # spacing 0.5 eV

    def real_of(comp, e):
        return np.full_like(e, 2.0) if comp == 'xx' else np.zeros_like(e)

    def imag_of(comp, e):
        return e if comp == 'xx' else np.zeros_like(e)

    _write_eps_grid(eps_dir, energies, real_of, imag_of)

    # Static limit -> real part only, imaginary part dropped.
    eps0 = read_static_epsilon(eps_dir)
    assert eps0.dtype == float or np.isrealobj(eps0)
    assert eps0[0, 0] == pytest.approx(2.0, abs=1e-12)

    # At 1.25 eV (between grid points) the imaginary part interpolates to 1.25.
    eps = read_epsilon_at(eps_dir, energy=1.25)
    assert np.iscomplexobj(eps)
    assert eps[0, 0].real == pytest.approx(2.0, abs=1e-12)
    assert eps[0, 0].imag == pytest.approx(1.25, abs=1e-12)
    # Tensor is symmetric.
    np.testing.assert_allclose(eps, eps.T, atol=1e-12)


def test_read_epsilon_at_missing_imag_is_zero(tmp_path):
    from PAOFLOW.phonon.io import read_epsilon_at

    eps_dir = os.path.join(str(tmp_path), 'eps')
    os.makedirs(eps_dir, exist_ok=True)
    energies = np.linspace(0.0, 2.0, 5)
    for comp in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz'):
        val = 3.0 if comp == 'yy' else 0.0
        np.savetxt(
            os.path.join(eps_dir, 'epsr_%s.dat' % comp),
            np.column_stack([energies, np.full_like(energies, val)]),
        )
    # No epsi files: the imaginary part defaults to zero without error.
    eps = read_epsilon_at(eps_dir, energy=1.0)
    assert eps[1, 1].real == pytest.approx(3.0, abs=1e-12)
    assert eps[1, 1].imag == pytest.approx(0.0, abs=1e-12)
