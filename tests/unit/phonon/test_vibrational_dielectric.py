"""Unit tests for the Stage 5 vibrational (ionic) dielectric machinery.

Exercises :func:`PAOFLOW.phonon.do_vibrational_dielectric.compute_vibrational_dielectric`
on a two-atom polar cell whose harmonic force constants and Born charges are set
explicitly, without any Quantum ESPRESSO or finite-displacement runtime.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon.do_phonopy import init_phonopy
from PAOFLOW.phonon.do_vibrational_dielectric import compute_vibrational_dielectric

sys.path.insert(0, os.path.dirname(__file__))
from test_ir import _diatomic_controller, _set_diatomic_force_constants


def _polar_controller(tmp_path, q=2.0, eps_inf=None):
    """Diatomic controller with a spring model and +q/-q Born charges set."""
    dc = _diatomic_controller(tmp_path, supercell_matrix=1)
    phonon = init_phonopy(dc)
    _set_diatomic_force_constants(phonon, k=1.0)
    arry, _ = dc.data_dicts()
    arry['born_charges'] = np.array([np.eye(3) * q, np.eye(3) * (-q)])
    arry['dielectric_tensor'] = (np.eye(3) * 5.0) if eps_inf is None else np.asarray(eps_inf)
    return dc


# ---------------------------------------------------------------------------
# Static limit and self-consistency.
# ---------------------------------------------------------------------------


def test_static_limit_matches_zero_frequency():
    """The omega=0 grid point reproduces the static dielectric tensor."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _polar_controller(tmp, q=2.0)
        res = compute_vibrational_dielectric(dc, freq_min=0.0, npoints=501, gamma=4.0, write=False)
    eps = res['eps']
    static = res['static']
    # grid starts at 0 -> eps(0) is real and equals the static tensor.
    np.testing.assert_allclose(np.imag(eps[0]), 0.0, atol=1e-10)
    np.testing.assert_allclose(np.real(eps[0]), static, rtol=1e-8, atol=1e-8)


def test_static_equals_inf_plus_ionic():
    """eps(0) = eps_inf + sum_v S_v / omega_v^2 (generalized LST)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _polar_controller(tmp, q=2.0, eps_inf=np.eye(3) * 5.0)
        res = compute_vibrational_dielectric(dc, write=False)
    static = res['static']
    eps_inf = res['eps_inf']
    ionic = static - eps_inf
    # Polar crystal: a positive ionic contribution on the diagonal.
    assert np.all(np.diag(ionic) > 0.0)
    # Off-diagonal stays ~0 for the isotropic +q/-q model.
    off = ionic - np.diag(np.diag(ionic))
    np.testing.assert_allclose(off, 0.0, atol=1e-8)


# ---------------------------------------------------------------------------
# Reststrahlen band (Re eps < 0 between omega_TO and omega_LO).
# ---------------------------------------------------------------------------


def test_reststrahlen_band_exists():
    """A polar crystal has a band where Re eps < 0 above the TO frequency."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _polar_controller(tmp, q=3.0)
        res = compute_vibrational_dielectric(dc, freq_min=0.0, npoints=4000, gamma=1.0, write=False)
    re_xx = np.real(res['eps'][:, 0, 0])
    assert np.any(re_xx < 0.0)


def test_zero_gamma_static_real():
    """With small damping the static value stays close to the LST result."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _polar_controller(tmp, q=2.0)
        res = compute_vibrational_dielectric(dc, freq_min=0.0, npoints=501, gamma=0.5, write=False)
    np.testing.assert_allclose(np.real(res['eps'][0]), res['static'], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# Nonpolar crystal (Z* = 0): no ionic contribution.
# ---------------------------------------------------------------------------


def test_nonpolar_is_flat():
    """Si-like Z* = 0 leaves eps(omega) = eps_inf at every frequency."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _diatomic_controller(tmp, supercell_matrix=1)
        phonon = init_phonopy(dc)
        _set_diatomic_force_constants(phonon, k=1.0)
        arry, _ = dc.data_dicts()
        arry['born_charges'] = np.zeros((2, 3, 3))
        arry['dielectric_tensor'] = np.eye(3) * 12.0
        res = compute_vibrational_dielectric(dc, write=False)
    eps_inf = res['eps_inf']
    eps = res['eps']
    np.testing.assert_allclose(np.real(eps), np.broadcast_to(eps_inf, eps.shape), atol=1e-10)
    np.testing.assert_allclose(np.imag(eps), 0.0, atol=1e-10)
    np.testing.assert_allclose(res['static'], eps_inf, atol=1e-10)


# ---------------------------------------------------------------------------
# Acoustic modes do not contribute (no division blow-up).
# ---------------------------------------------------------------------------


def test_acoustic_modes_excluded():
    """Near-zero acoustic frequencies must not diverge in eps(omega)."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _polar_controller(tmp, q=2.0)
        res = compute_vibrational_dielectric(dc, freq_min=0.0, npoints=501, write=False)
    assert np.all(np.isfinite(res['eps']))
    assert np.all(np.isfinite(res['static']))


# ---------------------------------------------------------------------------
# Requirements / error handling.
# ---------------------------------------------------------------------------


def test_requires_born_charges():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _diatomic_controller(tmp, supercell_matrix=1)
        phonon = init_phonopy(dc)
        _set_diatomic_force_constants(phonon, k=1.0)
        arry, _ = dc.data_dicts()
        arry['dielectric_tensor'] = np.eye(3) * 5.0
        with pytest.raises(ValueError, match='Born effective charges'):
            compute_vibrational_dielectric(dc, write=False)


def test_requires_dielectric():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        dc = _diatomic_controller(tmp, supercell_matrix=1)
        phonon = init_phonopy(dc)
        _set_diatomic_force_constants(phonon, k=1.0)
        arry, _ = dc.data_dicts()
        arry['born_charges'] = np.array([np.eye(3) * 2.0, np.eye(3) * (-2.0)])
        with pytest.raises(ValueError, match='dielectric tensor'):
            compute_vibrational_dielectric(dc, write=False)


# ---------------------------------------------------------------------------
# Output files are readable by plot_optical (read_dos_PAO).
# ---------------------------------------------------------------------------


def test_output_files_written_and_readable(tmp_path):
    from PAOFLOW.inputs.read_pao_output import read_dos_PAO

    dc = _polar_controller(str(tmp_path), q=2.0)
    compute_vibrational_dielectric(dc, freq_min=0.0, npoints=400, outdir='vibdielectric', fname='t')

    outdir = os.path.join(str(tmp_path), 'vibdielectric')
    for tag in ('epsr', 'epsi', 'eels', 'refl'):
        for comp in ('xx', 'yy', 'zz', 'xy', 'xz', 'yz'):
            fn = os.path.join(outdir, '%s_%s.dat' % (tag, comp))
            assert os.path.isfile(fn)
    # read_dos_PAO has no comment handling: the files must be header-free.
    x, y = read_dos_PAO(os.path.join(outdir, 'epsr_xx.dat'))
    assert x.shape == y.shape and x.size == 400
    # Frequency axis emitted in eV by default (phonon range ~< 0.2 eV).
    assert x.max() < 1.0

    assert os.path.isfile(os.path.join(str(tmp_path), 't_vibdielectric_static.dat'))


def test_emit_units_axis(tmp_path):
    """emit_ev=False writes the frequency axis in the requested units."""
    from PAOFLOW.inputs.read_pao_output import read_dos_PAO

    dc = _polar_controller(str(tmp_path), q=2.0)
    compute_vibrational_dielectric(
        dc,
        freq_min=0.0,
        freq_max=500.0,
        npoints=200,
        units='cm-1',
        emit_ev=False,
        outdir='vib_cm',
        fname='t',
    )
    x, _ = read_dos_PAO(os.path.join(str(tmp_path), 'vib_cm', 'epsr_xx.dat'))
    assert x.max() == pytest.approx(500.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Reststrahlen (phonon) emissivity.
# ---------------------------------------------------------------------------


def test_emissivity_not_computed_by_default(tmp_path):
    """Without emissivity=True no emissivity is returned or written."""
    dc = _polar_controller(str(tmp_path), q=2.0)
    res = compute_vibrational_dielectric(dc, freq_min=0.0, npoints=200, outdir='vib')
    assert res['emissivity'] is None
    assert not os.path.isfile(os.path.join(str(tmp_path), 'vib', 'emish_xx.dat'))


def test_emissivity_files_written_and_bounded(tmp_path):
    """emissivity=True writes directional/hemispherical/total files in [0, 1]."""
    from PAOFLOW.inputs.read_pao_output import read_dos_PAO

    dc = _polar_controller(str(tmp_path), q=2.0)
    res = compute_vibrational_dielectric(
        dc,
        freq_min=0.0,
        npoints=400,
        outdir='vib',
        fname='t',
        emissivity=True,
        emis_angles=(0.0, 60.0),
        emis_ntheta=24,
        emis_temperature=(300.0, 600.0),
    )
    emis = res['emissivity']
    assert emis is not None

    outdir = os.path.join(str(tmp_path), 'vib')
    # Spectral hemispherical + per-angle directional + total files for diagonals.
    for comp in ('xx', 'yy', 'zz'):
        assert os.path.isfile(os.path.join(outdir, 'emish_%s.dat' % comp))
        assert os.path.isfile(os.path.join(outdir, 'emist_%s.dat' % comp))
        for deg in (0, 60):
            assert os.path.isfile(os.path.join(outdir, 'emis_th%d_%s.dat' % (deg, comp)))
            assert os.path.isfile(os.path.join(outdir, 'refl_th%d_%s.dat' % (deg, comp)))

    # Emissivity (= 1 - R) is bounded in [0, 1] and header-free.
    x, e = read_dos_PAO(os.path.join(outdir, 'emish_xx.dat'))
    assert x.size == 400
    assert np.all(e >= -1e-9) and np.all(e <= 1.0 + 1e-9)
    # Returned arrays carry the diagonal components and temperatures.
    assert emis['hemispherical'].shape == (400, 3)
    assert emis['total'].shape == (2, 3)
    assert np.all(emis['total'] >= -1e-9) and np.all(emis['total'] <= 1.0 + 1e-9)


def test_emissivity_reststrahlen_dip(tmp_path):
    """Inside the reststrahlen band the emissivity drops (R -> 1)."""
    dc = _polar_controller(str(tmp_path), q=2.0)
    res = compute_vibrational_dielectric(
        dc, freq_min=0.0, npoints=600, emissivity=True, write=False
    )
    emis = res['emissivity']
    emis_xx = emis['hemispherical'][:, 0]
    eps_xx = np.real(res['eps'][:, 0, 0])
    # Where Re eps < 0 (reststrahlen) the emissivity is strongly suppressed
    # relative to the high-frequency tail (eps -> eps_inf, partially absorbing).
    band = eps_xx < 0.0
    assert np.any(band)
    tail = emis_xx[-20:].mean()
    assert emis_xx[band].min() < 0.5 * tail
