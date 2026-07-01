"""Unit tests for the Stage 4 quasi-harmonic approximation (QHA) driver.

These exercise :mod:`PAOFLOW.phonon.do_qha` (volume-scan bookkeeping, QE input
generation, the EOS and parabolic QHA back-ends and the output writer) plus the
supporting helpers (``parse_qe_total_energy`` and the scaled structure bridge)
without any Quantum ESPRESSO runtime -- energies and thermal properties are
fabricated so the thermodynamics can be validated directly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon import do_qha
from PAOFLOW.phonon.do_qha import (
    RY_TO_EV,
    _qha_via_phonopy,
    _qha_via_quadratic,
    _volume_scales,
    generate_qha_inputs,
    run_qha,
)
from PAOFLOW.phonon.io import parse_qe_total_energy
from PAOFLOW.phonon.structure import paoflow_to_phonopy


class _StubController:
    """Minimal ``DataController`` stand-in exposing ``data_dicts()``."""

    def __init__(self, arry, attr, rank=0):
        self._arry = arry
        self._attr = attr
        self.rank = rank

    def data_dicts(self):
        return self._arry, self._attr


def _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05):
    """Two-atom fcc silicon controller in PAOFLOW (Bohr/alat) conventions."""
    alat = 10.20  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    cell_bohr = a_vectors * alat
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=float) @ cell_bohr
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': ['Si', 'Si'],
        'species': [('Si', 'Si_ONCV_PBE_sr.UPF')],
    }
    attr = {
        'alat': alat,
        'natoms': 2,
        'omega': omega,
        'opath': str(tmp_path),
        'savedir': 'silicon.save',
        'fpath': str(tmp_path),
        'ecutwfc': 40.0,
        'ecutrho': 160.0,
        'nspin': 1,
        'insulator': True,
        'nk1': 4,
        'nk2': 4,
        'nk3': 4,
        'phonon_supercell_matrix': supercell_matrix,
        'phonon_displacement_distance': distance,
        'verbose': False,
    }
    return _StubController(arry, attr)


# --------------------------------------------------------------------------- #
# volume-scan bookkeeping
# --------------------------------------------------------------------------- #
def test_volume_scales_five_points_symmetric():
    scales = _volume_scales(5, 0.02)
    assert len(scales) == 5
    np.testing.assert_allclose(scales, [0.98, 0.99, 1.0, 1.01, 1.02])
    assert scales[2] == pytest.approx(1.0)


def test_volume_scales_three_points():
    scales = _volume_scales(3, 0.02)
    np.testing.assert_allclose(scales, [0.98, 1.0, 1.02])


@pytest.mark.parametrize('nvol', [1, 2, 4, 6])
def test_volume_scales_rejects_unsupported(nvol):
    with pytest.raises(ValueError):
        _volume_scales(nvol, 0.02)


# --------------------------------------------------------------------------- #
# QE energy parser
# --------------------------------------------------------------------------- #
def test_parse_qe_total_energy_returns_last_converged(tmp_path):
    out = tmp_path / 'scf.out'
    out.write_text(
        'iteration #  1\n'
        '     total energy              =     -15.00000000 Ry\n'
        'End of self-consistent calculation\n'
        '!    total energy              =     -15.83729100 Ry\n'
        '     convergence has been achieved\n'
    )
    assert parse_qe_total_energy(str(out)) == pytest.approx(-15.837291)


def test_parse_qe_total_energy_raises_when_missing(tmp_path):
    out = tmp_path / 'scf.out'
    out.write_text('no energy here\n')
    with pytest.raises(ValueError):
        parse_qe_total_energy(str(out))


# --------------------------------------------------------------------------- #
# scaled structure bridge
# --------------------------------------------------------------------------- #
def test_paoflow_to_phonopy_scale_preserves_fractional_and_scales_volume(tmp_path):
    dc = _silicon_controller(tmp_path)
    base = paoflow_to_phonopy(dc, scale=1.0)
    scaled = paoflow_to_phonopy(dc, scale=1.02)
    # Volume scales as scale**3; fractional positions are preserved.
    assert scaled.volume == pytest.approx(base.volume * 1.02**3)
    np.testing.assert_allclose(scaled.scaled_positions, base.scaled_positions)


# --------------------------------------------------------------------------- #
# input generation
# --------------------------------------------------------------------------- #
def test_generate_qha_inputs_writes_per_volume_files(tmp_path):
    dc = _silicon_controller(tmp_path, supercell_matrix=2)
    dirs = generate_qha_inputs(dc, nvolumes=3, strain=0.02, qha_dir='qha')
    assert len(dirs) == 3
    for i in range(3):
        vol_dir = tmp_path / 'qha' / ('vol-%02d' % i)
        assert (vol_dir / 'scf.in').is_file()
        supercells = list(vol_dir.glob('supercell-*.in'))
        assert supercells, 'expected displaced-supercell inputs in %s' % vol_dir


# --------------------------------------------------------------------------- #
# QHA back-ends
# --------------------------------------------------------------------------- #
def _synthetic_data(nvol, strain=0.02, ntemp=7, tmax=300.0):
    """E(V) parabola + a volume-softening F_vib giving positive expansion."""
    scales = _volume_scales(nvol, strain)
    v0 = 40.0
    volumes = v0 * scales**3  # Angstrom^3
    ve = volumes[len(volumes) // 2]
    ke = 0.02  # eV/Angstrom^6
    energies = ke * (volumes - ve) ** 2 - 100.0  # eV
    temps = np.linspace(0.0, tmax, ntemp)
    # F_vib slopes down with volume ever more strongly as T rises -> V*(T) grows.
    fvib = 10.0 - 0.001 * np.outer(temps, volumes)  # kJ/mol, (ntemp, nvol)
    cv = np.full((ntemp, nvol), 25.0)  # J/K/mol
    entropy = 0.05 * np.outer(temps, np.ones(nvol))  # J/K/mol
    return {
        'volumes': volumes,
        'energies': energies,
        'temperatures': temps,
        'free_energy': fvib,
        'cv': cv,
        'entropy': entropy,
    }


def test_qha_via_quadratic_positive_expansion_and_bulk_modulus():
    data = _synthetic_data(3)
    result = _qha_via_quadratic(data, pressure=0.0)
    assert result['temperatures'].size > 0
    # Volume grows with temperature -> positive thermal expansion, positive B.
    assert np.all(result['thermal_expansion'] > 0.0)
    assert np.all(result['bulk_modulus'] > 0.0)
    assert np.all(np.diff(result['volume']) > 0.0)
    assert np.all(np.isfinite(result['gruneisen']))


def test_qha_via_phonopy_runs_and_returns_aligned_arrays():
    data = _synthetic_data(5)
    qha, result = _qha_via_phonopy(data, eos='vinet', pressure=0.0, t_max=300.0)
    n = result['temperatures'].size
    assert n > 0
    for key in (
        'volume',
        'thermal_expansion',
        'bulk_modulus',
        'gibbs',
        'heat_capacity',
        'gruneisen',
    ):
        assert result[key].size == n
    assert result['B0'] > 0.0


# --------------------------------------------------------------------------- #
# end-to-end analysis (QE harvesting monkeypatched out)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('nvol', [3, 5])
def test_run_qha_writes_all_output_tables(tmp_path, monkeypatch, nvol):
    dc = _silicon_controller(tmp_path)
    data = _synthetic_data(nvol)

    def _fake_collect(*args, **kwargs):
        return data

    monkeypatch.setattr(do_qha, '_collect_volume_data', _fake_collect)

    run_qha(dc, nvolumes=nvol, qha_dir='qha', t_min=0.0, t_max=300.0, t_step=50.0, fname='qha')

    for suffix in (
        '_ev',
        '_volume',
        '_thermal_expansion',
        '_bulk_modulus',
        '_gibbs',
        '_heat_capacity',
        '_gruneisen',
    ):
        path = tmp_path / ('qha%s.dat' % suffix)
        assert path.is_file(), 'missing %s' % path
        # Every table has at least one data row.
        rows = [ln for ln in path.read_text().splitlines() if not ln.startswith('#')]
        assert rows, 'empty table %s' % path


def test_ry_to_ev_constant_is_reasonable():
    # Guard against an accidental edit of the conversion factor.
    assert RY_TO_EV == pytest.approx(13.605693, abs=1e-4)
    assert os.path.basename(do_qha.__file__) == 'do_qha.py'


# --------------------------------------------------------------------------- #
# mode Gruneisen dispersion
# --------------------------------------------------------------------------- #
def _phonons_with_forces(dc, scales, seed=0):
    """Build phonopy objects at the given scales with fabricated fc2."""
    from PAOFLOW.phonon.do_phonopy import generate_displacements, produce_force_constants
    from PAOFLOW.phonon.do_qha import _init_phonopy_scaled

    rng = np.random.default_rng(seed)
    phonons = []
    forces = None
    for scale in scales:
        phonon = _init_phonopy_scaled(dc, scale)
        generate_displacements(dc)
        nsuper = len(phonon.supercell)
        ndisp = len(phonon.supercells_with_displacements)
        if forces is None:
            forces = rng.standard_normal((ndisp, nsuper, 3))
            forces -= forces.mean(axis=1, keepdims=True)
        produce_force_constants(dc, forces=forces)
        phonons.append(phonon)
    return phonons


def test_compute_gruneisen_band_writes_dispersion(tmp_path):
    dc = _silicon_controller(tmp_path, supercell_matrix=2)
    scales = _volume_scales(3, 0.02)
    phonons = _phonons_with_forces(dc, scales)

    path = do_qha.compute_gruneisen_band(
        dc,
        phonons,
        q_path=[[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]],
        q_labels=['G', 'X'],
        npoints=11,
        fname='qha',
    )
    assert path is not None and os.path.isfile(path)

    data = np.loadtxt(path)
    nbranch = 3 * dc._attr['natoms']
    # distance + [gruneisen, frequency] per branch.
    assert data.shape[1] == 1 + 2 * nbranch
    assert data.shape[0] == 11
    # A labels file with the two tick marks is written alongside.
    assert os.path.isfile(str(tmp_path / 'qha_gruneisen_band.labels'))


def test_compute_gruneisen_band_needs_three_volumes(tmp_path):
    dc = _silicon_controller(tmp_path)
    assert do_qha.compute_gruneisen_band(dc, None) is None
    assert do_qha.compute_gruneisen_band(dc, [object()]) is None


def test_compute_gruneisen_band_uses_ibrav_path(tmp_path):
    # No explicit q_path: the path must be derived from ibrav, exactly like the
    # phonon dispersion (do_phonopy.default_q_path).
    dc = _silicon_controller(tmp_path, supercell_matrix=2)
    dc._attr['ibrav'] = 2  # fcc
    phonons = _phonons_with_forces(dc, _volume_scales(3, 0.02))

    path = do_qha.compute_gruneisen_band(dc, phonons, q_path=None, npoints=11, fname='qha')
    assert path is not None and os.path.isfile(path)
    assert os.path.isfile(str(tmp_path / 'qha_gruneisen_band.labels'))


def test_compute_gruneisen_band_cutoff_masks_low_frequency_modes(tmp_path):
    dc = _silicon_controller(tmp_path, supercell_matrix=2)
    phonons = _phonons_with_forces(dc, _volume_scales(3, 0.02))
    qpath = [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]]

    # A huge cutoff masks every mode -> all gruneisen columns are nan.
    p_masked = do_qha.compute_gruneisen_band(
        dc, phonons, q_path=qpath, npoints=11, cutoff_frequency=1.0e9, fname='qha_masked'
    )
    masked = np.loadtxt(p_masked)
    assert np.isnan(masked[:, 1::2]).all()

    # Disabling the cutoff keeps every gruneisen value finite.
    p_full = do_qha.compute_gruneisen_band(
        dc, phonons, q_path=qpath, npoints=11, cutoff_frequency=0.0, fname='qha_full'
    )
    full = np.loadtxt(p_full)
    assert np.isfinite(full[:, 1::2]).all()
