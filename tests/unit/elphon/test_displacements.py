"""Unit tests for the P0 electron-phonon scaffold (no QE runtime).

Exercises the reference-cell displacement bookkeeping and the two-phase QE input
generation using a single-atom fcc aluminium stub controller, mirroring the
``examples/new_qe_examples/phonon_examples/Al`` setup.
"""

import json
import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.elphon.displacements import (
    generate_eph_displacements,
    reference_supercell_atoms,
)
from PAOFLOW.elphon.do_elphon import generate_eph_inputs
from PAOFLOW.phonon.do_phonopy import init_phonopy


class _StubController:
    """Minimal ``DataController`` stand-in exposing ``data_dicts()``."""

    def __init__(self, arry, attr, rank=0):
        self._arry = arry
        self._attr = attr
        self.rank = rank

    def data_dicts(self):
        return self._arry, self._attr


def _aluminium_controller(tmp_path, supercell_matrix=2):
    """Single-atom fcc aluminium in PAOFLOW (Bohr/alat) conventions."""
    alat = 7.6326928726  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    tau = np.zeros((1, 3), dtype=float)
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': ['Al'],
        'species': [('Al', 'Al.upf')],
    }
    attr = {
        'alat': alat,
        'natoms': 1,
        'omega': omega,
        'opath': str(tmp_path),
        'savedir': 'Al1.save',
        'fpath': str(tmp_path),
        'ecutwfc': 44.0,
        'nspin': 1,
        'insulator': False,
        'degauss': 0.05,
        'nk1': 22,
        'nk2': 22,
        'nk3': 22,
        'phonon_supercell_matrix': supercell_matrix,
        'verbose': False,
    }
    return _StubController(arry, attr)


def test_reference_atoms_match_p2s_map(tmp_path):
    dc = _aluminium_controller(tmp_path)
    phonon = init_phonopy(dc)
    np.testing.assert_array_equal(
        reference_supercell_atoms(phonon),
        np.asarray(phonon.primitive.p2s_map, dtype=int),
    )


def test_symmetry_reduces_fcc_to_single_displacement(tmp_path):
    # fcc Al needs only ONE symmetry-inequivalent displacement (not 6).
    dc = _aluminium_controller(tmp_path)
    phonon = init_phonopy(dc)
    cells, meta = generate_eph_displacements(phonon, distance=0.06, is_plusminus='auto')

    assert len(cells) == len(meta) == 1
    m = meta[0]
    assert m['index'] == 0
    assert m['distance'] == pytest.approx(0.06)
    assert np.linalg.norm(m['displacement']) == pytest.approx(0.06, abs=1e-6)

    # The displaced cell moves exactly the recorded supercell atom by that vector.
    base = np.asarray(phonon.supercell.positions)
    diff = np.asarray(cells[0].positions) - base
    moved = np.argwhere(np.linalg.norm(diff, axis=1) > 1.0e-10)
    assert moved.shape == (1, 1)
    assert moved[0, 0] == m['sc_atom']
    np.testing.assert_allclose(diff[m['sc_atom']], m['displacement'], atol=1e-8)


def test_plusminus_adds_the_opposite_displacement(tmp_path):
    dc = _aluminium_controller(tmp_path)
    phonon = init_phonopy(dc)
    cells, meta = generate_eph_displacements(phonon, distance=0.06, is_plusminus=True)

    # Central differences: the symmetry-reduced set plus its explicit minus.
    assert len(cells) == len(meta) == 2
    v0 = np.asarray(meta[0]['displacement'])
    v1 = np.asarray(meta[1]['displacement'])
    np.testing.assert_allclose(v1, -v0, atol=1e-8)


def test_generate_inputs_writes_files_and_manifest(tmp_path):
    dc = _aluminium_controller(tmp_path)
    paths = generate_eph_inputs(dc, supercell_matrix=2, displacement_distance=0.06, nbnd=104)

    # fcc Al: a single symmetry-inequivalent displacement.
    assert len(paths) == 1
    edir = tmp_path / 'elphon'
    assert (edir / 'supercell.in').is_file()
    assert (edir / 'displacements.json').is_file()
    assert all(os.path.isfile(p) for p in paths)

    manifest = json.loads((edir / 'displacements.json').read_text())
    assert manifest['displacement_distance'] == pytest.approx(0.06)
    assert manifest['configuration'] == 'standard'
    assert manifest['nbnd'] == 104
    assert manifest['is_plusminus'] == 'auto'
    assert manifest['reference_prefix'].endswith('_ref')
    assert len(manifest['displacements']) == 1

    d0 = manifest['displacements'][0]
    assert 'displacement' in d0 and 'prefix' in d0
    assert d0['prefix'] != manifest['reference_prefix']

    # The band count is written into the SCF inputs.
    text = (edir / 'disp-001.in').read_text()
    assert "calculation = 'scf'" in text
    assert 'nbnd = 104' in text
    assert 'ATOMIC_POSITIONS' in text
    assert 'K_POINTS' in text


def test_plusminus_generation_writes_two_inputs(tmp_path):
    dc = _aluminium_controller(tmp_path)
    paths = generate_eph_inputs(
        dc, supercell_matrix=2, displacement_distance=0.06, nbnd=104, is_plusminus=True
    )
    assert len(paths) == 2
    manifest = json.loads((tmp_path / 'elphon' / 'displacements.json').read_text())
    assert manifest['is_plusminus'] is True
    prefixes = [d['prefix'] for d in manifest['displacements']]
    assert len(set(prefixes)) == 2


def test_cartesian_mode_generates_three_axis_displacements(tmp_path):
    dc = _aluminium_controller(tmp_path)
    paths = generate_eph_inputs(
        dc,
        supercell_matrix=2,
        displacement_distance=0.06,
        nbnd=104,
        displacement_mode='cartesian',
    )
    # One reference atom x, y, z (forward differences).
    assert len(paths) == 3
    manifest = json.loads((tmp_path / 'elphon' / 'displacements.json').read_text())
    vecs = np.array([d['displacement'] for d in manifest['displacements']])
    # The three displacement vectors are the +x, +y, +z axes times the distance.
    np.testing.assert_allclose(vecs, np.eye(3) * 0.06, atol=1e-12)


def test_cartesian_mode_plusminus_generates_six(tmp_path):
    # dc = _aluminium_controller(tmp_path)
    _, meta = generate_eph_displacements(
        init_phonopy(_aluminium_controller(tmp_path)),
        distance=0.06,
        is_plusminus=True,
        displacement_mode='cartesian',
    )
    assert len(meta) == 6  # 3 axes x +/-


def test_generate_inputs_without_pseudo_warns_and_omits_nbnd(tmp_path):
    dc = _aluminium_controller(tmp_path)  # no real Al.upf under fpath
    with pytest.warns(UserWarning):
        generate_eph_inputs(dc, supercell_matrix=2, displacement_distance=0.06)
    edir = tmp_path / 'elphon'
    manifest = json.loads((edir / 'displacements.json').read_text())
    assert manifest['nbnd'] is None
    assert 'nbnd =' not in (edir / 'disp-001.in').read_text()
