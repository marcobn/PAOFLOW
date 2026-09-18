"""Unit tests for the Stage 2 NAC / Born-charge plumbing.

Exercises the BORN-file I/O (:func:`PAOFLOW.phonon.io.write_born_file` /
:func:`read_born_file`), the NAC attachment
(:func:`PAOFLOW.phonon.do_phonopy.attach_nac`) and the primitive-atom ordering
helper, without any Quantum ESPRESSO runtime.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon.do_phonopy import attach_nac, init_phonopy
from PAOFLOW.phonon.io import read_born_file, write_born_file
from PAOFLOW.phonon.structure import primitive_atom_info


class _StubController:
    """Minimal ``DataController`` stand-in exposing ``data_dicts()``."""

    def __init__(self, arry, attr, rank=0):
        self._arry = arry
        self._attr = attr
        self.rank = rank

    def data_dicts(self):
        return self._arry, self._attr


def _gaas_controller(tmp_path, supercell_matrix=2):
    """Two-atom fcc GaAs controller (inequivalent atoms carry +/- Z*)."""
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
    return _StubController(arry, attr)


def test_write_read_born_round_trip(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)

    born = np.array([np.eye(3) * 2.18, np.eye(3) * (-2.18)])
    eps = np.eye(3) * 10.9

    path = write_born_file(dc, born, eps, phonon_dir='phonon')
    assert os.path.isfile(path)

    nac = read_born_file(dc, path)
    assert set(nac) == {'born', 'dielectric', 'factor'}
    assert nac['born'].shape == (2, 3, 3)
    np.testing.assert_allclose(nac['born'], born, atol=1e-8)
    np.testing.assert_allclose(nac['dielectric'], eps, atol=1e-8)
    assert nac['factor'] > 0.0


def test_attach_nac_from_arrays_sets_nac_params(tmp_path):
    dc = _gaas_controller(tmp_path)
    phonon = init_phonopy(dc)

    born = np.array([np.eye(3) * 2.18, np.eye(3) * (-2.18)])
    eps = np.eye(3) * 10.9

    nac = attach_nac(dc, born=born, dielectric=eps)

    assert phonon.nac_params is not None
    np.testing.assert_allclose(phonon.nac_params['born'], born, atol=1e-8)
    np.testing.assert_allclose(phonon.nac_params['dielectric'], eps, atol=1e-8)
    # QE NAC conversion factor.
    assert nac['factor'] == pytest.approx(2.0)
    # Arrays stashed on the controller for downstream use.
    arry, _ = dc.data_dicts()
    np.testing.assert_allclose(arry['born_charges'], born, atol=1e-8)
    np.testing.assert_allclose(arry['dielectric_tensor'], eps, atol=1e-8)


def test_attach_nac_from_file(tmp_path):
    dc = _gaas_controller(tmp_path)
    phonon = init_phonopy(dc)

    born = np.array([np.eye(3) * 2.18, np.eye(3) * (-2.18)])
    eps = np.eye(3) * 10.9
    path = write_born_file(dc, born, eps, phonon_dir='phonon')

    attach_nac(dc, born_file=path)
    np.testing.assert_allclose(phonon.nac_params['born'], born, atol=1e-8)
    np.testing.assert_allclose(phonon.nac_params['dielectric'], eps, atol=1e-8)


def test_attach_nac_requires_inputs(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    with pytest.raises(ValueError):
        attach_nac(dc)


def test_primitive_atom_info(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    info = primitive_atom_info(dc)
    assert info['natom'] == 2
    assert sorted(info['symbols']) == ['As', 'Ga']
    assert info['masses'].shape == (2,)
    assert info['scaled_positions'].shape == (2, 3)
    assert info['cell'].shape == (3, 3)
