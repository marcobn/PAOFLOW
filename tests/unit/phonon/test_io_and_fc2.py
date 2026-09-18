"""Unit tests for the Stage 1 finite-displacement phonon pipeline.

These exercise :mod:`PAOFLOW.phonon.io` (QE input writing, k-grid scaling,
pseudopotential mapping) and the :mod:`PAOFLOW.phonon.do_phonopy` force-constant
pipeline without any Quantum ESPRESSO runtime: forces are fabricated so the
fc2 -> frequencies path can be validated for shape and self-consistency.
"""

from __future__ import annotations

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon.do_phonopy import (
    generate_displacements,
    init_phonopy,
    produce_force_constants,
)
from PAOFLOW.phonon.io import (
    _namelists,
    _pp_filenames,
    _supercell_kgrid,
    resolve_phonon_dir,
    write_displaced_supercells,
)


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
# io helpers
# --------------------------------------------------------------------------- #
def test_pp_filenames_maps_symbol_to_basename(tmp_path):
    dc = _silicon_controller(tmp_path)
    dc._arry['species'] = [('Si1', '/some/dir/Si_ONCV_PBE_sr.UPF')]
    assert _pp_filenames(dc) == {'Si': 'Si_ONCV_PBE_sr.UPF'}


def test_supercell_kgrid_scales_diagonal(tmp_path):
    dc = _silicon_controller(tmp_path, supercell_matrix=2)
    kg = _supercell_kgrid(dc, np.diag([2, 2, 2]))
    np.testing.assert_array_equal(kg, [2, 2, 2])


def test_supercell_kgrid_anisotropic_and_floor(tmp_path):
    dc = _silicon_controller(tmp_path)
    dc._attr.update(nk1=4, nk2=4, nk3=1)
    kg = _supercell_kgrid(dc, np.diag([2, 4, 4]))
    # ceil(4/2)=2, ceil(4/4)=1, ceil(1/4)->floored to 1.
    np.testing.assert_array_equal(kg, [2, 1, 1])


def test_resolve_phonon_dir_creates_relative_to_opath(tmp_path):
    dc = _silicon_controller(tmp_path)
    path = resolve_phonon_dir(dc, 'phonon')
    assert path == str(tmp_path / 'phonon')
    import os

    assert os.path.isdir(path)


def test_namelists_contain_required_scf_settings(tmp_path):
    dc = _silicon_controller(tmp_path)
    phonon = init_phonopy(dc)
    text = _namelists(dc, phonon.supercell, prefix='silicon', pp_dir=str(tmp_path))
    assert "calculation = 'scf'" in text
    assert 'tprnfor = .true.' in text
    assert 'ibrav = 0' in text
    assert 'nat = %d' % len(phonon.supercell) in text
    assert 'ecutwfc = 40.00' in text


# --------------------------------------------------------------------------- #
# displaced-supercell writing
# --------------------------------------------------------------------------- #
def test_write_displaced_supercells_emits_inputs(tmp_path):
    import os

    dc = _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05)
    init_phonopy(dc)
    generate_displacements(dc)
    written = write_displaced_supercells(dc, phonon_dir='phonon', prefix='silicon')

    assert len(written) >= 1
    for path in written:
        assert os.path.isfile(path)
        with open(path) as f:
            content = f.read()
        assert 'K_POINTS automatic' in content
        assert 'ATOMIC_POSITIONS' in content
        assert 'tprnfor = .true.' in content

    # Perfect reference cell and provenance file must also be present.
    assert os.path.isfile(tmp_path / 'phonon' / 'supercell.in')
    assert os.path.isfile(tmp_path / 'phonon' / 'phonopy_disp.yaml')


def test_supercell_inputs_use_ibrav_not_cell_parameters(tmp_path):
    """A uniform supercell of a canonical-ibrav cell is written with ibrav."""

    dc = _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05)
    init_phonopy(dc)
    generate_displacements(dc)
    written = write_displaced_supercells(dc, phonon_dir='phonon', prefix='silicon')

    with open(written[0]) as f:
        content = f.read()
    # fcc silicon (ibrav=2): supercell shares the Bravais lattice, celldm doubled.
    assert 'ibrav = 2' in content
    assert 'celldm(1) = 20.4000' in content
    assert 'CELL_PARAMETERS' not in content
    assert 'ATOMIC_POSITIONS (crystal)' in content


def test_detect_ibrav_cell_preserves_cartesian_geometry(tmp_path):
    """The ibrav crystal coords reproduce the exact phonopy Cartesian geometry."""
    from PAOFLOW.inputs.lattice_format import lattice_format_QE
    from PAOFLOW.phonon.io import _detect_ibrav_cell

    dc = _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05)
    phonon = init_phonopy(dc)
    generate_displacements(dc)

    cell = phonon.supercells_with_displacements[0]
    detected = _detect_ibrav_cell(cell)
    assert detected is not None
    ibrav, celldm, frac = detected
    assert ibrav == 2

    lat_qe = np.asarray(lattice_format_QE(ibrav, celldm), dtype=float)
    cart_qe = frac @ lat_qe
    cart_ref = np.asarray(cell.scaled_positions) @ np.asarray(cell.cell)
    np.testing.assert_allclose(cart_qe, cart_ref, atol=1e-9)


def test_detect_ibrav_cell_rejects_rotated_cell(tmp_path):
    """A cell rotated off QE's canonical axes must fall back to CELL_PARAMETERS."""
    from phonopy.structure.atoms import PhonopyAtoms

    from PAOFLOW.phonon.io import _detect_ibrav_cell

    dc = _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05)
    phonon = init_phonopy(dc)
    cell = phonon.supercell

    theta = 0.37
    rot = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    rotated = PhonopyAtoms(
        symbols=[str(s) for s in cell.symbols],
        cell=np.asarray(cell.cell) @ rot.T,
        scaled_positions=cell.scaled_positions,
    )
    assert _detect_ibrav_cell(rotated) is None


# --------------------------------------------------------------------------- #
# fc2 pipeline (fabricated forces, no QE)
# --------------------------------------------------------------------------- #
def test_produce_force_constants_shape_and_frequencies(tmp_path):
    dc = _silicon_controller(tmp_path, supercell_matrix=2, distance=0.05)
    phonon = init_phonopy(dc)
    generate_displacements(dc)

    supercells = phonon.supercells_with_displacements
    nsuper = len(phonon.supercell)
    ndisp = len(supercells)

    # Fabricate a deterministic, drift-free force set so the least-squares fc2
    # solve runs end to end without a QE calculation.
    rng = np.random.default_rng(0)
    forces = rng.standard_normal((ndisp, nsuper, 3))
    forces -= forces.mean(axis=1, keepdims=True)  # remove net drift

    produce_force_constants(dc, forces=forces)

    fc2 = phonon.force_constants
    assert fc2.shape == (nsuper, nsuper, 3, 3)

    # The frequency pipeline must return 3*natoms branches per q-point.
    phonon.run_qpoints([[0.0, 0.0, 0.0]])
    freqs = phonon.qpoints.frequencies
    assert freqs.shape == (1, 3 * dc._attr['natoms'])
