"""Unit tests for :mod:`PAOFLOW.phonon.structure` (Stage 0 structure bridge).

The converters only consume ``DataController.data_dicts()``, so a lightweight
stub exposing that method is enough to validate the PAOFLOW <-> phonopy
structure mapping without any QE runtime dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon.structure import (
    _element_symbol,
    paoflow_to_phonopy,
    phonopy_to_paoflow,
    verify_round_trip,
)


class _StubController:
    """Minimal stand-in for ``DataController`` exposing ``data_dicts()``."""

    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr

    def data_dicts(self):
        return self._arry, self._attr


def _silicon_controller():
    """Two-atom fcc silicon cell in PAOFLOW conventions (Bohr / alat units)."""
    alat = 10.20  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    cell_bohr = a_vectors * alat
    # Second atom at fractional (1/4, 1/4, 1/4).
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=float) @ cell_bohr
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {'a_vectors': a_vectors, 'tau': tau, 'atoms': ['Si', 'Si']}
    attr = {'alat': alat, 'natoms': 2, 'omega': omega}
    return _StubController(arry, attr)


def test_element_symbol_strips_site_markers():
    assert _element_symbol('Si') == 'Si'
    assert _element_symbol('Fe1') == 'Fe'
    assert _element_symbol('fe2') == 'Fe'
    assert _element_symbol('O') == 'O'
    with pytest.raises(ValueError):
        _element_symbol('123')


def test_paoflow_to_phonopy_lattice_and_masses():
    dc = _silicon_controller()
    arry, attr = dc.data_dicts()
    cell = paoflow_to_phonopy(dc)

    # QE convention: phonopy cell is in Bohr (calculator='qe').
    expected_lattice = arry['a_vectors'] * attr['alat']
    np.testing.assert_allclose(np.asarray(cell.cell), expected_lattice, atol=1e-12)

    assert list(cell.symbols) == ['Si', 'Si']
    np.testing.assert_allclose(cell.masses, [28.0855, 28.0855], atol=1e-3)

    expected_scaled = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    np.testing.assert_allclose(np.asarray(cell.scaled_positions), expected_scaled, atol=1e-12)


def test_phonopy_to_paoflow_inverts_forward_map():
    dc = _silicon_controller()
    arry, attr = dc.data_dicts()
    cell = paoflow_to_phonopy(dc)

    back = phonopy_to_paoflow(cell, alat=attr['alat'])

    np.testing.assert_allclose(back['a_vectors'], arry['a_vectors'], atol=1e-12)
    np.testing.assert_allclose(back['tau'], arry['tau'], atol=1e-10)
    assert back['atoms'] == ['Si', 'Si']
    np.testing.assert_allclose(back['omega'], attr['omega'], rtol=1e-12)


def test_round_trip_helper_reports_ok():
    dc = _silicon_controller()
    res = verify_round_trip(dc)

    assert res['ok'] is True
    assert res['symbols_match'] is True
    assert res['lattice_dev_bohr'] < 1e-10
    assert res['position_dev_bohr'] < 1e-9


def test_default_alat_uses_first_lattice_vector_norm():
    dc = _silicon_controller()
    arry, attr = dc.data_dicts()
    cell = paoflow_to_phonopy(dc)

    back = phonopy_to_paoflow(cell)  # alat inferred
    cell_bohr = np.asarray(cell.cell)  # already Bohr (QE convention)
    expected_alat = np.linalg.norm(cell_bohr[0])

    assert np.isclose(back['alat'], expected_alat, rtol=1e-12)
    # Cartesian lattice (Bohr) must be invariant regardless of alat choice.
    np.testing.assert_allclose(
        back['a_vectors'] * back['alat'],
        arry['a_vectors'] * attr['alat'],
        atol=1e-10,
    )
