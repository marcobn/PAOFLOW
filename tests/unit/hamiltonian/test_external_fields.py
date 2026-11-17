"""External fields of ``add_external_fields``: each orbital is shifted by the field at its own centre.

Pins the 16 Nov 2026 fix of ``hamiltonian/add_ext_field.py``: the electric field now reads the orbital
centres from :func:`PAOFLOW.hamiltonian.orbital_centres.orbital_centres_bohr` (the projection basis, or
``tau`` in bohr for DFT runs and in alat units for the models). Before, it gave ``nawf // natoms``
consecutive orbitals to each atom and read the models' ``tau`` as bohr (a field too small by ``alat``).
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.hamiltonian.add_ext_field import add_ext_field
from PAOFLOW.hamiltonian.orbital_centres import orbital_centres_bohr
from PAOFLOW.utils.constants import ANGSTROM_AU

pytestmark = pytest.mark.unit


def _controller(nawf: int, arrays: dict, attributes: dict, efield=(0.0, 0.0, 0.0), bfield=(0.0, 0.0, 0.0),
                hubbard=None) -> SimpleNamespace:  # fmt: skip
    arrays = dict(arrays)
    arrays.setdefault('HRs', np.zeros((nawf, nawf, 1, 1, 1, 1), complex))
    arrays['Efield'] = np.asarray(efield, float)
    arrays['Bfield'] = np.asarray(bfield, float)
    arrays['HubbardU'] = np.zeros(nawf) if hubbard is None else np.asarray(hubbard, float)
    return SimpleNamespace(data_arrays=arrays, data_attributes=dict(attributes))


def _old_add_ext_field(arrays: dict, attributes: dict) -> np.ndarray:
    """The rule before 16 Nov 2026, verbatim: nawf // natoms consecutive orbitals per atom, tau read as bohr."""
    hrs = np.array(arrays['HRs'], complex)
    nawf = hrs.shape[0]
    hrs = hrs.reshape(nawf, nawf, -1, hrs.shape[-1])
    nwf = nawf // attributes['natoms']
    tau_wf = np.zeros((nawf, 3))
    row = 0
    for atom in range(attributes['natoms']):
        for _ in range(nwf):
            tau_wf[row] = arrays['tau'][atom]
            row += 1
    tau_wf /= ANGSTROM_AU
    if arrays['Efield'].any():
        for n in range(nawf):
            hrs[n, n, 0, :] -= arrays['Efield'].dot(tau_wf[n])
    if arrays['Bfield'].any():
        field = sum(arrays['Bfield'][i] * arrays['Sj'][i] for i in range(3))
        hrs[:, :, 0, :] -= field[:, :, None]
    if arrays['HubbardU'].any():
        for n in range(nawf):
            hrs[n, n, 0, :] -= arrays['HubbardU'][n] / 2.0
    return hrs.reshape(arrays['HRs'].shape)


# --------------------------------------------------------------------------- orbital centres
def test_orbital_centres_from_each_source():
    hrs = lambda n: np.zeros((n, n, 1, 1, 1, 1), complex)
    tau = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    # projection basis: one centre per orbital, bohr; doubled to block spinors by adhoc_spin_orbit
    basis = [{'tau': tau[0]}, {'tau': tau[1]}, {'tau': tau[1]}]
    np.testing.assert_allclose(orbital_centres_bohr({'HRs': hrs(3), 'basis': basis, 'tau': tau}, {'alat': 5.0}),
                               tau[[0, 1, 1]])  # fmt: skip
    np.testing.assert_allclose(orbital_centres_bohr({'HRs': hrs(6), 'basis': basis, 'tau': tau}, {'alat': 5.0}),
                               tau[[0, 1, 1, 0, 1, 1]])  # fmt: skip
    # models: one row per orbital in alat units
    centres = orbital_centres_bohr({'HRs': hrs(2), 'tau': tau}, {'alat': 5.0, 'tau_units': 'alat'})
    np.testing.assert_allclose(centres, 5.0 * tau)
    # Slater-Koster models: tau per atom (alat units), orbitals from the shells
    shells = [{'orbitals': ['s']}, {'orbitals': ['s', 'px', 'py', 'pz']}]
    centres = orbital_centres_bohr(
        {'HRs': hrs(5), 'tau': tau, 'shells': shells}, {'alat': 5.0, 'tau_units': 'alat'}
    )
    np.testing.assert_allclose(centres, 5.0 * tau[[0, 1, 1, 1, 1]])
    # DFT without the basis: naw when it adds up, else the equal split of the former rule
    centres = orbital_centres_bohr(
        {'HRs': hrs(5), 'tau': tau, 'naw': np.array([1, 4])}, {'alat': 5.0}
    )
    np.testing.assert_allclose(centres, tau[[0, 1, 1, 1, 1]])
    centres = orbital_centres_bohr(
        {'HRs': hrs(4), 'tau': tau, 'naw': np.array([0, 0])}, {'alat': 5.0}
    )
    np.testing.assert_allclose(centres, tau[[0, 0, 1, 1]])
    with pytest.raises(ValueError):
        orbital_centres_bohr({'HRs': hrs(5), 'tau': tau}, {'alat': 5.0})


# --------------------------------------------------------------------------- electric field
def test_efield_uses_each_orbital_centre():
    # an s atom and an sp atom (5 orbitals): the former rule gave 2 orbitals to each atom and none to the fifth
    tau = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    dc = _controller(
        5, {'tau': tau, 'naw': np.array([1, 4])}, {'alat': 7.0, 'natoms': 2}, efield=(0.0, 0.0, 0.3)
    )
    add_ext_field(dc)
    onsite = np.real(np.diag(dc.data_arrays['HRs'][:, :, 0, 0, 0, 0]))
    np.testing.assert_allclose(
        onsite, -0.3 * np.array([0.0, 2.0, 2.0, 2.0, 2.0]) / ANGSTROM_AU, atol=1e-15
    )


def test_homogeneous_dft_basis_is_unchanged():
    # every atom with the same orbitals, grouped by atom: the former rule and the new one agree to the bit
    rng = np.random.default_rng(3)
    nawf, tau = 8, rng.random((2, 3)) * 5.0
    hrs = rng.normal(size=(nawf, nawf, 2, 1, 1, 1)) + 1j * rng.normal(size=(nawf, nawf, 2, 1, 1, 1))
    sj = rng.normal(size=(3, nawf, nawf)) + 1j * rng.normal(size=(3, nawf, nawf))
    arrays = {'HRs': hrs, 'tau': tau, 'naw': np.array([4, 4]), 'Sj': sj}
    attributes = {'alat': 6.0, 'natoms': 2}
    dc = _controller(nawf, arrays, attributes, efield=(0.1, -0.2, 0.3), bfield=(0.0, 0.0, 0.4),
                     hubbard=rng.random(nawf))  # fmt: skip
    reference = _old_add_ext_field(dc.data_arrays, attributes)
    add_ext_field(dc)
    np.testing.assert_array_equal(dc.data_arrays['HRs'], reference)


def test_model_field_is_in_physical_units(tmp_path):
    # a buckled Kane-Mele layer, A above B by h: E_z gives the staggered potential lambda_v = E_z h / 2 and the
    # gap at K closes at E_c = 3 sqrt(3) lambda_SO / (h / 2) (it closed at alat E_c before the fix)
    import dataclasses

    from PAOFLOW import PAOFLOW
    from PAOFLOW.hamiltonian.bloch_derivatives import bloch_terms_from_HRs, hamiltonian_block
    from PAOFLOW.models.models import kane_mele_model

    soc, height = 0.06, 0.46
    km = kane_mele_model(t=1.0, soc=soc, rashba=0.0, delta=0.0)
    dz = height * ANGSTROM_AU / (km.alat * km.lattice_vectors[2, 2])
    positions = np.array(km.orbital_positions, float)
    positions[[0, 2], 2] += dz / 2
    positions[[1, 3], 2] -= dz / 2
    e_critical = 3 * np.sqrt(3) * soc / (height / 2)
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=dataclasses.replace(km, orbital_positions=positions), outputdir=str(tmp_path),
                            verbose=False)  # fmt: skip
        p.add_external_fields(Efield=[0.0, 0.0, e_critical])
    arrays, attributes = p.data_controller.data_dicts()
    onsite = np.real(np.diag(arrays['HRs'][:, :, 0, 0, 0, 0]))
    np.testing.assert_allclose(
        onsite, -e_critical * height / 2 * np.array([1, -1, 1, -1]), atol=1e-12
    )
    terms = bloch_terms_from_HRs(arrays['HRs'], arrays['a_vectors'], attributes['alat'], None)
    bands = np.linalg.eigvalsh(hamiltonian_block(terms, np.array([[2 / 3, 1 / 3, 0.0]])))[0]
    assert bands[2] - bands[1] < 1e-10
