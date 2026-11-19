"""Magnetic fields on site: the Zeeman coupling mu_B (L + g_s S).B, the spinor basis, add_external_fields.

* Lande g-factors of spin-orbit multiplets of p and d shells (closed form, any field direction);
* the sign: the moment -mu_B (L + g_s S) lines up with B;
* collinear runs: -+ g_s mu_B B_z / 2 on the two channels;
* ``spinor_basis`` on a spin-degenerate model;
* ``add_external_fields`` applies only the fields of each call (16 Nov 2026).
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import numpy as np
import pytest

from PAOFLOW.hamiltonian.do_spin_orbit import _angular_momentum_matrices, _shell_soc_block
from PAOFLOW.hamiltonian.magnetic_field import (
    BOHR_MAGNETON_EV_PER_TESLA,
    ELECTRON_SPIN_G_FACTOR,
    PAULI,
    onsite_magnetic_field,
)

pytestmark = pytest.mark.unit

MU_B = BOHR_MAGNETON_EV_PER_TESLA
G_S = ELECTRON_SPIN_G_FACTOR


def _atom(l: int, soc: float) -> SimpleNamespace:
    """One shell with lambda L.S, block-ordered spinors, as a one-cell 'crystal'."""
    n = 2 * l + 1
    hrs = (soc * _shell_soc_block(0.0, 0.0, l)).reshape(2 * n, 2 * n, 1, 1, 1, 1)
    lmat = np.stack([np.kron(np.eye(2), m) for m in _angular_momentum_matrices(l)])
    smat = 0.5 * np.stack([np.kron(sigma, np.eye(n)) for sigma in PAULI])
    return SimpleNamespace(data_arrays={'HRs': hrs, 'Lj': lmat, 'Sj': smat}, data_attributes={})


def _lande(l: int, j: float, g_s: float = G_S) -> float:
    s = 0.5
    return (
        (j * (j + 1) + l * (l + 1) - s * (s + 1)) + g_s * (j * (j + 1) - l * (l + 1) + s * (s + 1))
    ) / (2 * j * (j + 1))


@pytest.mark.parametrize('l', [1, 2])
@pytest.mark.parametrize('direction', [(0.0, 0.0, 1.0), (0.48, -0.6, 0.64)])
def test_lande_g_factors_of_spin_orbit_multiplets(l, direction):
    soc, field = 1.0, 0.1  # eV, T: second order (mu_B B)^2 / soc ~ 3e-11 eV
    atom = _atom(l, soc)
    levels0 = np.linalg.eigvalsh(atom.data_arrays['HRs'][:, :, 0, 0, 0, 0])
    onsite_magnetic_field(atom, field * np.asarray(direction), g_s=G_S)
    levels = np.linalg.eigvalsh(atom.data_arrays['HRs'][:, :, 0, 0, 0, 0])
    expected = []
    for j, centre in ((l - 0.5, -soc * (l + 1) / 2), (l + 0.5, soc * l / 2)):
        m = np.arange(-j, j + 1)
        expected.extend(centre + _lande(l, j) * MU_B * field * m)
    centres = np.repeat([-soc * (l + 1) / 2, soc * l / 2], [2 * l, 2 * l + 2])  # j = l -+ 1/2
    np.testing.assert_allclose(levels0, centres, atol=1e-12)
    np.testing.assert_allclose(levels, np.sort(expected), atol=1e-10)


def test_moment_lines_up_with_the_field():
    atom = _atom(1, 0.0)
    onsite_magnetic_field(atom, np.array([0.0, 0.0, 5.0]), orbital=False)
    energies, vectors = np.linalg.eigh(atom.data_arrays['HRs'][:, :, 0, 0, 0, 0])
    lowest = vectors[:, 0]
    assert np.real(lowest.conj() @ atom.data_arrays['Sj'][2] @ lowest) == pytest.approx(-0.5)
    assert energies[-1] - energies[0] == pytest.approx(G_S * MU_B * 5.0)


def test_collinear_run_splits_the_channels():
    hrs = np.zeros((2, 2, 1, 1, 1, 2), complex)
    lz = np.zeros((3, 2, 2), complex)
    controller = SimpleNamespace(data_arrays={'HRs': hrs, 'Lj': lz}, data_attributes={})
    onsite_magnetic_field(controller, np.array([0.0, 0.0, 2.0]))
    onsite = np.real(controller.data_arrays['HRs'][0, 0, 0, 0, 0, :])
    np.testing.assert_allclose(
        onsite, [0.5 * G_S * MU_B * 2.0, -0.5 * G_S * MU_B * 2.0], atol=1e-15
    )
    with pytest.raises(ValueError):
        onsite_magnetic_field(controller, np.array([1.0, 0.0, 0.0]))


def _paoflow(model, tmp_path):
    from PAOFLOW import PAOFLOW

    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)


def _gamma_levels(p) -> np.ndarray:
    hrs = np.asarray(p.data_controller.data_arrays['HRs'])
    return np.linalg.eigvalsh(hrs[..., 0].sum(axis=(2, 3, 4)))


def test_spinor_basis_and_zeeman_on_a_spinless_model(tmp_path):
    from PAOFLOW.models.models import honeycomb_model

    p = _paoflow(honeycomb_model(delta=0.4, t=-1.0), tmp_path)
    levels0 = _gamma_levels(p)
    with contextlib.redirect_stdout(io.StringIO()):
        with pytest.raises(ValueError):  # no labels for the orbital angular momentum of a model
            p.magnetic_field([0.0, 0.0, 10.0])
        p.magnetic_field([0.0, 0.0, 10.0], orbital=False)
    arrays, attributes = p.data_controller.data_dicts()
    assert arrays['HRs'].shape[0] == 4 and attributes['dftSO'] and attributes['nelec'] == 2.0
    assert arrays['Dnm'].shape == (4, 4, 3)
    np.testing.assert_allclose(
        _gamma_levels(p)[[0, 2]] + 0.5 * G_S * MU_B * 10.0, levels0, atol=1e-12
    )
    splitting = np.diff(_gamma_levels(p))[[0, 2]]
    np.testing.assert_allclose(splitting, G_S * MU_B * 10.0, atol=1e-12)
    np.testing.assert_allclose(arrays['magnetic_field'], [0.0, 0.0, 10.0])


def test_add_external_fields_applies_each_field_once(tmp_path):
    from PAOFLOW.models.models import kane_mele_model

    p = _paoflow(kane_mele_model(soc=0.06, rashba=0.0, delta=0.1), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        p.add_external_fields(Bfield=[0.0, 0.0, 0.1])
        first = _gamma_levels(p)
        p.add_external_fields(HubbardU=np.full(4, 1e-12))
    second = _gamma_levels(p)
    np.testing.assert_allclose(first[1] - first[0], 0.1, atol=1e-12)
    np.testing.assert_allclose(second, first, atol=1e-11)  # the field was not applied again
    np.testing.assert_allclose(p.data_controller.data_arrays['Bfield'], [0.0, 0.0, 0.1])


def test_bfield_needs_a_spinor_basis(tmp_path):
    from PAOFLOW.models.models import honeycomb_model

    p = _paoflow(honeycomb_model(delta=0.4, t=-1.0), tmp_path)
    with contextlib.redirect_stdout(io.StringIO()), pytest.raises(ValueError):
        p.add_external_fields(Bfield=[0.0, 0.0, 0.1])
