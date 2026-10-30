"""The dense Fermi level honours the spin degeneracy (1 for spinor bands, 2 for scalar ones)."""

import numpy as np
import pytest

from PAOFLOW.elphon.elph_bloch import RY_TO_EV, _fermi_level, precompute_dense_electrons

pytestmark = pytest.mark.unit


def _flat_two_band_hrs():
    """Two orbitals, on-site energies 0 and 2 eV, no hopping: flat bands at 0 and 2 eV."""
    hrs = np.zeros((2, 2, 2, 2, 2, 1), dtype=complex)
    hrs[0, 0, 0, 0, 0, 0] = 0.0
    hrs[1, 1, 0, 0, 0, 0] = 2.0
    return hrs


def test_fermi_level_bisection_uses_the_spin_degeneracy():
    energies = np.array([[0.0, 2.0]] * 8) / RY_TO_EV  # (nk, nbnd), Ry
    sigma = 0.01 / RY_TO_EV
    ef_scalar = _fermi_level(energies, sigma, nelec=2, spin_deg=2.0) * RY_TO_EV
    ef_spinor = _fermi_level(energies, sigma, nelec=2, spin_deg=1.0) * RY_TO_EV
    assert 0.0 < ef_scalar < 2.0  # one doubly occupied band
    assert ef_spinor > 2.0  # both spinor bands filled


def test_precompute_dense_electrons_forwards_the_spin_degeneracy():
    hrs = _flat_two_band_hrs()
    sigma_ry = [0.01 / RY_TO_EV]
    scalar = precompute_dense_electrons(hrs, np.eye(3), 2, sigma_ry, 2, (2, 2, 2))
    spinor = precompute_dense_electrons(
        hrs, np.eye(3), 2, sigma_ry, 2, (2, 2, 2), spin_degeneracy=1.0
    )
    one_spinor = precompute_dense_electrons(
        hrs, np.eye(3), 2, sigma_ry, 1, (2, 2, 2), spin_degeneracy=1.0
    )
    assert 0.0 < scalar['ef_sig'][0] * RY_TO_EV < 2.0
    assert spinor['ef_sig'][0] * RY_TO_EV > 2.0
    assert 0.0 < one_spinor['ef_sig'][0] * RY_TO_EV < 2.0
