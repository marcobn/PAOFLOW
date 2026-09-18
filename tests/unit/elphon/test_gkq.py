"""Unit tests for the phonon ingredients and the g_mn^v contraction (P2).

``phonon_modes`` is validated directly against phonopy (a fabricated but valid
diagonal fc2 for single-atom fcc Al); the contraction ``assemble_g_kq`` is
checked for shape, the 1/sqrt(omega) scaling, linearity and acoustic masking
with fabricated inputs.
"""

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.elphon.gkq import (
    assemble_g_kq,
    frequency_prefactor,
    mode_displacement_pattern,
    phonon_modes,
)
from PAOFLOW.phonon.do_phonopy import init_phonopy


class _StubController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def _aluminium_phonon(tmp_path):
    alat = 7.6326928726
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    arry = {'a_vectors': a_vectors, 'tau': np.zeros((1, 3)), 'atoms': ['Al']}
    attr = {
        'alat': alat,
        'natoms': 1,
        'opath': str(tmp_path),
        'phonon_supercell_matrix': 2,
        'verbose': False,
    }
    phonon = init_phonopy(_StubController(arry, attr))
    # Fabricate a translationally invariant (acoustic-sum-rule) fc2 from a
    # nearest-neighbour chain so the three acoustic branches vanish at Gamma.
    nsc = len(phonon.supercell)
    fc = np.zeros((nsc, nsc, 3, 3))
    k = 2.0
    for i in range(nsc):
        j = (i + 1) % nsc
        fc[i, i] += k * np.eye(3)
        fc[j, j] += k * np.eye(3)
        fc[i, j] -= k * np.eye(3)
        fc[j, i] -= k * np.eye(3)
    phonon.force_constants = fc
    return phonon


def test_phonon_modes_shapes_and_masses(tmp_path):
    phonon = _aluminium_phonon(tmp_path)
    q = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]]
    freqs, ev, masses = phonon_modes(phonon, q)

    assert freqs.shape == (2, 3)  # 3 acoustic branches
    assert ev.shape == (2, 1, 3, 3)  # (nq, natom, 3, nmode)
    np.testing.assert_allclose(masses, [26.9815386], rtol=1e-6)


def test_phonon_modes_gamma_acoustic_are_zero(tmp_path):
    phonon = _aluminium_phonon(tmp_path)
    freqs, _, _ = phonon_modes(phonon, [[0.0, 0.0, 0.0]])
    # The three acoustic modes are (near) zero frequency at Gamma.
    np.testing.assert_allclose(freqs[0], 0.0, atol=1e-6)


def test_mode_displacement_pattern_mass_weighting():
    natom, nmode = 2, 6
    eigvec = np.zeros((natom, 3, nmode), dtype=complex)
    eigvec[0, 0, 0] = 1.0  # atom 0, x, mode 0
    eigvec[1, 1, 1] = 1.0  # atom 1, y, mode 1
    masses = np.array([4.0, 9.0])
    pat = mode_displacement_pattern(eigvec, masses)
    assert pat.shape == (nmode, natom * 3)
    assert pat[0, 0] == pytest.approx(1.0 / 2.0)  # 1/sqrt(4)
    assert pat[1, 4] == pytest.approx(1.0 / 3.0)  # 1/sqrt(9), atom1*3 + y


def test_frequency_prefactor_scaling_and_floor():
    omega = np.array([0.0, 1.0, 4.0])  # THz
    pref = frequency_prefactor(omega, units='THz')
    assert pref[0] == 0.0  # acoustic Gamma masked
    # sqrt(1/(2 w)) so quadrupling omega halves the prefactor.
    assert pref[2] / pref[1] == pytest.approx(0.5)


def _random_g_inputs(nawf=4, nbnd_k=3, nbnd_kq=3, natom=1, seed=0):
    rng = np.random.default_rng(seed)

    def cplx(*shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    ncart = 3 * natom
    dHdu = cplx(ncart, nawf, nawf)
    v_k = cplx(nawf, nbnd_k)
    v_kq = cplx(nawf, nbnd_kq)
    eigvec = cplx(natom, 3, ncart)
    omega = np.abs(rng.standard_normal(ncart)) + 0.5  # THz, no zero modes
    masses = np.abs(rng.standard_normal(natom)) + 1.0
    return dHdu, v_k, v_kq, eigvec, omega, masses


def test_assemble_g_shape():
    dHdu, v_k, v_kq, eigvec, omega, masses = _random_g_inputs(nbnd_k=3, nbnd_kq=2)
    g = assemble_g_kq(dHdu, v_k, v_kq, eigvec, omega, masses)
    assert g.shape == (3, 2, 3)  # (nmode, nbnd_kq, nbnd_k)


def test_assemble_g_frequency_scaling():
    dHdu, v_k, v_kq, eigvec, omega, masses = _random_g_inputs()
    g1 = assemble_g_kq(dHdu, v_k, v_kq, eigvec, omega, masses)
    g4 = assemble_g_kq(dHdu, v_k, v_kq, eigvec, 4.0 * omega, masses)
    # g ~ 1/sqrt(omega): quadrupling omega halves g.
    np.testing.assert_allclose(g4, 0.5 * g1, rtol=1e-10)


def test_assemble_g_linear_in_derivative():
    dHdu, v_k, v_kq, eigvec, omega, masses = _random_g_inputs()
    g1 = assemble_g_kq(dHdu, v_k, v_kq, eigvec, omega, masses)
    g2 = assemble_g_kq(2.0 * dHdu, v_k, v_kq, eigvec, omega, masses)
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-10)


def test_assemble_g_rejects_mismatched_cartesian():
    dHdu, v_k, v_kq, eigvec, omega, masses = _random_g_inputs(natom=1)
    bad = dHdu[:2]  # only 2 Cartesian rows, not 3*natom
    with pytest.raises(ValueError):
        assemble_g_kq(bad, v_k, v_kq, eigvec, omega, masses)
