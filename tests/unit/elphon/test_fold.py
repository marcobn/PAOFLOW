"""Unit tests for the supercell -> primitive fold of the e-phonon derivative."""

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.elphon.fold import (
    _fft_integers,
    fold_dV_to_primitive,
    supercell_atom_translations,
)
from PAOFLOW.phonon.do_phonopy import init_phonopy


class _StubController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def _aluminium_phonon(supercell_matrix=2):
    alat = 7.6326928726
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    arry = {'a_vectors': a_vectors, 'tau': np.zeros((1, 3)), 'atoms': ['Al']}
    attr = {
        'alat': alat,
        'natoms': 1,
        'opath': '.',
        'phonon_supercell_matrix': supercell_matrix,
        'verbose': False,
    }
    return init_phonopy(_StubController(arry, attr))


def test_fft_integers():
    np.testing.assert_array_equal(_fft_integers(4), [0, 1, -2, -1])
    np.testing.assert_array_equal(_fft_integers(3), [0, 1, -1])


def test_supercell_translations_fcc_al():
    ph = _aluminium_phonon(2)
    s2p, T = supercell_atom_translations(ph)
    assert s2p.shape == (8,)
    assert np.all(s2p == 0)  # single primitive atom
    # The 8 sub-cell translations of a 2x2x2 supercell.
    expected = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    )
    # Rows must match as a set.
    assert {tuple(t) for t in T} == {tuple(e) for e in expected}


def test_fold_shape_and_placement():
    # One primitive atom, 1 orbital, 2x2x2 supercell, tiny 2x2x2 supercell grid.
    ph = _aluminium_phonon(2)
    s2p, T = supercell_atom_translations(ph)
    naw = np.ones(8, dtype=int)  # 1 orbital per atom
    n = 2
    nspin = 1
    dV_sc = np.zeros((8, 8, n, n, n, nspin), dtype=complex)

    # bra = reference atom (T=0), ket = atom with T=(1,0,0), supercell R = 0.
    # -> R_e = S*0 + T_ket - T_bra = (1,0,0); R_p = -T_bra = (0,0,0).
    a0 = [a for a in range(8) if tuple(T[a]) == (0, 0, 0)][0]
    b = [a for a in range(8) if tuple(T[a]) == (1, 0, 0)][0]
    dV_sc[a0, b, 0, 0, 0, 0] = 7.0

    g = fold_dV_to_primitive(dV_sc, s2p, T, naw, ph.supercell_matrix)
    assert g.shape == (1, 1, 4, 4, 4, 2, 2, 2, 1)  # (i,j, R_e[4^3], R_p[2^3], spin)

    assert g[0, 0, 1, 0, 0, 0, 0, 0, 0] == pytest.approx(7.0)
    assert np.count_nonzero(g) == 1


def test_fold_supercell_R_combines_with_subcell():
    ph = _aluminium_phonon(2)
    s2p, T = supercell_atom_translations(ph)
    naw = np.ones(8, dtype=int)
    n = 2
    dV_sc = np.zeros((8, 8, n, n, n, 1), dtype=complex)
    a0 = [a for a in range(8) if tuple(T[a]) == (0, 0, 0)][0]
    b = [a for a in range(8) if tuple(T[a]) == (1, 0, 0)][0]
    # Supercell R index (1,0,0) -> integer R_sc = +1 -> R_e = S*1 + T_ket = (3,0,0).
    dV_sc[a0, b, 1, 0, 0, 0] = 3.0
    g = fold_dV_to_primitive(dV_sc, s2p, T, naw, ph.supercell_matrix)
    assert g[0, 0, 3, 0, 0, 0, 0, 0, 0] == pytest.approx(3.0)
    assert np.count_nonzero(g) == 1


def test_fold_bra_cell_sets_phonon_cell_index():
    # A bra atom in cell T=(1,0,0) contributes to the phonon-cell slice R_p=-T mod S=(1,0,0).
    ph = _aluminium_phonon(2)
    s2p, T = supercell_atom_translations(ph)
    naw = np.ones(8, dtype=int)
    dV_sc = np.zeros((8, 8, 2, 2, 2, 1), dtype=complex)
    a = [a for a in range(8) if tuple(T[a]) == (1, 0, 0)][0]  # bra with T=(1,0,0)
    b = [a for a in range(8) if tuple(T[a]) == (1, 0, 0)][0]  # ket same cell
    dV_sc[a, b, 0, 0, 0, 0] = 5.0
    g = fold_dV_to_primitive(dV_sc, s2p, T, naw, ph.supercell_matrix)
    # R_e = T_ket - T_bra = 0; R_p = -T_bra mod 2 = (1,0,0).
    assert g[0, 0, 0, 0, 0, 1, 0, 0, 0] == pytest.approx(5.0)
    assert np.count_nonzero(g) == 1


def test_fold_naw_mismatch_raises():
    ph = _aluminium_phonon(2)
    s2p, T = supercell_atom_translations(ph)
    dV_sc = np.zeros((8, 8, 2, 2, 2, 1), dtype=complex)
    with pytest.raises(ValueError):
        fold_dV_to_primitive(dV_sc, s2p, T, np.full(8, 2), ph.supercell_matrix)  # sums to 16 != 8
