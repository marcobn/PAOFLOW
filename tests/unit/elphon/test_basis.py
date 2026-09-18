"""Unit tests for the electron-phonon PAO basis-size helper (no UPF needed)."""

import pytest

from PAOFLOW.elphon import basis
from PAOFLOW.elphon.basis import _orbitals_per_shell, supercell_nbnd


class _StubController:
    def __init__(self, arry, attr):
        self._arry = arry
        self._attr = attr
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


class _Cell:
    def __init__(self, symbols):
        self.symbols = list(symbols)


@pytest.mark.parametrize(
    'label,expected',
    [('3S', 1), ('3P', 3), ('3D', 5), ('4F', 7), ('1s', 1), ('2p', 3)],
)
def test_orbitals_per_shell(label, expected):
    assert _orbitals_per_shell(label) == expected


def test_orbitals_per_shell_rejects_bad_label():
    with pytest.raises(ValueError):
        _orbitals_per_shell('X')


def test_supercell_nbnd_sums_over_atoms(monkeypatch):
    # Avoid touching real UPF files: fix the per-species orbital count.
    monkeypatch.setattr(basis, 'species_pao_orbitals', lambda path, configuration='standard': 9)

    arry = {'species': [('Al', '/some/dir/Al.upf')]}
    attr = {'fpath': '/some/dir'}
    dc = _StubController(arry, attr)
    cell = _Cell(['Al'] * 8)  # 2x2x2 supercell of a single-atom cell

    # 8 atoms x 9 orbitals + 4 margin.
    assert supercell_nbnd(dc, cell, configuration='standard', margin=4) == 8 * 9 + 4
    assert supercell_nbnd(dc, cell, configuration='standard') == 8 * 9


def test_supercell_nbnd_missing_pseudo_raises(monkeypatch):
    arry = {'species': [('Al', 'Al.upf')]}
    attr = {'fpath': '/nonexistent'}
    dc = _StubController(arry, attr)
    cell = _Cell(['Al'])
    with pytest.raises((FileNotFoundError, ValueError)):
        supercell_nbnd(dc, cell)


def test_supercell_nbnd_unknown_species_raises():
    arry = {'species': [('Al', 'Al.upf')]}
    attr = {'fpath': '.'}
    dc = _StubController(arry, attr)
    cell = _Cell(['Cu'])  # not in the species map
    with pytest.raises(ValueError):
        supercell_nbnd(dc, cell)
