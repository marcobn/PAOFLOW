"""Spin-orbit basis: every (j, m_j) record carries the radial function of its own j.

``assign_jm`` labels the 2(2l+1) records of an l > 0 shell as 2l states j = l-1/2 followed by 2l+2
states j = l+1/2. Until 13 Nov 2026 the builders attached the radial functions per block of 2l+1,
so |j = l+1/2, m_j = -j> took the j = l-1/2 one and the basis broke time reversal and the point
group (meV in the PAO Hamiltonian of TaAs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.inputs.read_upf import UPF
from PAOFLOW.projection.do_atwfc_proj import (
    _upf_record_sources,
    build_aewfc_basis,
    build_pswfc_basis_all,
    spinor_record_sources,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
FR_PSEUDOS = REPO_ROOT / 'PSEUDOS' / 'nc-fr-04_pbe_standard'


class _Controller:
    def __init__(self, arrays, attributes):
        self.data_arrays, self.data_attributes = arrays, attributes

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


@pytest.mark.parametrize('l', [1, 2, 3])
def test_record_sources_follow_assign_jm(l):
    sources = spinor_record_sources(l, 'minus', 'plus')
    assert sources == ['minus'] * (2 * l) + ['plus'] * (2 * l + 2)
    with pytest.raises(ValueError):
        spinor_record_sources(0, 'minus', 'plus')


def test_upf_sources_use_the_jchi_of_each_pseudo_orbital():
    """PseudoDojo lists j = l+1/2 first: the j = l-1/2 records must still take the j = l-1/2 orbital."""
    pswfc = [{'label': lbl} for lbl in ('5S', '5P', '5P', '5D', '5D', '6S')]
    upf_j = [0.5, 1.5, 0.5, 2.5, 1.5, 0.5]
    sources = _upf_record_sources(pswfc, upf_j, spin_orbit=True)
    assert sources[0] == [0] and sources[5] == [5]
    assert sources[1] + sources[2] == [2, 2] + [1] * 4  # 5P: j = 1/2 is orbital 2
    assert sources[3] + sources[4] == [4] * 4 + [3] * 6  # 5D: j = 3/2 is orbital 4
    # no jchi: the file order is taken as j = l-1/2 first; no spin-orbit: every record its own
    plain = _upf_record_sources(pswfc, [], spin_orbit=True)
    assert plain[1] + plain[2] == [1, 1] + [2] * 4
    scalar = _upf_record_sources(pswfc, upf_j, spin_orbit=False)
    assert scalar == [[0], [1] * 3, [2] * 3, [3] * 5, [4] * 5, [5]]


def _write_table(path: Path, r: np.ndarray, scale: float) -> np.ndarray:
    wfc = r * np.exp(-scale * r)
    np.savetxt(path, np.column_stack([r, wfc]))
    return wfc


def test_table_basis_gives_each_jm_record_its_own_radial(tmp_path):
    r = np.linspace(1e-4, 12.0, 400)
    element = tmp_path / 'X'
    element.mkdir()
    tables = {
        '2S': _write_table(element / '2S.dat', r, 1.0),
        '2P_j1': _write_table(element / '2P_j1.dat', r, 1.1),
        '2P_j3': _write_table(element / '2P_j3.dat', r, 1.3),
        '3D_j3': _write_table(element / '3D_j3.dat', r, 0.9),
        '3D_j5': _write_table(element / '3D_j5.dat', r, 0.7),
    }
    _write_table(element / '2P.dat', r, 1.2)
    _write_table(element / '3D.dat', r, 0.8)
    arrays = {
        'atoms': ['X'],
        'tau': np.zeros((1, 3)),
        'configuration': {'X': ['2S', '2P', '3D']},
    }
    attributes = {
        'ecutrho': 40.0, 'omega': 200.0, 'verbose': False, 'dftSO': True,
        'basispath': str(tmp_path),
    }  # fmt: skip
    basis, shells = build_aewfc_basis(_Controller(arrays, attributes))
    assert shells == {'X': [0, 1, 1, 2, 2]}
    assert arrays['jchia']['X'] == [0.5, 0.5, 1.5, 1.5, 2.5]
    assert len(basis) == 2 + 6 + 10
    expected = ['2S'] * 2 + ['2P_j1'] * 2 + ['2P_j3'] * 4 + ['3D_j3'] * 4 + ['3D_j5'] * 6
    for record, name in zip(basis, expected):
        assert np.allclose(record['wfc'], tables[name]), name
    # one wfc_g object per table: the records of one (l, j) multiplet share it
    assert basis[4]['wfc_g'] is basis[7]['wfc_g'] and basis[4]['wfc_g'] is not basis[3]['wfc_g']
    assert [record['jm'] for record in basis] == [0, 1] + list(range(2, 8)) + list(range(8, 18))


@pytest.mark.skipif(not (FR_PSEUDOS / 'Ta.upf').exists(), reason='PseudoDojo FR Ta.upf missing')
def test_fully_relativistic_upf_basis_takes_the_pseudo_orbital_of_each_j():
    upf = UPF(str(FR_PSEUDOS / 'Ta.upf'))
    arrays = {'atoms': ['Ta'], 'tau': np.zeros((1, 3)), 'species': [('Ta', 'Ta.upf')]}
    attributes = {
        'ecutrho': 40.0, 'omega': 200.0, 'verbose': False, 'dftSO': True,
        'fpath': str(FR_PSEUDOS),
    }  # fmt: skip
    basis, _ = build_pswfc_basis_all(_Controller(arrays, attributes))
    by_label_j = {(p['label'], j): p['wfc'] for p, j in zip(upf.pswfc, upf.jchia)}
    start = 0
    for label, l in (('5S', 0), ('5P', 1), ('5D', 2)):
        records = basis[start : start + 2 * (2 * l + 1)]
        start += len(records)
        if l == 0:
            continue
        for index, record in enumerate(records):
            j = l - 0.5 if index < 2 * l else l + 0.5
            assert np.array_equal(record['wfc'], by_label_j[(label, j)]), (label, index)
