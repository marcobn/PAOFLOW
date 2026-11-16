"""The basis records built from radial shells (:mod:`PAOFLOW.projection.radial_shell_basis`)."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.basis_gen.driver import _write_two_col
from PAOFLOW.projection.do_atwfc_proj import build_aewfc_basis
from PAOFLOW.projection.radial_shell_basis import RadialShell, build_basis_from_radial_shells


class _Controller:
    def __init__(self, basispath=None, so=False):
        self.data_arrays = {
            'atoms': ['X', 'Y', 'X'],
            'tau': np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]),
            'configuration': {'X': ['3S', '3P', '3D'], 'Y': ['2P']},
        }
        self.data_attributes = {
            'ecutrho': 60.0, 'omega': 200.0, 'verbose': False, 'dftSO': so, 'basispath': basispath,
        }  # fmt: skip

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


def _gaussian_shells(tmp_path):
    r = np.linspace(0.0, 10.0, 801)[1:-1]
    rng = np.random.default_rng(0)
    layout = {'X': ['3S', '3P', '3D'], 'Y': ['2P']}
    shells = {}
    for elem, labels in layout.items():
        (tmp_path / elem).mkdir(parents=True, exist_ok=True)
        shells[elem] = []
        for label in labels:
            l = 'SPDF'.index(label[1])
            u = r ** (l + 1) * np.exp(-((r / rng.uniform(1.0, 2.5)) ** 2))
            path = tmp_path / elem / f'{label}.dat'
            _write_two_col(str(path), r, u / np.sqrt(np.sum(u * u) * (r[1] - r[0])))
            data = np.loadtxt(path)  # the arrays the table route reads
            shells[elem].append(RadialShell(label, l, data[:, 0], data[:, 1], 'table'))
    return shells


def test_records_match_the_table_builder(tmp_path):
    shells = _gaussian_shells(tmp_path)
    reference, ref_shells = build_aewfc_basis(_Controller(str(tmp_path)))
    basis, built_shells = build_basis_from_radial_shells(_Controller(), shells)
    assert built_shells == ref_shells == {'X': [0, 1, 2], 'Y': [1]}
    assert len(basis) == len(reference) == (1 + 3 + 5) * 2 + 3
    for record, expected in zip(basis, reference):
        for key in ('atom', 'l', 'm', 'label'):
            assert record[key] == expected[key]
        assert np.array_equal(record['tau'], expected['tau'])
        assert np.array_equal(record['r'], expected['r'])
        assert np.array_equal(record['wfc'], expected['wfc'])
        assert np.array_equal(record['wfc_g'], expected['wfc_g'])
    labels = [(b['atom'], b['label']) for b in basis]
    assert labels[:9] == [('X', '3S')] + [('X', '3P')] * 3 + [('X', '3D')] * 5
    assert labels[9:12] == [('Y', '2P')] * 3 and labels[12] == ('X', '3S')


def test_spin_orbit_duplication_matches_the_table_builder(tmp_path):
    shells = _gaussian_shells(tmp_path)
    dc_tables, dc_shells = _Controller(str(tmp_path), so=True), _Controller(so=True)
    reference, ref_shells = build_aewfc_basis(dc_tables)
    basis, built_shells = build_basis_from_radial_shells(dc_shells, shells)
    assert built_shells == ref_shells == {'X': [0, 1, 1, 2, 2], 'Y': [1, 1]}
    assert dc_shells.data_arrays['jchia'] == dc_tables.data_arrays['jchia']
    assert len(basis) == len(reference)
    for record, expected in zip(basis, reference):
        assert record['jm'] == expected['jm'] and record['l'] == expected['l']
        assert np.array_equal(record['wfc_g'], expected['wfc_g'])


def test_j_resolved_pair_fills_the_two_blocks(tmp_path):
    shells = _gaussian_shells(tmp_path)
    p_minus = shells['Y'][0]
    p_plus = RadialShell('2P', 1, p_minus.r, 0.5 * p_minus.wfc, 'j+', j=1.5)
    p_minus = RadialShell('2P', 1, p_minus.r, p_minus.wfc, 'j-', j=0.5)
    shells['Y'] = [p_plus, p_minus]  # any order
    basis, built_shells = build_basis_from_radial_shells(_Controller(so=True), shells)
    y_records = [b for b in basis if b['atom'] == 'Y']
    assert len(y_records) == 6 and built_shells['Y'] == [1, 1]
    assert np.array_equal(y_records[0]['wfc'], p_minus.wfc)  # j = l - 1/2 block first
    assert np.array_equal(y_records[3]['wfc'], p_plus.wfc)


def test_mixed_meshes_are_resampled_on_one_mesh(tmp_path):
    shells = _gaussian_shells(tmp_path)
    fine = np.linspace(0.0, 10.0, 1601)[1:-1]
    coarse_p = shells['X'][1]
    shells['X'][1] = RadialShell('3P', 1, fine, np.interp(fine, coarse_p.r, coarse_p.wfc), 'fine')
    basis, _ = build_basis_from_radial_shells(_Controller(), shells)
    x_records = [b for b in basis if b['atom'] == 'X']
    assert all(np.array_equal(b['r'], x_records[0]['r']) for b in x_records)
    p_record = next(b for b in x_records if b['label'] == '3P')
    assert np.allclose(p_record['wfc'], coarse_p.wfc, atol=2e-3)


def test_generator_shells_skip_and_filter(monkeypatch):
    from PAOFLOW.basis_gen.driver import GeneratedShell
    from PAOFLOW.projection import radial_shell_basis as module

    r = np.linspace(0.01, 10.0, 50)
    generated = [
        GeneratedShell('3S', 0, None, r, r, float('nan'), '3S.dat'),
        GeneratedShell('3P', 1, 0.5, r, r, 0.1, '3P_j1.dat'),
        GeneratedShell('3P', 1, 1.5, r, r, 0.2, '3P_j3.dat'),
        GeneratedShell('3P', 1, None, r, r, 0.15, '3P.dat'),
    ]
    monkeypatch.setattr(
        module, 'generate_basis_shells', lambda *a, **k: (generated, [('4F', 'no state')])
    )

    class _Relativistic:
        has_spinorbit = True
        title = 'rel'

    scalar, skipped = module.shells_from_generator(_Relativistic(), ['3S', '3P', '4F'])
    assert [(s.label, s.j) for s in scalar] == [('3S', None), ('3P', None)] and skipped == [
        ('4F', 'no state')
    ]
    assert scalar[0].source.startswith('pseudo-orbital') and 'box solver' in scalar[1].source
    spinor, _ = module.shells_from_generator(_Relativistic(), ['3S', '3P'], spin_orbit=True)
    assert [(s.label, s.j) for s in spinor] == [('3S', None), ('3P', 0.5), ('3P', 1.5)]


def test_mixed_tables_and_pseudo_orbitals_with_spin_orbit(tmp_path, monkeypatch):
    """Pseudo-orbitals of a dataset plus tables, duplicated into the j blocks."""
    from PAOFLOW.projection import radial_shell_basis as module

    shells = _gaussian_shells(tmp_path)
    r = shells['X'][0].r

    class _Dataset:
        title = 'fake'
        pswfc = [{'label': '3S', 'occ': 2.0, 'wfc': shells['X'][0].wfc}]

        def __init__(self):
            self.r = r

    monkeypatch.setattr(
        module, 'load_pseudopotentials', lambda dc: {'X': _Dataset(), 'Y': _Dataset()}
    )
    dc = _Controller(str(tmp_path), so=True)
    basis, built = module.build_mixed_basis_with_tables(dc, {'X': ['3S'], 'Y': []})
    reference, ref_shells = build_aewfc_basis(_Controller(str(tmp_path), so=True))
    assert built == ref_shells and len(basis) == len(reference)
    assert np.array_equal(basis[0]['wfc'], shells['X'][0].wfc)  # the 3S pseudo-orbital, both copies
    assert np.array_equal(basis[1]['wfc'], shells['X'][0].wfc) and basis[1]['jm'] == 1
    assert np.array_equal(basis[2]['wfc_g'], reference[2]['wfc_g'])  # the 3P table shell
    with pytest.raises(RuntimeError, match='no pseudo-orbital'):
        module.build_mixed_basis_with_tables(dc, {'X': ['3P'], 'Y': []})
