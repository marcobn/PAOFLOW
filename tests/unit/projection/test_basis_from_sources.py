"""Two-source basis (UPF pseudo-orbitals + tables) of ``build_basis_from_sources``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.projection.do_atwfc_proj import build_basis_from_sources

REPO_ROOT = Path(__file__).resolve().parents[3]
UPF_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example10'
BASIS = REPO_ROOT / 'BASIS'


class _Controller:
    def __init__(self):
        self.data_arrays = {
            'atoms': ['Ga', 'As'],
            'tau': np.array([[0.0, 0.0, 0.0], [2.67, 2.67, 2.67]]),
            'species': [
                ('Ga', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'),
                ('As', 'As.pbe-n-kjpaw_psl.1.0.0.UPF'),
            ],
            'configuration': {'Ga': ['3D', '4S', '4P', '4D', '5S'], 'As': ['4S', '4P', '4D']},
        }
        self.data_attributes = {
            'ecutrho': 200.0, 'omega': 300.0, 'verbose': False, 'dftSO': False,
            'fpath': str(UPF_DIR), 'basispath': str(BASIS),
        }  # fmt: skip

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


@pytest.mark.skipif(
    not (UPF_DIR / 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF').exists()
    or not (BASIS / 'Ga' / '4D.dat').exists(),
    reason='example10 UPFs or BASIS tables missing',
)
def test_shells_come_from_the_requested_source_in_atom_order():
    dc = _Controller()
    basis, shells = build_basis_from_sources(dc, {'Ga': ['3D', '4S', '4P'], 'As': ['4S', '4P']})
    labels = [(b['atom'], b['label']) for b in basis]
    # atom-major, configuration order, 2l+1 records per shell
    assert labels[:5] == [('Ga', '3D')] * 5 and labels[5:6] == [('Ga', '4S')]
    assert labels[9:14] == [('Ga', '4D')] * 5 and labels[14:15] == [('Ga', '5S')]
    assert labels[15:16] == [('As', '4S')] and labels[-5:] == [('As', '4D')] * 5
    assert len(basis) == 5 + 1 + 3 + 5 + 1 + 1 + 3 + 5
    assert shells == {'Ga': [2, 0, 1, 2, 0], 'As': [0, 1, 2]}
    # the UPF shells carry the UPF mesh and function, the table shells the uniform table mesh
    from PAOFLOW.inputs.read_upf import UPF

    upf = UPF(str(UPF_DIR / 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'))
    ga_3d = basis[0]
    ga_4d = basis[9]
    # one mesh per species (that of the tables); the pseudo-orbital is resampled on it
    assert np.allclose(ga_3d['r'], ga_4d['r'])
    chi = [c for c in upf.pswfc if c['label'].upper() == '3D'][0]['wfc']
    inside = (ga_3d['r'] > upf.r[0]) & (ga_3d['r'] < upf.r[-1])
    assert np.allclose(ga_3d['wfc'][inside], np.interp(ga_3d['r'][inside], upf.r, chi), atol=2e-3)
    assert ga_4d['label'] == '4D' and ga_4d['atom'] == 'Ga'


def test_missing_pseudo_orbital_is_an_error():
    dc = _Controller()
    with pytest.raises(RuntimeError):
        build_basis_from_sources(dc, {'Ga': ['5S'], 'As': []})


def test_spin_orbit_is_refused():
    dc = _Controller()
    dc.data_attributes['dftSO'] = True
    with pytest.raises(NotImplementedError):
        build_basis_from_sources(dc, {'Ga': ['3D'], 'As': []})
