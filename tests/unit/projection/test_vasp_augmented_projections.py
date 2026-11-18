"""The augmented (PAW) metric of the projections for VASP input.

The WAVECAR bands of a VASP run are orthonormal in ``S = 1 + sum q_ij |p_i><p_j|`` built from
the POTCAR datasets (reciprocal-space projector tables). Needs the VASP run ``nscf_nspin1`` of
the Si case ``examples/cases/vasp_si_bands`` (as ``../example01/nscf_nspin1`` next to the
repository, or ``PAOFLOW_VASP_EXAMPLES/example01/nscf_nspin1``) and the ``potpaw_PBE`` library;
skipped without them.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.inputs.read_potcar import read_potcar
from PAOFLOW.projection.augmentation import build_augmentation, k_plus_g_vectors
from PAOFLOW.projection.do_atwfc_proj import read_dft_wfc, read_VASP_wfc, read_WAVECAR_header

REPO_ROOT = Path(__file__).resolve().parents[3]
LIBRARY = Path(os.environ.get('PAOFLOW_POTCAR_DIR', REPO_ROOT.parent / 'potpaw_PBE_64'))
EXAMPLES = Path(os.environ.get('PAOFLOW_VASP_EXAMPLES', REPO_ROOT.parent))
SI_RUN = EXAMPLES / 'example01' / 'nscf_nspin1'


class _Controller:
    rank = 0

    def __init__(self, fpath, potcar_dir):
        self.data_arrays = {}
        self.data_attributes = {
            'verbose': False,
            'dft': 'VASP',
            'fpath': str(fpath),
            'potcar_dir': potcar_dir,
        }

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


@pytest.fixture(scope='module')
def controller():
    if not (SI_RUN / 'WAVECAR').exists() or not (LIBRARY / 'Si' / 'POTCAR').exists():
        pytest.skip('Si WAVECAR example or POTCAR library not available')
    from PAOFLOW.inputs.read_VASP import parse_vasprun_data

    dc = _Controller(SI_RUN, str(LIBRARY))
    parse_vasprun_data(dc, str(SI_RUN / 'vasprun.xml'))
    return dc


def test_wavecar_bands_are_orthonormal_in_the_augmented_metric(controller):
    arry, attr = controller.data_dicts()
    assert attr['vasp_lreal'] == 'F'
    augmentation = build_augmentation(controller)
    assert augmentation is not None and augmentation.nproj == 16  # 2 Si x (1 + 1 + 3 + 3)
    for ik in (0, 3, 20):
        gkspace, wfc = read_dft_wfc(controller, ik, 0, orthonormalize=False)
        plain = wfc['wfc'].conj() @ wfc['wfc'].T
        assert np.abs(plain - np.eye(plain.shape[0])).max() > 1e-2  # not plainly orthonormal
        beta = augmentation.beta(k_plus_g_vectors(gkspace))
        overlap = augmentation.overlap(wfc['wfc'], wfc['wfc'], beta)
        assert np.abs(overlap - np.eye(overlap.shape[0])).max() < 1e-5


def test_raw_and_orthonormalised_coefficients(controller):
    read_WAVECAR_header(controller)
    _, raw = read_VASP_wfc(controller, 0, 0, orthonormalize=False)
    _, ortho = read_VASP_wfc(controller, 0, 0)
    assert raw['npol'] == 1 and ortho['nbnd'] == raw['nbnd']
    assert not np.allclose(raw['wfc'], ortho['wfc'])
    gram = ortho['wfc'].conj() @ ortho['wfc'].T
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-6)
    # the header is read once
    arry, attr = controller.data_dicts()
    nplws = arry['nplws']
    read_WAVECAR_header(controller)
    assert arry['nplws'] is nplws


def test_projections_use_the_augmented_metric(controller):
    from PAOFLOW.projection.do_atwfc_proj import augmented_proj_k, calc_proj_k
    from PAOFLOW.projection.radial_shell_basis import build_generated_basis

    arry, attr = controller.data_dicts()
    attr['ecutrho'] = attr['ecutwfc'] * 5
    arry['configuration'] = {'Si': ['3S', '3P']}
    basis, shells = build_generated_basis(controller)
    assert shells == {'Si': [0, 1]} and len(basis) == 8
    arry['augmentation'] = build_augmentation(controller)
    proj = calc_proj_k(controller, basis, 0, 0)
    reference, Sk = augmented_proj_k(controller, arry['augmentation'], basis, 0, 0)
    assert np.allclose(proj, reference)
    assert Sk.shape == (8, 8) and np.allclose(Sk, Sk.conj().T)
    # the four valence bands are fully projectable on the pseudo-orbitals
    weights = np.sum(np.abs(proj) ** 2, axis=1)
    assert np.all(weights[:4] > 0.95) and np.all(weights <= 1.0 + 1e-6)


def test_missing_potcar_falls_back_to_the_plain_overlap(capsys):
    class _NoPotcar:
        rank = 0

        def data_dicts(self):
            arrays = {
                'species': [('Si', 'PAW_PBE Si 05Jan2001')],
                'atoms': ['Si'],
                'tau': np.zeros((1, 3)),
            }
            attrs = {
                'dft': 'VASP',
                'fpath': '/nowhere',
                'potcar_dir': '/nowhere_else',
                'ecutwfc': 20.0,
                'omega': 100.0,
                'alat': 1.0,
            }
            return arrays, attrs

    import os as _os

    saved = _os.environ.pop('PAOFLOW_POTCAR_DIR', None)
    try:
        assert build_augmentation(_NoPotcar()) is None
    finally:
        if saved is not None:
            _os.environ['PAOFLOW_POTCAR_DIR'] = saved
    assert 'potcar_dir' in capsys.readouterr().out


def test_scalar_dataset_in_a_spin_orbit_run_couples_the_spins_diagonally():
    if not (LIBRARY / 'Pt' / 'POTCAR').exists():
        pytest.skip('POTCAR library not available')
    from PAOFLOW.elphon.pseudopotential_terms import KleinmanBylanderProjectors

    pt = read_potcar(str(LIBRARY / 'Pt' / 'POTCAR'))[0]
    kb = KleinmanBylanderProjectors(
        {'Pt': pt}, ['Pt'], 100.0, 6.0, spin_orbit=True, require_norm_conserving=False
    )
    assert kb.Q is None and kb.Q_so is not None and kb.Q_so.shape == (2, 2, kb.nproj, kb.nproj)
    scalar = KleinmanBylanderProjectors(
        {'Pt': pt}, ['Pt'], 100.0, 6.0, require_norm_conserving=False
    )
    assert np.allclose(kb.Q_so[0, 0], scalar.Q) and np.allclose(kb.Q_so[1, 1], scalar.Q)
    assert np.allclose(kb.Q_so[0, 1], 0.0) and np.allclose(kb.Q_so[1, 0], 0.0)
    # the tables come from the reciprocal-space tables of the file
    q = np.arange(kb.tables['Pt'].shape[1]) * kb.dq
    q_table, values = pt.projector_q_tables[0]
    inside = q <= q_table[-1]
    from scipy.interpolate import CubicSpline

    expected = CubicSpline(q_table, values)(q[inside]) / np.sqrt(100.0)
    assert np.allclose(kb.tables['Pt'][0][inside], expected)
