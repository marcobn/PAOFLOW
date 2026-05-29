"""Unit tests for UPF v2 PP_NONLOCAL (PP_BETA + PP_DIJ) parsing.

These tests cover Phase 3a of the non-local velocity correction: the
extension of :class:`PAOFLOW.inputs.read_upf.UPF` to expose Kleinman--Bylander
beta projectors and their coupling matrix ``D_{ij}``.

The tests use real norm-conserving UPF files that ship with the repository
under ``examples/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.inputs.read_upf import UPF

REPO_ROOT = Path(__file__).resolve().parents[3]

# Representative UPF v2 NC pseudos available in-repo.
UPFS = {
    'Si_ONCV': REPO_ROOT
    / 'examples'
    / 'qe_examples'
    / 'example16'
    / 'Si_ONCV'
    / 'Si_ONCV_PBE_sr.UPF',
    'Si_TM': REPO_ROOT / 'examples' / 'acbn0_examples' / 'Si' / 'Si.pbe-tm-new-gipaw-v2.1.UPF',
    'Mn': REPO_ROOT / 'examples' / 'acbn0_examples' / 'MnO' / 'Mn.upf',
}


@pytest.fixture(scope='module', params=list(UPFS.keys()))
def upf(request):
    path = UPFS[request.param]
    if not path.exists():
        pytest.skip(f'UPF file missing: {path}')
    return UPF(str(path))


def test_beta_attribute_populated(upf):
    """Every NC v2 UPF in the suite must expose ``nproj`` beta projectors."""
    assert hasattr(upf, 'beta')
    assert isinstance(upf.beta, list)
    assert len(upf.beta) == upf.nproj
    assert upf.nproj > 0


def test_beta_entries_well_formed(upf):
    """Each beta entry carries l, a finite radial array, and an optional cutoff."""
    for i, b in enumerate(upf.beta):
        assert set(b.keys()) >= {'l', 'wfc', 'cutoff_index', 'cutoff_radius', 'label'}
        assert isinstance(b['l'], int) and 0 <= b['l'] <= upf.lmax
        wfc = b['wfc']
        assert isinstance(wfc, np.ndarray)
        assert wfc.ndim == 1
        assert wfc.size > 0
        assert np.all(np.isfinite(wfc)), f'beta[{i}] has non-finite samples'
        if b['cutoff_index'] is not None:
            assert 0 < b['cutoff_index'] <= wfc.size
        # Beyond the cutoff QE pads with zeros; require the projector to
        # have nontrivial support inside it.
        assert np.any(np.abs(wfc) > 0.0)


def test_dion_shape_and_units(upf):
    """``dion`` is a square real matrix of shape (nproj, nproj) in Hartree."""
    D = upf.dion
    assert D is not None
    assert D.shape == (upf.nproj, upf.nproj)
    assert D.dtype.kind == 'f'
    assert np.all(np.isfinite(D))
    # NC pseudos: D is real symmetric (often diagonal).
    assert np.allclose(D, D.T, atol=1e-10), 'D_ij must be symmetric for NC pseudos'


def test_si_oncv_dion_matches_reference():
    """Spot-check the Si ONCV diagonal D values against the UPF file."""
    path = UPFS['Si_ONCV']
    if not path.exists():
        pytest.skip(f'UPF file missing: {path}')
    pp = UPF(str(path))
    assert pp.nproj == 6
    # Diagonal D values from the file, Rydberg → Hartree (÷ 2).
    expected_diag_ry = np.array(
        [
            1.0337930497e01,
            1.6597653773e00,
            5.1425645739e00,
            1.1566139757e00,
            -4.8546211098e00,
            -9.7619361064e-01,
        ]
    )
    np.testing.assert_allclose(np.diag(pp.dion), expected_diag_ry / 2.0, rtol=1e-12, atol=0)


def test_si_oncv_beta_angular_momenta():
    """Si ONCV PBE sr ships with two s, two p, two d projectors."""
    path = UPFS['Si_ONCV']
    if not path.exists():
        pytest.skip(f'UPF file missing: {path}')
    pp = UPF(str(path))
    ls = sorted(b['l'] for b in pp.beta)
    assert ls == [0, 0, 1, 1, 2, 2]


def test_beta_consistent_with_mesh(upf):
    """The beta arrays are sampled on the same radial mesh as :attr:`r`."""
    for i, b in enumerate(upf.beta):
        assert (
            b['wfc'].size == upf.r.size
        ), f'beta[{i}] has {b["wfc"].size} samples but mesh has {upf.r.size}'
