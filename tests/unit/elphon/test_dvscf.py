"""Unit tests for the finite-difference dV core (no QE runtime).

``finite_difference_dV`` is exercised directly; ``compute_dV`` is exercised with
the per-cell PAO Hamiltonian build monkeypatched out, so the manifest parsing,
the +/- pairing and the central difference are validated without QE.
"""

import json
import os

import numpy as np
import pytest

from PAOFLOW.elphon import dvscf_fd
from PAOFLOW.elphon.dvscf_fd import compute_dV, finite_difference_dV
from PAOFLOW.elphon.io import MANIFEST


class _StubController:
    def __init__(self, opath):
        self._arry = {}
        self._attr = {'opath': str(opath)}
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def test_finite_difference_central():
    rng = np.random.default_rng(0)
    hp = rng.standard_normal((3, 3, 2, 2, 2, 1))
    hm = rng.standard_normal((3, 3, 2, 2, 2, 1))
    dv = finite_difference_dV(hp, hm, distance=0.05)
    np.testing.assert_allclose(dv, (hp - hm) / (2.0 * 0.05))


def test_finite_difference_shape_mismatch():
    with pytest.raises(ValueError):
        finite_difference_dV(np.zeros((2, 2)), np.zeros((3, 3)), distance=0.05)


def test_finite_difference_zero_distance():
    with pytest.raises(ValueError):
        finite_difference_dV(np.zeros((2, 2)), np.zeros((2, 2)), distance=0.0)


def _write_manifest(edir, distance=0.06, plusminus=False):
    """Write a minimal manifest: one symmetry-reduced displacement (+/- optional)."""
    os.makedirs(edir, exist_ok=True)
    vec = [-0.0424, 0.0, 0.0424]  # phonopy's symmetry-adapted direction for fcc
    disps = [
        {
            'index': 0,
            'sc_atom': 0,
            'displacement': vec,
            'distance': distance,
            'prefix': 'Al1_disp001',
        }
    ]
    if plusminus:
        disps.append(
            {
                'index': 1,
                'sc_atom': 0,
                'displacement': [-v for v in vec],
                'distance': distance,
                'prefix': 'Al1_disp002',
            }
        )
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump(
            {
                'displacement_distance': distance,
                'configuration': 'standard',
                'is_plusminus': plusminus,
                'reference_prefix': 'Al1_ref',
                'displacements': disps,
            },
            f,
        )
    return vec


def _fake_builder(tables):
    def _build(savedir, **kwargs):
        prefix = os.path.basename(os.path.dirname(savedir)).replace('tmp_', '')
        return tables[prefix]

    return _build


def test_compute_dV_forward_difference_vs_reference(tmp_path, monkeypatch):
    edir = tmp_path / 'elphon'
    vec = _write_manifest(str(edir), distance=0.06, plusminus=False)
    dc = _StubController(tmp_path)

    tables = {
        'Al1_ref': np.full((2, 2, 1, 1, 1, 1), 1.0),
        'Al1_disp001': np.full((2, 2, 1, 1, 1, 1), 3.0),
    }
    monkeypatch.setattr(dvscf_fd, 'build_supercell_HRs', _fake_builder(tables))

    result = compute_dV(dc, elphon_dir='elphon', project_good_subspace=False)
    directional = result['directional']
    assert len(directional) == 1
    norm = float(np.linalg.norm(vec))
    np.testing.assert_allclose(directional[0]['dV'], (3.0 - 1.0) / norm)
    np.testing.assert_allclose(directional[0]['displacement'], vec)
    assert dc._arry['elphon_dV']['directional'] is directional


def test_compute_dV_central_difference_for_plusminus(tmp_path, monkeypatch):
    edir = tmp_path / 'elphon'
    vec = _write_manifest(str(edir), distance=0.06, plusminus=True)
    dc = _StubController(tmp_path)

    tables = {
        'Al1_disp001': np.full((2, 2, 1, 1, 1, 1), 5.0),  # +vec
        'Al1_disp002': np.full((2, 2, 1, 1, 1, 1), 2.0),  # -vec
    }
    monkeypatch.setattr(dvscf_fd, 'build_supercell_HRs', _fake_builder(tables))

    result = compute_dV(dc, elphon_dir='elphon', project_good_subspace=False)
    directional = result['directional']
    assert len(directional) == 1  # the +/- pair collapses to one central derivative
    norm = float(np.linalg.norm(vec))
    np.testing.assert_allclose(directional[0]['dV'], (5.0 - 2.0) / (2.0 * norm))


def test_compute_dV_forward_needs_reference_prefix(tmp_path, monkeypatch):
    edir = tmp_path / 'elphon'
    os.makedirs(edir)
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump(
            {
                'displacement_distance': 0.06,
                'configuration': 'standard',
                'reference_prefix': None,  # forward diff impossible
                'displacements': [
                    {
                        'index': 0,
                        'sc_atom': 0,
                        'displacement': [0.06, 0.0, 0.0],
                        'distance': 0.06,
                        'prefix': 'p1',
                    }
                ],
            },
            f,
        )
    dc = _StubController(tmp_path)
    monkeypatch.setattr(dvscf_fd, 'build_supercell_HRs', lambda savedir, **k: np.zeros((2, 2)))
    with pytest.raises(ValueError):
        compute_dV(dc, elphon_dir='elphon')


def _write_save_xml(edir, prefix, positions):
    """Write a minimal QE data-file-schema.xml with the given Cartesian positions."""
    save = os.path.join(edir, 'tmp_%s' % prefix, '%s.save' % prefix)
    os.makedirs(save, exist_ok=True)
    atoms = '\n'.join(
        '<atom name="Al" index="%d">%.12e %.12e %.12e</atom>' % (i + 1, *p)
        for i, p in enumerate(positions)
    )
    xml = (
        '<espresso><output><atomic_structure><atomic_positions>\n'
        + atoms
        + '\n</atomic_positions></atomic_structure></output></espresso>'
    )
    with open(os.path.join(save, 'data-file-schema.xml'), 'w') as f:
        f.write(xml)


def test_read_save_positions_parses_xml(tmp_path):
    _write_save_xml(str(tmp_path), 'Al1_ref', [[0.0, 0.0, 0.0], [3.8, 0.0, 3.8]])
    pos = dvscf_fd._read_save_positions(str(tmp_path), 'Al1_ref')
    assert pos.shape == (2, 3)
    np.testing.assert_allclose(pos[1], [3.8, 0.0, 3.8])
    # Missing save -> None (verification skipped, no false positive).
    assert dvscf_fd._read_save_positions(str(tmp_path), 'nope') is None


def test_verify_save_displacements_passes_when_consistent(tmp_path):
    edir = str(tmp_path)
    _write_save_xml(edir, 'Al1_ref', [[0.0, 0.0, 0.0]])
    _write_save_xml(edir, 'Al1_disp001', [[0.06, 0.0, 0.0]])
    disps = [{'sc_atom': 0, 'displacement': [0.06, 0.0, 0.0], 'prefix': 'Al1_disp001'}]
    dvscf_fd._verify_save_displacements(edir, disps, 'Al1_ref')  # no raise


def test_verify_save_displacements_catches_stale(tmp_path):
    edir = str(tmp_path)
    _write_save_xml(edir, 'Al1_ref', [[0.0, 0.0, 0.0]])
    # Save was computed with a <101>-type displacement, but the manifest says +x.
    _write_save_xml(edir, 'Al1_disp001', [[-0.0424, 0.0, 0.0424]])
    disps = [{'sc_atom': 0, 'displacement': [0.06, 0.0, 0.0], 'prefix': 'Al1_disp001'}]
    with pytest.raises(ValueError, match='Stale/mismatched'):
        dvscf_fd._verify_save_displacements(edir, disps, 'Al1_ref')


def test_verify_save_displacements_pair_check_without_reference(tmp_path):
    edir = str(tmp_path)
    # No reference save -> verify the +/- pair are true negatives of each other.
    _write_save_xml(edir, 'p_plus', [[0.06, 0.0, 0.0]])
    _write_save_xml(edir, 'p_minus', [[0.06, 0.0, 0.0]])  # wrong: should be -x
    disps = [
        {'sc_atom': 0, 'displacement': [0.06, 0.0, 0.0], 'prefix': 'p_plus'},
        {'sc_atom': 0, 'displacement': [-0.06, 0.0, 0.0], 'prefix': 'p_minus'},
    ]
    with pytest.raises(ValueError, match='Stale/mismatched'):
        dvscf_fd._verify_save_displacements(edir, disps, None)


def test_good_subspace_projectors_isolate_below_eta():
    # On-site H(k) = diag(1, 2, eta, eta): the good subspace is the first two.
    eta = 10.0
    HR = np.zeros((4, 4, 2, 2, 2, 1), dtype=complex)
    HR[:, :, 0, 0, 0, 0] = np.diag([1.0, 2.0, eta, eta])
    Pg = dvscf_fd.good_subspace_projectors(HR, eta, tol=0.05)
    expected = np.diag([1.0, 1.0, 0.0, 0.0]).astype(complex)
    for idx in [(0, 0, 0), (1, 1, 1)]:
        P = Pg[:, :, idx[0], idx[1], idx[2], 0]
        np.testing.assert_allclose(P, expected, atol=1e-10)
        np.testing.assert_allclose(P @ P, P, atol=1e-10)  # idempotent


def test_project_dV_good_removes_complement_coupling():
    # With P_good = diag(1,1,0,0), P dV P must zero all rows/cols of the complement.
    eta = 10.0
    HR = np.zeros((4, 4, 2, 2, 2, 1), dtype=complex)
    HR[:, :, 0, 0, 0, 0] = np.diag([1.0, 2.0, eta, eta])
    Pg = dvscf_fd.good_subspace_projectors(HR, eta, tol=0.05)
    rng = np.random.default_rng(0)
    dV = rng.standard_normal((4, 4, 2, 2, 2, 1)) + 1j * rng.standard_normal((4, 4, 2, 2, 2, 1))
    dVg = dvscf_fd.project_dV_good(dV, Pg)
    # Complement rows/columns (indices 2,3) are removed at every k.
    assert np.allclose(dVg[2:, :, :, :, :, :], 0.0, atol=1e-10)
    assert np.allclose(dVg[:, 2:, :, :, :, :], 0.0, atol=1e-10)
    # Projecting again is idempotent.
    np.testing.assert_allclose(dvscf_fd.project_dV_good(dVg, Pg), dVg, atol=1e-10)
