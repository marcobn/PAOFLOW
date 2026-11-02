"""Unit tests of the random-displacement mode and its local least-squares fit.

The linear algebra is exercised on a synthetic linear model (exact recovery with
and without the locality cutoff); the generate phase on the fcc aluminium stub;
``compute_dV`` end to end with the per-cell Hamiltonian build monkeypatched to a
linear model of the displacements, including fake ``pw.x`` force blocks.
"""

import json
import os

import numpy as np
import pytest

from PAOFLOW.elphon import dvscf_fd, random_displacements as rd
from PAOFLOW.elphon.dvscf_fd import compute_dV
from PAOFLOW.elphon.io import MANIFEST


# ----------------------------------------------------------------------------
# Synthetic linear model
# ----------------------------------------------------------------------------


def _chain(nat=4, spacing=2.0):
    """Periodic chain of ``nat`` atoms along x; cutoff 2.5 keeps nearest neighbours."""
    positions = np.zeros((nat, 3))
    positions[:, 0] = spacing * np.arange(nat)
    cell = np.diag([spacing * nat, 20.0, 20.0])
    return positions, cell


def test_neighbour_sets_minimum_image():
    positions, cell = _chain()
    sets = rd.neighbour_sets(positions, cell, cutoff=2.5)
    # Atom 0 sees 1 and, through the periodic image, 3.
    np.testing.assert_array_equal(sets[0], [0, 1, 3])
    np.testing.assert_array_equal(sets[2], [1, 2, 3])
    full = rd.neighbour_sets(positions, cell, cutoff=None)
    assert all(len(s) == 4 for s in full)


def test_random_configurations_norm_and_seed():
    u = rd.random_configurations(5, 0.03, 7, seed=3)
    assert u.shape == (7, 5, 3)
    np.testing.assert_allclose(np.linalg.norm(u, axis=2), 0.03)
    np.testing.assert_allclose(rd.random_configurations(5, 0.03, 7, seed=3), u)
    assert not np.allclose(rd.random_configurations(5, 0.03, 7, seed=4), u)


def _local_truth(nsets, naw, nrest=2, seed=0):
    """Random complex derivatives with the locality structure of ``nsets``."""
    rng = np.random.default_rng(seed)
    nat = len(nsets)
    nawf = naw * nat
    orbital_atoms = np.repeat(np.arange(nat), naw)
    W = rng.standard_normal((3 * nat, nawf, nawf, nrest)) + 1j * rng.standard_normal(
        (3 * nat, nawf, nawf, nrest)
    )
    for a in range(nat):
        for b in range(nat):
            allowed = np.union1d(nsets[a], nsets[b])
            for k in range(nat):
                if k not in allowed:
                    W[np.ix_(rd._active_indices([k]), orbital_atoms == a, orbital_atoms == b)] = 0.0
    return W, orbital_atoms


def _accumulate(W, patterns):
    nat = patterns.shape[1]
    neq = rd.NormalEquations(3 * nat, W.shape[1:], dtype=complex)
    for u in patterns:
        D = np.tensordot(u.ravel(), W, axes=(0, 0))
        neq.add(u.ravel(), D)
    return neq


def test_local_fit_recovers_exact_derivatives():
    # Ring of 8 atoms: N(k) has 3 atoms, N(a) U N(b) at most 6 (18 unknowns).
    positions, cell = _chain(nat=8)
    nsets = rd.neighbour_sets(positions, cell, cutoff=2.5)
    assert max(len(np.union1d(nsets[a], nsets[b])) for a in range(8) for b in range(8)) == 6
    W, orbital_atoms = _local_truth(nsets, naw=2)
    # 20 configurations: fewer than the 24 global unknowns but more than the
    # 18 unknowns of any local active set.
    patterns = rd.random_configurations(8, 0.05, 20, seed=1)
    neq = _accumulate(W, patterns)
    Wfit, residual, pair_res = rd.fit_hamiltonian_derivatives(neq, orbital_atoms, nsets, ridge=0.0)
    np.testing.assert_allclose(Wfit, W, atol=1e-9)
    assert np.all(residual < 1e-10)
    assert np.all(pair_res < 1e-5)


def test_global_fit_matches_local_with_enough_configurations():
    positions, cell = _chain()
    nsets = rd.neighbour_sets(positions, cell, cutoff=2.5)
    W, orbital_atoms = _local_truth(nsets, naw=1)
    patterns = rd.random_configurations(4, 0.05, 30, seed=2)
    neq = _accumulate(W, patterns)
    full = rd.neighbour_sets(positions, cell, cutoff=None)
    Wfit, residual, _ = rd.fit_hamiltonian_derivatives(neq, orbital_atoms, full, ridge=0.0)
    np.testing.assert_allclose(Wfit, W, atol=1e-9)


def test_ridge_damps_underdetermined_fit():
    positions, cell = _chain()
    full = rd.neighbour_sets(positions, cell, cutoff=None)
    W, orbital_atoms = _local_truth(full, naw=1)
    patterns = rd.random_configurations(4, 0.05, 6, seed=5)  # 6 < 12 unknowns
    neq = _accumulate(W, patterns)
    Wfit, residual, _ = rd.fit_hamiltonian_derivatives(neq, orbital_atoms, full, ridge=1e-3)
    assert np.all(np.isfinite(Wfit))
    # Under-determined: the fit reproduces the data (small residual) but not the truth.
    assert np.sum(residual) < 1e-3 * np.sum(neq.y2)
    assert not np.allclose(Wfit, W, atol=1e-3)


def test_residual_reports_nonlinear_contamination():
    positions, cell = _chain()
    full = rd.neighbour_sets(positions, cell, cutoff=None)
    W, orbital_atoms = _local_truth(full, naw=1)
    patterns = rd.random_configurations(4, 0.05, 40, seed=6)
    nat = 4
    neq = rd.NormalEquations(3 * nat, W.shape[1:], dtype=complex)
    rng = np.random.default_rng(0)
    noise = 0.01
    for u in patterns:
        D = np.tensordot(u.ravel(), W, axes=(0, 0))
        D = D + noise * np.linalg.norm(D) / np.sqrt(D.size) * rng.standard_normal(D.shape)
        neq.add(u.ravel(), D)
    _, residual, _ = rd.fit_hamiltonian_derivatives(neq, orbital_atoms, full, ridge=0.0)
    rel = np.sqrt(np.sum(residual) / np.sum(neq.y2))
    assert 0.3 * noise < rel < 1.5 * noise


def test_force_constants_fit_symmetric_and_in_ev_per_angstrom2():
    positions, cell = _chain()
    nsets = rd.neighbour_sets(positions, cell, cutoff=2.5)
    nat = 4
    rng = np.random.default_rng(7)
    phi = rng.standard_normal((3 * nat, 3 * nat))
    phi = 0.5 * (phi + phi.T)
    for k in range(nat):
        for kp in range(nat):
            if kp not in nsets[k]:
                phi[np.ix_(rd._active_indices([k]), rd._active_indices([kp]))] = 0.0
    patterns = rd.random_configurations(nat, 0.05, 12, seed=8)
    neq = rd.NormalEquations(3 * nat, (3 * nat,), dtype=float)
    for u in patterns:
        f = phi @ u.ravel()  # -(F+ - F-)/2 = Phi u
        neq.add(u.ravel(), f)
    fc, rel = rd.fit_force_constants(neq, nsets, ridge=0.0)
    assert fc.shape == (nat, nat, 3, 3)
    expect = phi.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3) * rd.RY_BOHR2_TO_EV_ANG2
    np.testing.assert_allclose(fc, expect, atol=1e-9)
    assert rel < 1e-6  # cancellation floor of the normal-equation residual


# ----------------------------------------------------------------------------
# Generate phase (phonopy)
# ----------------------------------------------------------------------------


class _StubController:
    def __init__(self, opath, arry=None, attr=None):
        self._arry = {} if arry is None else arry
        self._attr = {'opath': str(opath)} if attr is None else attr
        self.rank = 0

    def data_dicts(self):
        return self._arry, self._attr


def _aluminium_controller(tmp_path, supercell_matrix=2):
    """Single-atom fcc aluminium stub (as in ``test_displacements``)."""
    alat = 7.6326928726  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    tau = np.zeros((1, 3), dtype=float)
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': ['Al'],
        'species': [('Al', 'Al.upf')],
    }
    attr = {
        'alat': alat,
        'natoms': 1,
        'omega': omega,
        'opath': str(tmp_path),
        'savedir': 'Al1.save',
        'fpath': str(tmp_path),
        'ecutwfc': 44.0,
        'nspin': 1,
        'insulator': False,
        'degauss': 0.05,
        'nk1': 22,
        'nk2': 22,
        'nk3': 22,
        'phonon_supercell_matrix': supercell_matrix,
        'verbose': False,
    }
    return _StubController(tmp_path, arry, attr)


def test_random_mode_writes_plus_minus_all_atom_patterns(tmp_path):
    pytest.importorskip('phonopy')
    from PAOFLOW.elphon.displacements import generate_eph_displacements
    from PAOFLOW.elphon.do_elphon import generate_eph_inputs
    from PAOFLOW.phonon.do_phonopy import init_phonopy

    dc = _aluminium_controller(tmp_path)
    phonon = init_phonopy(dc)
    cells, meta = generate_eph_displacements(
        phonon, distance=0.04, displacement_mode='random', nconfig=3, seed=11
    )
    assert len(cells) == len(meta) == 6
    nat = len(phonon.supercell.positions)
    u0 = np.asarray(meta[0]['displacement'])
    assert u0.shape == (nat, 3)
    np.testing.assert_allclose(np.linalg.norm(u0, axis=1), 0.04)
    np.testing.assert_allclose(np.asarray(meta[1]['displacement']), -u0)
    assert meta[0]['sc_atom'] is None and meta[0]['config'] == 0 and meta[1]['sign'] == -1
    base = np.asarray(phonon.supercell.positions)
    np.testing.assert_allclose(np.asarray(cells[0].positions) - base, u0, atol=1e-10)

    with pytest.raises(ValueError):
        generate_eph_displacements(phonon, distance=0.04, displacement_mode='random')

    paths = generate_eph_inputs(
        dc,
        supercell_matrix=2,
        displacement_distance=0.04,
        nbnd=104,
        displacement_mode='random',
        nconfig=2,
        seed=5,
    )
    assert len(paths) == 4
    manifest = json.loads((tmp_path / 'elphon' / 'displacements.json').read_text())
    assert manifest['displacement_mode'] == 'random'
    assert manifest['nconfig'] == 2 and manifest['seed'] == 5 and manifest['is_plusminus'] is True
    assert 'tprnfor = .true.' in (tmp_path / 'elphon' / 'disp-001.in').read_text()


# ----------------------------------------------------------------------------
# Analyse phase end to end (monkeypatched Hamiltonian build)
# ----------------------------------------------------------------------------


def _write_random_manifest(edir, patterns, distance):
    os.makedirs(edir, exist_ok=True)
    disps = []
    index = 0
    for m, u in enumerate(patterns):
        for sign in (+1, -1):
            index += 1
            disps.append(
                {
                    'index': index - 1,
                    'sc_atom': None,
                    'config': m,
                    'sign': sign,
                    'displacement': (sign * u).tolist(),
                    'distance': distance,
                    'prefix': 'X_disp%03d' % index,
                    'input': 'disp-%03d.in' % index,
                }
            )
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump(
            {
                'displacement_distance': distance,
                'configuration': 'standard',
                'is_plusminus': True,
                'displacement_mode': 'random',
                'nconfig': len(patterns),
                'seed': 0,
                'reference_prefix': 'X_ref',
                'displacements': disps,
            },
            f,
        )
    return disps


def _write_forces_out(edir, name, forces):
    lines = ['     Forces acting on atoms (cartesian axes, Ry/au):', '']
    for i, f in enumerate(forces):
        lines.append('     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i + 1, *f))
    lines += ['', '     Total force =     0.000000     Total SCF correction =     0.000000']
    with open(os.path.join(edir, os.path.splitext(name)[0] + '.out'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def test_compute_dV_random_mode_end_to_end(tmp_path, monkeypatch):
    nat, naw = 2, 1
    tau = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    a_vectors = np.eye(3)
    alat = 6.0
    rng = np.random.default_rng(1)
    dV_true = rng.standard_normal((3 * nat, 2, 2, 1, 1, 1, 1))
    H0 = rng.standard_normal((2, 2, 1, 1, 1, 1))
    phi = rng.standard_normal((6, 6))
    phi = 0.5 * (phi + phi.T)

    patterns = rd.random_configurations(nat, 0.05, 8, seed=2)
    edir = tmp_path / 'elphon'
    disps = _write_random_manifest(str(edir), patterns, 0.05)
    for d in disps:
        u = np.asarray(d['displacement'])
        _write_forces_out(str(edir), d['input'], -(phi @ u.ravel()).reshape(nat, 3))
    by_prefix = {d['prefix']: np.asarray(d['displacement']) for d in disps}
    seen = {}

    def _build(savedir, **kwargs):
        prefix = os.path.basename(os.path.dirname(savedir)).replace('tmp_', '')
        seen[prefix] = kwargs.get('positions')
        if prefix == 'X_ref':
            hrs = np.array(H0)
            if kwargs.get('return_symmetry_inputs'):
                return hrs, {'tau': tau, 'a_vectors': a_vectors, 'alat': alat, 'naw': [naw] * nat}
            return hrs
        u = by_prefix[prefix].ravel()
        # Quadratic term: cancels in the antisymmetric difference.
        return H0 + np.tensordot(u, dV_true, axes=(0, 0)) + 0.3 * np.sum(u**2) * H0

    monkeypatch.setattr(dvscf_fd, 'build_supercell_HRs', _build)
    dc = _StubController(tmp_path)
    result = compute_dV(dc, elphon_dir='elphon', fixed_basis=True, verify_saves=False, ridge=0.0)

    assert result['displacement_mode'] == 'random'
    directional = result['directional']
    assert len(directional) == 3 * nat
    for k in range(nat):
        for alpha in range(3):
            e = directional[3 * k + alpha]
            assert e['sc_atom'] == k
            assert e['displacement'][alpha] == 1.0 and sum(e['displacement']) == 1.0
            np.testing.assert_allclose(e['dV'], dV_true[3 * k + alpha], atol=1e-9)
    assert result['fit']['relative_residual'] < 1e-6
    assert result['fit']['nconfig'] == 8 and result['fit']['nforce'] == 8
    # Fixed basis: every displaced build received the reference positions.
    assert all(np.allclose(v, tau) for p, v in seen.items() if p != 'X_ref')
    expect = phi.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3) * rd.RY_BOHR2_TO_EV_ANG2
    # Forces are parsed from text with 8 decimals: fc to ~1e-5 eV/A^2.
    np.testing.assert_allclose(result['force_constants'], expect, atol=1e-4)
    assert dc._arry['elphon_dV'] is result


def test_compute_dV_random_mode_needs_reference(tmp_path, monkeypatch):
    patterns = rd.random_configurations(2, 0.05, 2, seed=2)
    edir = tmp_path / 'elphon'
    _write_random_manifest(str(edir), patterns, 0.05)
    with open(edir / MANIFEST) as f:
        manifest = json.load(f)
    manifest['reference_prefix'] = None
    with open(edir / MANIFEST, 'w') as f:
        json.dump(manifest, f)
    monkeypatch.setattr(dvscf_fd, 'build_supercell_HRs', lambda savedir, **k: np.zeros((2, 2)))
    with pytest.raises(ValueError, match='reference'):
        compute_dV(_StubController(tmp_path), elphon_dir='elphon', verify_saves=False)


def _write_save_xml(edir, prefix, positions):
    """Minimal QE ``data-file-schema.xml`` with the given Cartesian positions."""
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


def test_verify_save_displacements_all_atom_patterns(tmp_path):
    edir = str(tmp_path)
    ref = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    u = np.array([[0.01, -0.02, 0.0], [0.0, 0.03, -0.01]])
    _write_save_xml(edir, 'X_ref', ref)
    _write_save_xml(edir, 'X_disp001', ref + u)
    _write_save_xml(edir, 'X_disp002', ref - u)
    disps = [
        {
            'sc_atom': None,
            'config': 0,
            'sign': 1,
            'displacement': u.tolist(),
            'prefix': 'X_disp001',
        },
        {
            'sc_atom': None,
            'config': 0,
            'sign': -1,
            'displacement': (-u).tolist(),
            'prefix': 'X_disp002',
        },
    ]
    dvscf_fd._verify_save_displacements(edir, disps, 'X_ref')  # no raise
    dvscf_fd._verify_save_displacements(edir, disps, None)  # pair check, no raise
    _write_save_xml(edir, 'X_disp002', ref + u)  # stale: not the mirror image
    with pytest.raises(ValueError, match='Stale/mismatched'):
        dvscf_fd._verify_save_displacements(edir, disps, 'X_ref')
