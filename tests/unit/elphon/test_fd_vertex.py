"""Unit tests of the finite-difference coupling source (``'fd'``).

Conventions of the cache array (index flip, units, mode ordering), the force
constants from Cartesian and random displaced runs (fake ``pw.x`` outputs), the
Gamma phonons of a cell from its force constants, the Gamma-mode mask and the
``ElphonSystem`` bookkeeping of the source.  No QE runtime.
"""

import json
import os

import numpy as np
import pytest

from PAOFLOW.elphon import fd_vertex as fv
from PAOFLOW.elphon.do_pao_eph import gamma_zero_mask
from PAOFLOW.elphon.elph_bloch import AMU_RY, RY_TO_EV, RY_TO_THZ
from PAOFLOW.elphon.io import MANIFEST


def test_flip_cells_is_minus_r_on_the_grid():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 2, 2, 3, 4))
    b = fv.flip_cells(a)
    for n1 in range(2):
        for n2 in range(3):
            for n3 in range(4):
                np.testing.assert_allclose(
                    b[..., n1, n2, n3], a[..., (-n1) % 2, (-n2) % 3, (-n3) % 4]
                )


def _directional(rng, natoms, nawf, grid):
    entries = []
    for k in range(natoms):
        for a in range(3):
            e = [0.0, 0.0, 0.0]
            e[a] = 1.0
            dV = rng.standard_normal((nawf, nawf) + grid + (1,)) + 1j * rng.standard_normal(
                (nawf, nawf) + grid + (1,)
            )
            entries.append({'sc_atom': k, 'displacement': e, 'dV': dV})
    return entries


def test_g_ReRp_from_fd_matches_the_kernel_convention():
    # Electrons: H(k) = sum_R exp(-2 pi i k.R) H(R); vertex: d(k) = sum_R exp(+2 pi i k.R) g(R).
    rng = np.random.default_rng(1)
    natoms, nawf, grid = 2, 3, (2, 3, 2)
    directional = _directional(rng, natoms, nawf, grid)
    g = fv.g_ReRp_from_fd(directional, natoms)
    assert g.shape == (nawf, nawf, 6) + grid + (1, 1, 1)
    # On the coarse grid the flip n -> (-n) mod N is exact; off-grid k are handled by the
    # Wigner-Seitz images of the kernel, which use the same cell labels as the electrons.
    cells = np.array(list(np.ndindex(*grid)), dtype=float)
    for k_cryst in (np.zeros(3), np.array([0.5, 1.0 / 3.0, 0.0]), np.array([0.5, 2.0 / 3.0, 0.5])):
        plus = np.exp(2j * np.pi * (cells @ k_cryst))
        minus = np.exp(-2j * np.pi * (cells @ k_cryst))
        for entry in directional:
            c = 3 * entry['sc_atom'] + int(np.argmax(entry['displacement']))
            dk_vertex = np.tensordot(
                g[:, :, c, ..., 0, 0, 0].reshape(nawf, nawf, -1), plus, axes=(2, 0)
            )
            dk_direct = np.tensordot(
                entry['dV'][..., 0].reshape(nawf, nawf, -1), minus, axes=(2, 0)
            )
            np.testing.assert_allclose(dk_vertex, dk_direct / RY_TO_EV, atol=1e-12)


def test_g_ReRp_from_fd_rejects_incomplete_or_oblique_sets():
    rng = np.random.default_rng(2)
    directional = _directional(rng, 2, 2, (1, 1, 1))
    with pytest.raises(ValueError, match='missing'):
        fv.g_ReRp_from_fd(directional[:-1], 2)
    oblique = dict(directional[0], displacement=[0.6, 0.8, 0.0])
    with pytest.raises(ValueError, match='Cartesian'):
        fv.g_ReRp_from_fd([oblique] + directional[1:], 2)
    # dV is the derivative per unit displacement whatever the length of the stored vector
    # (the Cartesian mode keeps +/- distance): the length is ignored, the sign is not.
    scaled = [dict(d, displacement=[0.06 * x for x in d['displacement']]) for d in directional]
    np.testing.assert_allclose(fv.g_ReRp_from_fd(scaled, 2), fv.g_ReRp_from_fd(directional, 2))
    flipped = [
        dict(d, displacement=[-0.06 * x for x in d['displacement']], dV=-d['dV'])
        for d in directional
    ]
    np.testing.assert_allclose(fv.g_ReRp_from_fd(flipped, 2), fv.g_ReRp_from_fd(directional, 2))


def _write_out(edir, name, forces):
    lines = ['     Forces acting on atoms (cartesian axes, Ry/au):', '']
    for i, f in enumerate(forces):
        lines.append('     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i + 1, *f))
    lines += ['', '     Total force =     0.000000     Total SCF correction =     0.000000']
    with open(os.path.join(edir, os.path.splitext(name)[0] + '.out'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _symmetric_phi(rng, nat):
    phi = rng.standard_normal((3 * nat, 3 * nat))
    return 0.5 * (phi + phi.T)


def _cartesian_manifest(edir, phi, nat, distance=0.05, drop=()):
    os.makedirs(edir, exist_ok=True)
    disps = []
    index = 0
    for k in range(nat):
        for a in range(3):
            for sign in (1, -1):
                if (k, a) in drop:
                    continue
                index += 1
                vec = [0.0, 0.0, 0.0]
                vec[a] = sign * distance
                disps.append(
                    {
                        'index': index - 1,
                        'sc_atom': k,
                        'displacement': vec,
                        'distance': distance,
                        'prefix': 'X_disp%03d' % index,
                        'input': 'disp-%03d.in' % index,
                    }
                )
                u = np.zeros(3 * nat)
                u[3 * k + a] = sign * distance
                _write_out(edir, disps[-1]['input'], (-(phi @ u)).reshape(nat, 3))
    manifest = {
        'displacement_distance': distance,
        'displacement_mode': 'cartesian',
        'reference_prefix': 'X_ref',
        'displacements': disps,
    }
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump(manifest, f)
    return manifest


def test_fd_force_constants_cartesian_pairs(tmp_path):
    rng = np.random.default_rng(3)
    nat = 2
    phi = _symmetric_phi(rng, nat)
    edir = str(tmp_path / 'elphon')
    _cartesian_manifest(edir, phi, nat)
    out = fv.fd_force_constants(edir)
    np.testing.assert_allclose(out, phi, atol=1e-5)
    # An atom without its z pair: incomplete -> None with a warning.
    edir2 = str(tmp_path / 'elphon2')
    _cartesian_manifest(edir2, phi, nat, drop=((1, 2),))
    with pytest.warns(UserWarning, match='force constants'):
        assert fv.fd_force_constants(edir2) is None


def test_fd_force_constants_random_mode(tmp_path):
    from PAOFLOW.elphon import random_displacements as rd

    rng = np.random.default_rng(4)
    nat = 2
    phi = _symmetric_phi(rng, nat)
    edir = str(tmp_path / 'elphon')
    os.makedirs(edir)
    patterns = rd.random_configurations(nat, 0.05, 8, seed=1)
    disps = []
    index = 0
    for m, u in enumerate(patterns):
        for sign in (1, -1):
            index += 1
            d = {
                'index': index - 1,
                'sc_atom': None,
                'config': m,
                'sign': sign,
                'displacement': (sign * u).tolist(),
                'distance': 0.05,
                'prefix': 'X_disp%03d' % index,
                'input': 'disp-%03d.in' % index,
            }
            disps.append(d)
            _write_out(edir, d['input'], (-(phi @ (sign * u).ravel())).reshape(nat, 3))
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump({'displacement_mode': 'random', 'displacements': disps}, f)
    out = fv.fd_force_constants(edir, ridge=0.0)
    np.testing.assert_allclose(out, phi, atol=1e-5)


def test_fd_phonon_at_q_gives_the_optical_mode_of_a_dimer():
    # Two atoms bound by an isotropic spring K: 3 acoustic modes and 3 optical at
    # omega = sqrt(2 K / M).
    K, mass = 0.02, 50.0
    blk = K * np.eye(3)
    phi = np.block([[blk, -blk], [-blk, blk]])
    phonon_at_q = fv.fd_phonon_at_q(phi, [mass, mass], np.eye(3))
    freq_thz, z = phonon_at_q(np.zeros(3))
    assert z.shape == (6, 6)
    expect = np.sqrt(2.0 * K / (mass * AMU_RY)) * RY_TO_THZ
    freq = np.sort(freq_thz)
    np.testing.assert_allclose(freq[:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(freq[3:], expect, rtol=1e-8)
    with pytest.raises(ValueError, match='atoms'):
        fv.fd_phonon_at_q(phi, [mass], np.eye(3))


def test_gamma_zero_mask():
    freq = np.array([-0.01, 0.0, 0.02, 1.5, 2.0])
    np.testing.assert_array_equal(gamma_zero_mask(freq, 'all'), np.ones(5, bool))
    np.testing.assert_array_equal(
        gamma_zero_mask(freq, 'acoustic', 0.1), [True, True, True, False, False]
    )
    with pytest.raises(ValueError):
        gamma_zero_mask(freq, 'optical')


def test_elphon_system_fd_source(tmp_path):
    from PAOFLOW.elphon.workflows import ElphonSystem

    with pytest.raises(ValueError, match='fd_dir'):
        ElphonSystem(
            prefix='x', root=str(tmp_path), masses_amu=[1.0], nelec=2, coupling_source='fd'
        )
    with pytest.raises(ValueError, match='Gamma-only'):
        ElphonSystem(
            prefix='x', root=str(tmp_path), masses_amu=[1.0], nelec=2, coupling_source='fd',
            fd_dir='fd/elphon', qgrid=(2, 2, 2),
        )  # fmt: skip
    system = ElphonSystem(
        prefix='x', root=str(tmp_path), masses_amu=[1.0], nelec=2, coupling_source='fd',
        fd_dir='fd/elphon', coarse_grid=(4, 4, 4), save_dir='fd/elphon/tmp_x_ref/x_ref.save',
        fd_options={'locality_cutoff': 7.0},
    )  # fmt: skip
    assert system.qgrid == (1, 1, 1)
    assert system.coupling_dir == os.path.join(str(tmp_path), 'fd', 'elphon')
    with pytest.raises(FileNotFoundError):
        system.check_inputs()
    os.makedirs(system.save_path)
    with pytest.raises(FileNotFoundError, match='displacements.json'):
        system.check_inputs()
    with open(os.path.join(system.coupling_dir, MANIFEST), 'w') as f:
        json.dump({'displacements': []}, f)
    system.check_inputs()  # no raise


def test_fd_cell_derivatives_rejects_unknown_options(tmp_path):
    with pytest.raises(ValueError, match='unknown'):
        fv.fd_cell_derivatives(str(tmp_path), nconfig=3)
