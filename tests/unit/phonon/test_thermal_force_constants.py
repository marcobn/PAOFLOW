"""Thermal force constants of a displaced supercell (A-SDM step): translation averaging,
phonopy objects from explicit force constants and the finite-difference force reader."""

import json
import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')
from phonopy import Phonopy  # noqa: E402
from phonopy.structure.atoms import PhonopyAtoms  # noqa: E402

from PAOFLOW.elphon.fd_vertex import fd_phonon_at_q  # noqa: E402
from PAOFLOW.elphon.fold import supercell_atom_translations  # noqa: E402
from PAOFLOW.elphon.io import MANIFEST  # noqa: E402
from PAOFLOW.phonon import special_displacement as sd  # noqa: E402


def _al_phonon(dims=(2, 1, 1)):
    alat = 7.6326928726
    cell = alat * np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
    unitcell = PhonopyAtoms(symbols=['Al'], cell=cell, positions=[[0.0, 0.0, 0.0]])
    return unitcell, Phonopy(
        unitcell, supercell_matrix=np.diag(dims), primitive_matrix='P', calculator='qe'
    )


def test_translation_average_is_invariant_and_idempotent():
    _, phonon = _al_phonon((2, 1, 1))
    s2p, translations = supercell_atom_translations(phonon)
    rng = np.random.default_rng(0)
    fc = rng.standard_normal((2, 2, 3, 3))
    avg = sd.translation_average_force_constants(fc, s2p, translations, 2)
    np.testing.assert_allclose(avg[0, 0], 0.5 * (fc[0, 0] + fc[1, 1]))
    np.testing.assert_allclose(avg[0, 1], 0.5 * (fc[0, 1] + fc[1, 0]))
    np.testing.assert_allclose(avg[1, 1], avg[0, 0])
    np.testing.assert_allclose(avg[1, 0], avg[0, 1])
    np.testing.assert_allclose(
        sd.translation_average_force_constants(avg, s2p, translations, (2, 1, 1)), avg
    )
    with pytest.raises(ValueError):
        sd.translation_average_force_constants(fc[:1], s2p, translations, 2)


def _write_out(edir, name, forces):
    lines = ['     Forces acting on atoms (cartesian axes, Ry/au):', '']
    for i, f in enumerate(forces):
        lines.append('     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i + 1, *f))
    lines += ['', '     Total force =     0.000000     Total SCF correction =     0.000000']
    with open(os.path.join(edir, os.path.splitext(name)[0] + '.out'), 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _cartesian_runs(edir, phi, nat, distance=0.05):
    os.makedirs(edir, exist_ok=True)
    disps, index = [], 0
    for k in range(nat):
        for a in range(3):
            for sign in (1, -1):
                index += 1
                vec = [0.0, 0.0, 0.0]
                vec[a] = sign * distance
                d = {
                    'index': index - 1, 'sc_atom': k, 'displacement': vec, 'distance': distance,
                    'prefix': 'X_disp%03d' % index, 'input': 'disp-%03d.in' % index,
                }  # fmt: skip
                disps.append(d)
                u = np.zeros(3 * nat)
                u[3 * k + a] = sign * distance
                _write_out(edir, d['input'], (-(phi @ u)).reshape(nat, 3))
    with open(os.path.join(edir, MANIFEST), 'w') as f:
        json.dump(
            {'displacement_mode': 'cartesian', 'reference_prefix': 'X_ref', 'displacements': disps},
            f,
        )


def test_thermal_force_constants_from_runs(tmp_path):
    unitcell, phonon = _al_phonon((2, 1, 1))
    rng = np.random.default_rng(1)
    phi = rng.standard_normal((6, 6))
    phi = 0.5 * (phi + phi.T)
    edir = str(tmp_path / 'elphon')
    _cartesian_runs(edir, phi, 2)
    raw = sd.thermal_force_constants(edir, phonon, translation_average=False)
    np.testing.assert_allclose(raw, phi.reshape(2, 3, 2, 3).transpose(0, 2, 1, 3), atol=1e-5)
    avg = sd.thermal_force_constants(edir, phonon)
    s2p, translations = supercell_atom_translations(phonon)
    np.testing.assert_allclose(
        avg, sd.translation_average_force_constants(raw, s2p, translations, 2), atol=1e-12
    )
    with pytest.raises(ValueError, match='atoms'):
        sd.thermal_force_constants(edir, _al_phonon((2, 2, 1))[1])


def test_phonopy_with_force_constants_matches_fd_phonons():
    unitcell, _ = _al_phonon((2, 1, 1))
    K, mass = 0.02, 26.98
    blk = K * np.eye(3)
    fc = np.array([[blk, -blk], [-blk, blk]])  # (2, 2, 3, 3), Ry/bohr^2
    phonon = sd.phonopy_with_force_constants(unitcell, (2, 1, 1), fc)
    # The 6 Gamma modes of the supercell are the primitive modes at q = 0 and q = (1/2, 0, 0).
    phonon.run_qpoints([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    freq = np.sort(np.asarray(phonon.qpoints.frequencies).ravel())
    phi = fc.transpose(0, 2, 1, 3).reshape(6, 6)
    expect = np.sort(fd_phonon_at_q(phi, [mass, mass], np.eye(3))(np.zeros(3))[0])
    np.testing.assert_allclose(freq[3:], expect[3:], rtol=1e-3)
    np.testing.assert_allclose(freq[:3], 0.0, atol=1e-4)
    with pytest.raises(ValueError, match='supercell'):
        sd.phonopy_with_force_constants(unitcell, 2, fc)
