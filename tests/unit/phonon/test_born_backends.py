"""Unit tests for the Stage 2b Born-charge / dielectric back-ends.

Covers the DFPT (``ph.x``) input writer and output parser, the finite-field
(``lelfield``) harvest bookkeeping, and the shared ``compute_born_and_epsilon``
post-processing (acoustic sum rule + symmetrization).  No Quantum ESPRESSO
runtime is required: QE outputs are fabricated as text fixtures.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

phonopy = pytest.importorskip('phonopy')

from PAOFLOW.phonon.do_born_charges import compute_born_and_epsilon
from PAOFLOW.phonon.do_phonopy import generate_displacements, init_phonopy
from PAOFLOW.phonon.io import (
    _E_CHARGE,
    _QE_EFIELD_AU_TO_V_PER_M,
    _RY_BOHR_TO_N,
    harvest_field_results,
    harvest_ph_results,
    read_hubbard_card,
    write_displaced_supercells,
    write_field_inputs,
    write_ph_epsil_input,
)


class _StubController:
    def __init__(self, arry, attr, rank=0):
        self._arry = arry
        self._attr = attr
        self.rank = rank

    def data_dicts(self):
        return self._arry, self._attr


def _gaas_controller(tmp_path):
    alat = 10.6829  # Bohr
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]], dtype=float)
    cell_bohr = a_vectors * alat
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=float) @ cell_bohr
    omega = alat**3 * a_vectors[0].dot(np.cross(a_vectors[1], a_vectors[2]))
    arry = {
        'a_vectors': a_vectors,
        'tau': tau,
        'atoms': ['Ga', 'As'],
        'species': [('Ga', 'Ga_ONCV_sr.upf'), ('As', 'As_ONCV_sr.upf')],
    }
    attr = {
        'alat': alat,
        'natoms': 2,
        'omega': omega,
        'opath': str(tmp_path),
        'savedir': os.path.join(str(tmp_path), 'GaAs.save'),
        'fpath': str(tmp_path),
        'nspin': 1,
        'insulator': True,
        'nk1': 4,
        'nk2': 4,
        'nk3': 4,
        'phonon_supercell_matrix': 2,
        'phonon_displacement_distance': 0.06,
        'verbose': False,
    }
    return _StubController(arry, attr)


_PH_OUTPUT = """
     Dielectric constant in cartesian axis

          (      10.900000000       0.000000000       0.000000000 )
          (       0.000000000      10.900000000       0.000000000 )
          (       0.000000000       0.000000000      10.900000000 )

     Effective charges (d Force / dE) in cartesian axis with asr applied:

           atom      1   Ga
      Ex  (        2.18000        0.00000        0.00000 )
      Ey  (        0.00000        2.18000        0.00000 )
      Ez  (        0.00000        0.00000        2.18000 )
           atom      2   As
      Ex  (       -2.10000        0.00000        0.00000 )
      Ey  (        0.00000       -2.10000        0.00000 )
      Ez  (        0.00000        0.00000       -2.10000 )

     Effective charges (d P / du) in cartesian axis

           atom      1   Ga
      Px  (        9.99900        0.00000        0.00000 )
"""


def test_write_ph_epsil_input_content(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)

    path = write_ph_epsil_input(dc, phonon_dir='phonon')
    assert os.path.isfile(path)
    text = open(path).read()
    assert 'epsil = .true.' in text
    assert 'trans = .false.' in text
    assert "prefix = 'GaAs'" in text
    # ph.x must read the existing DFT save (outdir = parent of GaAs.save).
    assert "outdir = '%s'" % str(tmp_path) in text


def test_harvest_ph_results_parses_dielectric_and_born(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    phonon_dir = os.path.join(str(tmp_path), 'phonon')
    os.makedirs(phonon_dir, exist_ok=True)
    with open(os.path.join(phonon_dir, 'ph_epsil.out'), 'w') as f:
        f.write(_PH_OUTPUT)

    res = harvest_ph_results(dc, phonon_dir='phonon')
    np.testing.assert_allclose(res['dielectric'], np.eye(3) * 10.9, atol=1e-8)
    assert res['born'].shape == (2, 3, 3)
    np.testing.assert_allclose(res['born'][0], np.eye(3) * 2.18, atol=1e-8)
    np.testing.assert_allclose(res['born'][1], np.eye(3) * (-2.10), atol=1e-8)


def test_compute_born_dfpt_sum_rule_and_symmetrize(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    phonon_dir = os.path.join(str(tmp_path), 'phonon')
    os.makedirs(phonon_dir, exist_ok=True)
    with open(os.path.join(phonon_dir, 'ph_epsil.out'), 'w') as f:
        f.write(_PH_OUTPUT)

    out = compute_born_and_epsilon(
        dc, method='dfpt', phonon_dir='phonon', enforce_sum_rule=True, symmetrize=True
    )
    born = out['born']
    # Acoustic sum rule: sum over atoms is zero after enforcement.
    np.testing.assert_allclose(born.sum(axis=0), np.zeros((3, 3)), atol=1e-10)
    # Raw drift (2.18 - 2.10 = 0.08) split evenly -> 0.04 shift per atom.
    np.testing.assert_allclose(born[0], np.eye(3) * (2.18 - 0.04), atol=1e-8)
    np.testing.assert_allclose(born[1], np.eye(3) * (-2.10 - 0.04), atol=1e-8)
    # BORN file written.
    assert os.path.isfile(os.path.join(phonon_dir, 'BORN'))


def _force_block(forces):
    lines = ['     Forces acting on atoms (cartesian axes, Ry/au):', '']
    for i, f in enumerate(forces, start=1):
        lines.append('     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i, f[0], f[1], f[2]))
    lines.append('')
    lines.append('     Total force =     0.000000')
    return '\n'.join(lines) + '\n'


def test_harvest_field_results_central_difference(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    phonon_dir = os.path.join(str(tmp_path), 'phonon')
    os.makedirs(phonon_dir, exist_ok=True)

    e_mag = 0.001
    # Target diagonal Born charges +/- z_target for Ga / As.
    z_target = 2.18
    # Invert the harvest formula to fabricate the forces:
    #   born = dF * _RY_BOHR_TO_N / (e * E_au_to_Vm),  dF = (F+ - F-)/(2 e_mag)
    e_si_per_au = _E_CHARGE * _QE_EFIELD_AU_TO_V_PER_M
    dF = z_target * e_si_per_au / _RY_BOHR_TO_N  # Ry/bohr per a.u.
    f_plus_mag = e_mag * dF  # antisymmetric: F- = -F+

    # Zero-field reference.
    open(os.path.join(phonon_dir, 'field-0.out'), 'w').write(_force_block([[0, 0, 0], [0, 0, 0]]))
    for b, axis in enumerate(('x', 'y', 'z')):
        fp = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        fm = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        fp[0][b] = f_plus_mag  # Ga -> +z_target
        fm[0][b] = -f_plus_mag
        fp[1][b] = -f_plus_mag  # As -> -z_target
        fm[1][b] = f_plus_mag
        open(os.path.join(phonon_dir, 'field-%s+.out' % axis), 'w').write(_force_block(fp))
        open(os.path.join(phonon_dir, 'field-%s-.out' % axis), 'w').write(_force_block(fm))
        # The harvest reads the field magnitude from the '+' input file.
        open(os.path.join(phonon_dir, 'field-%s+.in' % axis), 'w').write(
            'efield_cart(%d) = %.8f\n' % (b + 1, e_mag)
        )

    res = harvest_field_results(dc, phonon_dir='phonon')
    np.testing.assert_allclose(res['born'][0], np.eye(3) * z_target, atol=1e-4)
    np.testing.assert_allclose(res['born'][1], np.eye(3) * (-z_target), atol=1e-4)
    # No polarization in the fabricated outputs -> isotropic fallback dielectric.
    assert res['dielectric'].shape == (3, 3)


# --- new-style HUBBARD card (U-only) injection -----------------------------

_GAAS_SCF_IN = """&control
    calculation = 'scf'
    prefix = 'GaAs'
/
&system
    ibrav = 2
    nat = 2
    ntyp = 2
    ecutwfc = 40
/
&electrons
/
ATOMIC_SPECIES
 Ga  69.723  Ga_ONCV_sr.upf
 As  74.922  As_ONCV_sr.upf
ATOMIC_POSITIONS crystal
 Ga 0.0 0.0 0.0
 As 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
HUBBARD (ortho-atomic)
 U Ga-4s 0.0
 U Ga-4p 0.3911078136627079
 U As-4s 0.0
 U As-4p 2.118807310520171
 V Ga-4s As-4s 1 2 1.0734882401131423
 V Ga-4s As-4p 1 2 1.1713950230473542
 V Ga-4p As-4p 1 2 1.797846550033977
"""


def _write_scf_in(tmp_path):
    path = os.path.join(str(tmp_path), 'scf.in')
    with open(path, 'w') as f:
        f.write(_GAAS_SCF_IN)
    return path


def test_read_hubbard_card_u_only_drops_v(tmp_path):
    path = _write_scf_in(tmp_path)
    card = read_hubbard_card(path, include_v=False)
    assert card is not None
    assert card.startswith('HUBBARD (ortho-atomic)')
    # All four on-site U manifold lines kept.
    assert 'U Ga-4s 0.0' in card
    assert 'U Ga-4p 0.3911078136627079' in card
    assert 'U As-4p 2.118807310520171' in card
    # All intersite V lines dropped.
    assert ' V ' not in card and '\n V' not in card
    assert card.count('\n U ') == 4


def test_read_hubbard_card_include_v_keeps_all(tmp_path):
    path = _write_scf_in(tmp_path)
    card = read_hubbard_card(path, include_v=True)
    assert card.count('\n U ') == 4
    assert card.count('\n V ') == 3


def test_read_hubbard_card_absent_returns_none(tmp_path):
    path = os.path.join(str(tmp_path), 'plain.in')
    with open(path, 'w') as f:
        f.write('&control\n/\n&system\n    nat = 1\n/\n')
    assert read_hubbard_card(path) is None


def test_supercell_inputs_contain_u_only_card(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    generate_displacements(dc)
    card = read_hubbard_card(_write_scf_in(tmp_path), include_v=False)

    paths = write_displaced_supercells(dc, phonon_dir='phonon', hubbard_card=card)
    assert paths
    text = open(paths[0]).read()
    assert 'HUBBARD (ortho-atomic)' in text
    assert 'U As-4p 2.118807310520171' in text
    # No intersite V lines (atom indices invalid for the supercell).
    assert ' V ' not in text
    # Card sits after the K_POINTS card.
    assert text.index('HUBBARD') > text.index('K_POINTS')


def test_field_inputs_contain_u_only_card(tmp_path):
    dc = _gaas_controller(tmp_path)
    init_phonopy(dc)
    generate_displacements(dc)
    card = read_hubbard_card(_write_scf_in(tmp_path), include_v=False)

    paths = write_field_inputs(dc, phonon_dir='phonon', hubbard_card=card)
    assert paths
    text = open(paths[0]).read()
    assert 'HUBBARD (ortho-atomic)' in text
    assert 'U Ga-4p 0.3911078136627079' in text
    assert ' V ' not in text
