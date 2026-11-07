"""Unit tests of the SSCHA bridge (python-sscha + CellConstructor) and of the SSCHA phonon source of ElphonSystem.

Fixtures (``data/sscha_tan``): delta-TaN on the 2x2x2 grid (Gamma, L, X), the harmonic DFPT matrices with Fermi-Dirac
electrons at 2800 K (unstable at X: -223 cm^-1) and the converged SSCHA (auxiliary) matrices at 2800 K (stable).
The key test is analytic: with harmonic forces F = -Phi u the SSCHA gradient vanishes exactly, so the minimisation
must return Phi and the free-energy Hessian must equal Phi, at any temperature.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.unit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data', 'sscha_tan')
HARMONIC = os.path.join(DATA, 'harmonic', 'tan.dyn')
SSCHA = os.path.join(DATA, 'sscha_T2800', 'tan.dyn')
THZ_TO_CM1 = 33.35641
FCC_AT = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])  # alat
X_CRYST = np.array([0.0, 0.5, 0.5])  # X = (0, 1, 0) 2 pi / a


def test_supercell_qpoint_files_gamma_first():
    from PAOFLOW.phonon.sscha_bridge import supercell_qpoint_files

    files = supercell_qpoint_files(HARMONIC, (2, 2, 2))
    assert [os.path.basename(f) for f in files] == ['tan.dyn1', 'tan.dyn2', 'tan.dyn3']
    assert [os.path.basename(f) for f in supercell_qpoint_files(HARMONIC, (1, 1, 1))] == [
        'tan.dyn1'
    ]


def test_fildyn_prefix_must_end_in_dyn():
    from PAOFLOW.phonon.sscha_bridge import supercell_qpoint_files

    with pytest.raises(ValueError, match='end in .dyn'):
        supercell_qpoint_files(os.path.join(DATA, 'harmonic', 'tan'), (2, 2, 2))


@pytest.fixture(scope='module')
def cc_phonons():
    pytest.importorskip('sscha')
    pytest.importorskip('cellconstructor')
    from PAOFLOW.phonon.sscha_bridge import require_sscha

    return require_sscha()[0]


def test_starting_matrices_are_positive_definite(cc_phonons, tmp_path):
    from PAOFLOW.phonon.sscha_bridge import starting_matrices

    matrices, harmonic_cm1, start_cm1 = starting_matrices(
        HARMONIC, (2, 2, 2), str(tmp_path / 'start')
    )
    assert harmonic_cm1[0] == pytest.approx(-223.4, abs=0.2)  # the X instability
    assert start_cm1[0] > -1.0e-3 and start_cm1[3] > 50.0  # translations at zero, the rest positive
    assert sorted(os.listdir(tmp_path / 'start')) == ['dyn1', 'dyn2', 'dyn3']


class _StubController:
    def __init__(self, arry, attr):
        self._arry, self._attr = arry, attr

    def data_dicts(self):
        return self._arry, self._attr


def test_fermi_dirac_lines():
    from PAOFLOW.phonon.io import fermi_dirac_lines

    lines = fermi_dirac_lines(2800.0)
    assert lines[1] == "    smearing = 'fd'" and lines[2] == '    degauss = 0.01773352'
    with pytest.raises(ValueError, match='positive'):
        fermi_dirac_lines(0.0)


def test_missing_configurations(tmp_path):
    from PAOFLOW.phonon.sscha_bridge import completed_outputs, missing_configurations

    for index in range(3):
        (tmp_path / ('cfg_%04d.in' % index)).write_text('input')
    (tmp_path / 'cfg_0000.out').write_text('Total force = 0.1\n JOB DONE.')
    (tmp_path / 'cfg_0001.out').write_text('partial: killed by the scheduler')
    assert [os.path.basename(p) for p in missing_configurations(str(tmp_path))] == [
        'cfg_0001.in',
        'cfg_0002.in',
    ]
    outputs, n_inputs = completed_outputs(str(tmp_path))
    assert [os.path.basename(p) for p in outputs] == ['cfg_0000.out'] and n_inputs == 3


def test_configuration_input_carries_the_parent_occupations(cc_phonons):
    import sscha.Ensemble

    from PAOFLOW.phonon.sscha_bridge import configuration_input

    matrices = cc_phonons.Phonons(SSCHA, 3)
    ensemble = sscha.Ensemble.Ensemble(matrices, 2800.0, matrices.GetSupercell())
    ensemble.generate(2)
    controller = _StubController(
        {'species': [('N', 'N.upf'), ('Ta', 'Ta.upf')]},
        {
            'insulator': False,
            'qe_smearing': 'fd',
            'qe_degauss': 0.017734,
            'ecutwfc': 84.0,
            'nspin': 1,
        },
    )
    text = configuration_input(
        controller, ensemble.structures[0], 'tan_sscha_p1_0000', '/pseudo', (6, 6, 6)
    )
    for expected in (
        "smearing = 'fd'",
        'degauss = 0.01773400',
        'nat = 16',
        'ibrav = 0',
        'tprnfor = .true.',
        'conv_thr = 1.0d-10',
        'mixing_beta = 0.3',
        'CELL_PARAMETERS angstrom',
        'ATOMIC_POSITIONS angstrom',
        '6 6 6 0 0 0',
        'Ta.upf',
        'N.upf',
        "outdir = './tmp_tan_sscha_p1_0000'",
    ):
        assert expected in text, expected
    positions = (
        text.split('ATOMIC_POSITIONS angstrom\n')[1].split('K_POINTS')[0].strip().splitlines()
    )
    assert len(positions) == 16
    assert np.allclose(
        [[float(x) for x in line.split()[1:]] for line in positions], ensemble.structures[0].coords
    )


def _harmonic_calculator(matrices):
    """ASE calculator with the harmonic forces F = -Phi u of the supercell of ``matrices`` (eV, eV / A)."""
    from ase.calculators.calculator import Calculator, all_changes

    supercell_dyn = matrices.GenerateSupercellDyn(matrices.GetSupercell())
    phi = (
        np.real(supercell_dyn.dynmats[0]) * 13.605693122994 / 0.529177210903**2
    )  # Ry/bohr^2 -> eV/A^2
    reference = np.asarray(supercell_dyn.structure.coords, dtype=float)

    class Harmonic(Calculator):
        implemented_properties = ['energy', 'forces']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            u = (atoms.get_positions() - reference).reshape(-1)
            force = -phi @ u
            self.results = {'energy': 0.5 * float(u @ phi @ u), 'forces': force.reshape(-1, 3)}

    return Harmonic()


def test_harmonic_forces_return_the_harmonic_matrices(cc_phonons):
    import sscha.Ensemble

    from PAOFLOW.phonon.sscha_bridge import RY_TO_CM1, free_energy_hessian, minimise

    matrices = cc_phonons.Phonons(SSCHA, 3)
    before = np.sort(matrices.DiagonalizeSupercell()[0] * RY_TO_CM1)
    ensemble = sscha.Ensemble.Ensemble(matrices, 2800.0, matrices.GetSupercell())
    ensemble.generate(40)
    ensemble.compute_ensemble(_harmonic_calculator(matrices), compute_stress=False)
    result = minimise(ensemble)
    assert np.allclose(result['frequencies_cm1'][3:], before[3:], rtol=1.0e-3)
    hessian = free_energy_hessian(ensemble)
    after = np.sort(hessian.DiagonalizeSupercell()[0] * RY_TO_CM1)
    assert np.allclose(after[3:], before[3:], rtol=1.0e-3)


def _frequencies_by_crystal_q(matrices) -> list[tuple[np.ndarray, np.ndarray]]:
    """(q in crystal coordinates, sorted frequencies in cm^-1) of every q of a CellConstructor ``Phonons``."""
    from PAOFLOW.phonon.sscha_bridge import RY_TO_CM1

    cell = np.asarray(matrices.structure.unit_cell)
    return [
        (np.asarray(q) @ cell.T, np.sort(matrices.DyagDinQ(iq)[0] * RY_TO_CM1))
        for iq, q in enumerate(matrices.q_tot)
    ]


def test_interpolated_correction_is_exact_on_the_supercell_grid(cc_phonons):
    from PAOFLOW.phonon.sscha_bridge import interpolate_correction

    harmonic_coarse = cc_phonons.Phonons(HARMONIC, 3)
    sscha_matrices = cc_phonons.Phonons(SSCHA, 3)
    harmonic_fine = harmonic_coarse.Interpolate((2, 2, 2), (4, 4, 4))
    fine = _frequencies_by_crystal_q(
        interpolate_correction(sscha_matrices, harmonic_coarse, harmonic_fine, (2, 2, 2), (4, 4, 4))
    )
    assert len(fine) == 64
    for q, w in _frequencies_by_crystal_q(
        sscha_matrices
    ):  # the 8 q of the 2x2x2 grid, matched modulo G
        match = [wf for qf, wf in fine if np.allclose(qf - np.round(qf - q), q, atol=1.0e-5)]
        assert len(match) == 1
        assert np.allclose(match[0], w, atol=0.05)
    same = interpolate_correction(
        harmonic_coarse, harmonic_coarse, harmonic_fine, (2, 2, 2), (4, 4, 4)
    )
    for (_, w_same), (_, w_harm) in zip(
        _frequencies_by_crystal_q(same), _frequencies_by_crystal_q(harmonic_fine)
    ):
        assert np.allclose(w_same, w_harm, atol=1.0e-6)


def test_written_force_constants_reproduce_the_matrices(cc_phonons, tmp_path):
    from PAOFLOW.phonon.qe_force_constants import (
        phonon_interpolator,
        read_dynamical_matrices,
        real_space_force_constants,
    )
    from PAOFLOW.phonon.sscha_bridge import write_force_constants

    matrices = cc_phonons.Phonons(SSCHA, 3)
    prefix, fc_path = write_force_constants(
        matrices, str(tmp_path / 'fc' / 'tan.dyn'), (2, 2, 2), str(tmp_path / 'tan.fc')
    )
    assert os.path.isfile(prefix + '0') and os.path.getsize(fc_path) > 0
    dynamical = read_dynamical_matrices('tan', str(tmp_path / 'fc'))
    interpolate = phonon_interpolator(
        real_space_force_constants(dynamical['force_constants_q']),
        dynamical['masses_ry'],
        dynamical['qgrid'],
        dynamical['at'],
    )
    x_cm1 = np.sort(interpolate(X_CRYST)[0]) * THZ_TO_CM1
    assert x_cm1[:3] == pytest.approx([100.3, 103.6, 103.6], abs=0.1)


def _system(**kwargs):
    from PAOFLOW.elphon.workflows.system import ElphonSystem

    return ElphonSystem(
        prefix='tan',
        root=DATA,
        masses_amu=[14.007, 180.9479],
        nelec=18,
        coarse_grid=(2, 2, 2),
        qgrid=(2, 2, 2),
        long_range=False,
        **kwargs,
    )


class _Electrons:
    nscf = {
        'at': FCC_AT,
        'bg': np.linalg.inv(FCC_AT).T,
        'tau_cryst': np.array([[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0]]),
        'orbital_positions': None,
    }


def test_elphon_system_sscha_phonon_source():
    assert _system().phonon_interpolator(_Electrons()) is None
    interpolate = _system(
        phonon_source='sscha', sscha_dynamical_matrices=SSCHA
    ).phonon_interpolator(_Electrons())
    x_cm1 = np.sort(interpolate(X_CRYST)[0]) * THZ_TO_CM1
    assert x_cm1[:3] == pytest.approx([100.3, 103.6, 103.6], abs=0.1)


def test_elphon_system_validates_the_phonon_source():
    with pytest.raises(ValueError, match='phonon_source'):
        _system(phonon_source='harmonic')
    with pytest.raises(ValueError, match='sscha_dynamical_matrices'):
        _system(phonon_source='sscha')


BOHR_ANGSTROM = 0.529177210903


def _fake_pw_output(path: str, phi: np.ndarray, reference_bohr: np.ndarray) -> None:
    """pw.x stand-in: read ``cfg_XXXX.in`` and write ``cfg_XXXX.out`` with the harmonic forces -Phi u (Ry, bohr)."""
    text = open(path).read()
    cell = text.split('CELL_PARAMETERS angstrom\n')[1].splitlines()[:3]
    cell = np.array([[float(x) for x in row.split()] for row in cell]) / BOHR_ANGSTROM
    rows = text.split('ATOMIC_POSITIONS angstrom\n')[1].split('K_POINTS')[0].strip().splitlines()
    symbols = [row.split()[0] for row in rows]
    positions = np.array([[float(x) for x in row.split()[1:]] for row in rows]) / BOHR_ANGSTROM
    u = (positions - reference_bohr).reshape(-1)
    force = -(phi @ u)
    lines = [
        '     Program PWSCF v.7.5 starts on  7Nov2026',
        '     celldm(1)=   1.000000  celldm(2)=   0.000000',
        '     number of atoms/cell      = %12d' % len(symbols),
        '     number of atomic types    =            2',
        '     crystal axes: (cart. coord. in units of alat)',
    ]
    lines += [
        '               a(%d) = ( %12.8f %12.8f %12.8f )' % (i + 1, *row)
        for i, row in enumerate(cell)
    ]
    lines += ['', '     site n.     atom                  positions (alat units)']
    lines += [
        '     %4d           %-3s tau(%4d) = ( %14.9f %14.9f %14.9f  )' % (i + 1, s, i + 1, *p)
        for i, (s, p) in enumerate(zip(symbols, positions))
    ]
    lines += [
        '',
        '!    total energy              = %18.8f Ry' % (0.5 * u @ phi @ u),
        '',
        '     Forces acting on atoms (cartesian axes, Ry/au):',
        '',
    ]
    lines += [
        '     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i + 1, *f)
        for i, f in enumerate(force.reshape(-1, 3))
    ]
    lines += [
        '',
        '     Total force = %12.6f     Total SCF correction =     0.000000' % np.linalg.norm(force),
        '',
        '   JOB DONE.',
    ]
    with open(path[:-3] + '.out', 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


def test_orchestrator_converges_to_the_harmonic_model(cc_phonons, tmp_path):
    """PAOFLOW.sscha_* on a harmonic model: start from the unstable harmonic matrices (made positive), forces from
    the stable SSCHA fixture taken as an exact harmonic potential; the run must converge to that potential, the
    Hessian must be stable and equal to it, and the exported force constants must reproduce it."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.phonon.qe_force_constants import (
        phonon_interpolator,
        read_dynamical_matrices,
        real_space_force_constants,
    )
    from PAOFLOW.phonon.sscha_bridge import RY_TO_CM1

    truth = cc_phonons.Phonons(SSCHA, 3)
    supercell_dyn = truth.GenerateSupercellDyn(truth.GetSupercell())
    phi = np.real(supercell_dyn.dynmats[0])  # Ry / bohr^2
    reference = np.asarray(supercell_dyn.structure.coords) / BOHR_ANGSTROM
    expected = np.sort(truth.DiagonalizeSupercell()[0] * RY_TO_CM1)[3:9]

    pao = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': 2.7, 'delta': 1.0},
        outputdir=str(tmp_path / 'out'),
        verbose=False,
    )
    arry, attr = pao.data_controller.data_dicts()
    arry['species'] = [('N', 'N.upf'), ('Ta', 'Ta.upf')]
    attr.update(
        insulator=False,
        qe_smearing='fd',
        qe_degauss=0.017734,
        ecutwfc=84.0,
        nspin=1,
        abort_on_exception=True,
    )
    sscha_dir = str(tmp_path / 'sscha')
    setup = dict(
        sscha_dir=sscha_dir,
        kgrid=(2, 2, 2),
        prefix='tan',
        pp_dir='/pseudo',
        population_sizes=[100] * 8,
        hessian_size=100,
    )
    np.random.seed(7)
    pao.sscha_setup(2800.0, 2, HARMONIC, **setup)
    assert pao._sscha_state()['harmonic_cm1'][0] == pytest.approx(-223.4, abs=0.2)

    first = pao.sscha_population()
    assert (
        pao.sscha_population()['inputs'] == first['inputs']
    )  # pending: returned again, not regenerated
    with pytest.raises(RuntimeError, match='0 of 100 configurations finished'):
        pao.sscha_minimize()  # no forces yet
    assert pao._sscha_state()['minimized'] == 0

    population = first
    while not population['hessian']:
        for path in population['inputs']:
            _fake_pw_output(path, phi, reference)
        result = pao.sscha_minimize()
        population = pao.sscha_population()
    assert result['converged'] and result['population'] <= 8
    assert np.allclose(result['lowest_cm1'], expected, rtol=1.0e-3)

    pao.sscha_setup(2800.0, 2, HARMONIC, **setup)  # continuing the same run keeps its state
    assert pao._sscha_state()['minimized'] == result['population']
    assert pao.sscha_status()['stage'] == 'forces'  # the Hessian population
    with pytest.raises(ValueError, match='use another sscha_dir'):
        pao.sscha_setup(2000.0, 2, HARMONIC, **setup)  # never overwrite another run
    pao.sscha_setup(2800.0, 2, HARMONIC, **setup)

    for path in population['inputs']:
        _fake_pw_output(path, phi, reference)
    assert pao.sscha_status()['stage'] == 'hessian'
    hessian = pao.sscha_hessian()
    assert pao.sscha_status()['stage'] == 'export'
    assert hessian['stable']
    assert min(w for _, w in hessian['lowest_cm1']) == pytest.approx(expected[0], rel=1.0e-3)

    exported = pao.sscha_force_constants()
    assert (
        exported['unstable_q'] == 0
        and attr['sscha_force_constants_file'] == exported['force_constants_file']
    )
    folder, name = os.path.split(exported['dynamical_matrices'])
    dynamical = read_dynamical_matrices(name[: -len('.dyn')], folder)
    interpolate = phonon_interpolator(
        real_space_force_constants(dynamical['force_constants_q']),
        dynamical['masses_ry'],
        dynamical['qgrid'],
        dynamical['at'],
    )
    assert np.sort(interpolate(X_CRYST)[0])[:3] * THZ_TO_CM1 == pytest.approx(
        [100.3, 103.6, 103.6], abs=0.1
    )
