"""Unit tests of ``PAOFLOW.elphon.workflows`` (system configuration and the pure helpers)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from PAOFLOW.elphon.workflows import (
    ElphonSystem,
    absorption_label,
    find_system_file,
    load_system,
    mobility_label,
    smearing_label,
    statistics_scissor,
    write_eliashberg,
)
from PAOFLOW.elphon.workflows import arpes as wa


def _system(tmp_path, **kw):
    options = dict(prefix='si', root=str(tmp_path), masses_amu=[28.085], nelec=8)
    options.update(kw)
    return ElphonSystem(**options)


def test_paths_default_to_the_qe_layout(tmp_path):
    system = _system(tmp_path, coarse_grid=(6, 6, 6), qgrid=(3, 3, 3))
    root = str(tmp_path)
    assert system.qe_path == os.path.join(root, 'qe')
    assert system.phonon_path == os.path.join(root, 'qe', 'phonon')
    assert system.nscf_path == os.path.join(root, 'qe', 'nscf')
    assert system.optics_path == os.path.join(root, 'qe', 'optics')
    assert system.save_path == os.path.join(root, 'qe', 'nscf', 'si.save')
    assert system.coupling_dir == system.phonon_path
    assert system.basis_path == os.path.join(root, 'BASIS_PS')
    assert system.cache_path == os.path.join(root, 'cache')
    assert system.quadrupoles_path is None
    assert system.masses_amu == [28.085]


def test_path_overrides_and_absolute_paths(tmp_path):
    system = _system(
        tmp_path, coupling_source='epw', phonon_dir='/data/ph', nscf_dir='runs/nscf',
        save_dir='/data/si.save', basis_dir=None, cache_dir=None, quadrupoles='coupling/quadrupole.fmt',
    )  # fmt: skip
    assert system.phonon_path == '/data/ph'
    assert system.nscf_path == os.path.join(str(tmp_path), 'runs', 'nscf')
    assert system.save_path == '/data/si.save'
    assert system.coupling_dir == system.nscf_path  # EPW reads the .epb files of the nscf run
    assert system.basis_path is None and system.cache_path is None
    assert system.quadrupoles_path == os.path.join(str(tmp_path), 'coupling', 'quadrupole.fmt')


def test_qgrid_defaults_to_the_k_grid_and_must_divide_it(tmp_path):
    assert _system(tmp_path, coarse_grid=(4, 4, 4)).qgrid == (4, 4, 4)
    with pytest.raises(ValueError, match='divide'):
        _system(tmp_path, coarse_grid=(6, 6, 6), qgrid=(4, 4, 4))
    with pytest.raises(ValueError, match='coupling_source'):
        _system(tmp_path, coupling_source='ahc')


def test_check_inputs_reports_what_is_missing(tmp_path):
    system = _system(tmp_path)
    with pytest.raises(FileNotFoundError, match='nscf'):
        system.check_inputs()
    os.makedirs(system.save_path)
    with pytest.raises(FileNotFoundError, match='ph.x'):
        system.check_inputs()
    os.makedirs(os.path.join(system.phonon_path, '_ph0'))
    system.check_inputs()
    epw = _system(tmp_path, coupling_source='epw')
    with pytest.raises(FileNotFoundError, match='epb'):
        epw.check_inputs()
    open(os.path.join(epw.nscf_path, 'si.epb1'), 'w').close()
    epw.check_inputs()


def test_find_and_load_system(tmp_path):
    (tmp_path / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='pb', root=__file__.rsplit('/', 1)[0], masses_amu=[207.2], nelec=14)\n"
        'TRANSPORT = dict(nk=4)\n'
    )
    driver_dir = tmp_path / 'transport'
    driver_dir.mkdir()
    driver = driver_dir / 'mobility.py'
    driver.write_text('')
    assert find_system_file(str(driver)) == str(tmp_path / 'system.py')
    assert find_system_file(str(driver_dir)) == str(tmp_path / 'system.py')
    module = load_system(str(driver))
    assert module.SYSTEM.prefix == 'pb' and module.TRANSPORT == {'nk': 4}
    assert module.SYSTEM.root == str(tmp_path)
    # a driver linked into a run directory finds the run directory's system.py
    run = tmp_path / 'run'
    (run / 'transport').mkdir(parents=True)
    (run / 'system.py').write_text(
        'from PAOFLOW.elphon.workflows import ElphonSystem\n'
        "SYSTEM = ElphonSystem(prefix='run', root='.', masses_amu=[1.0], nelec=2)\n"
    )
    link = run / 'transport' / 'mobility.py'
    os.symlink(driver, link)
    assert load_system(str(link)).SYSTEM.prefix == 'run'
    with pytest.raises(FileNotFoundError):
        find_system_file('/')
    (tmp_path / 'bad').mkdir()
    (tmp_path / 'bad' / 'system.py').write_text('X = 1\n')
    with pytest.raises(AttributeError, match='SYSTEM'):
        load_system(str(tmp_path / 'bad'))


def test_scissor_and_labels():
    assert statistics_scissor() == (None, None, 'none')
    assert statistics_scissor(1.17) == (None, 1.17, 'target')
    scissor, target, mode = statistics_scissor(1.17, (4.73e-4, 636.0))
    assert scissor is None and mode == 'target' and callable(target)
    assert abs(target(0.0) - 1.17) < 1e-12 and target(300.0) < 1.17
    shift, target, mode = statistics_scissor(None, (4.73e-4, 636.0))
    assert target is None and mode == 'relative' and shift(300.0) < 0.0
    assert smearing_label(0.05, None) == 'sigma 0.050 eV'
    assert smearing_label(0.001, 0.7) == 'adaptive a = 0.70 (floor 1.0 meV)'
    assert absorption_label(12, 6, 0.05, None, True, 'full') == (
        'k 12^3, q 6^3, degauss 0.050 eV, position dipoles (full)'
    )
    assert mobility_label(48, 48, 0.01, 0.7, 0.3) == (
        'k 48^3, q 48^3, smearing adaptive a = 0.70 (floor 10.0 meV), window 0.30 eV'
    )


def test_write_eliashberg_writes_the_files_and_pops_the_fs_coupling(tmp_path, capsys):
    omega = np.linspace(0.001, 0.02, 20)
    out = {
        'lambda': 1.2, 'omega_log': 0.0045, 'Tc_mcmillan': 4.5, 'Tc_allen_dynes': 4.9,
        'omega': omega, 'a2F': np.ones_like(omega), 'a2F_tr': 0.8 * np.ones_like(omega),
        'lambda_tr': 1.0, 'omega_plasma_ev': 9.0,
        'fs_coupling': {'band': np.array([1, 2]), 'coupling': np.zeros((2, 2, 3)),
                        'freq_ev': np.array([0.001, 0.002, 0.003]), 'weight': np.ones(2),
                        'k_cryst': np.zeros((2, 3)), 'energy_ev': np.zeros(2), 'bg': np.eye(3),
                        'nk_dense': 2, 'nq_dense': 2, 'fsthick_ev': 0.2},
    }  # fmt: skip
    paths = write_eliashberg(out, str(tmp_path / 'out'), mu_star=0.1, label='test')
    names = sorted(os.path.basename(p) for p in paths)
    assert names == ['alpha2F.dat', 'alpha2F_tr.dat', 'eliashberg.npz', 'fs_coupling.npz']
    assert 'fs_coupling' not in out
    data = np.loadtxt(tmp_path / 'out' / 'alpha2F.dat')
    assert data.shape == (20, 2) and abs(data[0, 0] - 1.0) < 1e-9  # meV
    text = capsys.readouterr().out
    assert 'lambda   = 1.2000' in text and 'lambda_tr = 1.0000' in text


def test_arpes_path_helpers():
    nscf = {'bg': np.eye(3), 'alat': 10.0}
    k_cryst, k_cart = wa.path_points(nscf, (0.5, 0.0, 0.5), 5, (0.0, 0.5))
    assert k_cryst.shape == (5, 3) and np.allclose(k_cryst[-1], (0.25, 0.0, 0.25))
    assert np.allclose(k_cart, k_cryst * 2.0 * np.pi / (10.0 * 0.529177210903))
    coupling = {
        'band': np.array([3, 3, 4]),
        'k_cryst': np.array([[0.1, 0, 0], [0.4, 0, 0], [0.2, 0, 0]]),
    }
    rots = np.array([np.eye(3), -np.eye(3)])
    assert wa.nearest_state(coupling, rots, np.array([0.62, 0.0, 0.0]), 3) == 1  # via -k
    assert wa.nearest_state(coupling, rots, np.array([0.1, 0.0, 0.0]), 4) == 2
    assert wa.nearest_state(coupling, rots, np.array([0.1, 0.0, 0.0]), 5) is None


def test_linewidth_table_and_on_shell_widths():
    w = np.linspace(-1.0, 1.0, 201)
    energies = np.array([[-0.5, 0.2], [-0.4, 0.3]])  # (nk, nband)
    sigma = np.zeros((2, 2, 2, w.size), dtype=complex)  # (ntemp, nk, nband, nw)
    sigma[0, :, :, :] = -0.01j * (1 + np.abs(w))[None, None, :]
    sigma[1] = 2.0 * sigma[0]
    result = {'w_ev': w, 'energies_ev': energies, 'sigma_ev': sigma, 'bands': np.array([1, 2]),
              'temps_k': [10.0, 300.0]}  # fmt: skip
    rows = wa.linewidth_table(result, ('G', 'X'))
    assert [(r[0], r[1]) for r in rows] == [(1, 'G'), (1, 'X'), (2, 'G'), (2, 'X')]
    assert np.allclose(rows[0][3], [10.0 * 1.5, 20.0 * 1.5])  # meV at E = -0.5
    widths = wa.on_shell_widths(result, 1)
    assert widths.shape == (2, 2) and abs(widths[1, 1] - 0.02 * 1.3) < 1e-9


def test_anisotropic_self_energy_interpolates_the_nearest_state():
    class Bloch:
        def crystal_coordinates(self, kpoints):
            return np.asarray(kpoints)

    w_grid = np.linspace(-0.1, 0.1, 21)
    states = np.array([w_grid + 0j, 2.0 * w_grid - 0.01j])
    coupling = {'band': np.array([3, 3]), 'k_cryst': np.array([[0.1, 0, 0], [0.3, 0, 0]])}
    self_energy = wa.AnisotropicSelfEnergy(coupling, [np.eye(3)], Bloch(), states, w_grid)
    out = self_energy(
        np.array([0.05]), kpoints=np.array([[0.12, 0, 0], [0.28, 0, 0]]), bands=[3, 4]
    )
    assert out.shape == (2, 2, 1)
    assert abs(out[0, 0, 0] - 0.05) < 1e-12 and abs(out[1, 0, 0] - (0.1 - 0.01j)) < 1e-12
    assert out[0, 1, 0] == 0.0  # band 4 has no Fermi-surface state
