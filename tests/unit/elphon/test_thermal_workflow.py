"""Unit tests of the thermal (ZG supercell) workflow layer, ``PAOFLOW.elphon.workflows.thermal``."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from PAOFLOW.elphon.workflows import thermal as th

KB_EV = 8.617333262e-5


def _result(grid, kdist, E_pc, centres, sigma, weight=1.0, temperature_k=100.0, label='x'):
    """Spectral function of one Gaussian peak per k at ``centres``."""
    spectral = th.spectral_function_from_weights(
        centres[:, None], np.full((kdist.size, 1), weight), grid, sigma
    )
    return {
        'kdist': kdist,
        'sym_ticks': [(0.0, 'G'), (float(kdist[-1]), 'X')],
        'energy_ev': grid,
        'spectral': spectral,
        'E_pc': E_pc,
        'sum_rule': np.full(kdist.size, 2.0),
        'temperature_k': temperature_k,
        'label': label,
        'path': 'G-X',
        'broadening_ev': sigma,
        'configurations': 1,
    }


def test_band_statistics_recover_shift_and_spread():
    grid = np.linspace(-0.5, 0.5, 4001)
    kdist = np.linspace(0.0, 1.0, 9)
    E_pc = np.stack([np.linspace(-0.2, 0.2, 9), np.linspace(0.3, 0.6, 9)], axis=1)
    shift, width, sigma = 0.03, 0.02, 0.004
    rng = np.random.default_rng(1)
    centres = E_pc[:, 0] + shift
    res = _result(grid, kdist, E_pc, centres, np.sqrt(width**2 + sigma**2))
    res['broadening_ev'] = sigma
    stats = th.band_statistics(res, band=0, half_window_ev=0.2)
    assert stats['k_index'].size == 9
    assert stats['shift_mean_ev'] == pytest.approx(shift, abs=1.0e-4)
    assert stats['spread_mean_ev'] == pytest.approx(np.sqrt(width**2 + sigma**2), rel=1.0e-2)
    assert stats['spread_intrinsic_ev'] == pytest.approx(width, rel=2.0e-2)
    assert stats['weight_mean'] == pytest.approx(1.0, rel=1.0e-3)
    # the window is clipped at the grid (every k kept); a window wider than the grid skips all
    assert th.band_statistics(res, band=0, half_window_ev=0.35)['k_index'].size == 9
    assert th.band_statistics(res, band=0, half_window_ev=1.2)['k_index'].size == 0
    assert th.band_statistics(res, band=None)['shift_mean_ev'] is None
    del rng


def test_ensemble_average_and_npz_round_trip(tmp_path):
    grid = np.linspace(-0.3, 0.3, 601)
    kdist = np.linspace(0.0, 1.0, 5)
    E_pc = np.linspace(-0.1, 0.1, 5)[:, None]
    a = _result(grid, kdist, E_pc, E_pc[:, 0] + 0.02, 0.005, label='s0')
    b = _result(grid, kdist, E_pc, E_pc[:, 0] - 0.02, 0.005, label='s1')
    mean = th.ensemble_spectral_function([a, b], label='T100K')
    np.testing.assert_allclose(mean['spectral'], 0.5 * (a['spectral'] + b['spectral']))
    assert mean['configurations'] == 2 and mean['label'] == 'T100K'
    stats = th.band_statistics(mean, band=0, half_window_ev=0.1)
    assert stats['shift_mean_ev'] == pytest.approx(0.0, abs=1.0e-4)
    assert stats['spread_mean_ev'] == pytest.approx(np.sqrt(0.02**2 + 0.005**2), rel=1.0e-2)
    path = th.save_thermal_spectral(mean, str(tmp_path / 'thermal_spectral_T100K.npz'))
    back = th.load_thermal_spectral(path)
    np.testing.assert_allclose(back['spectral'], mean['spectral'])
    assert back['temperature_k'] == 100.0 and back['configurations'] == 2
    assert back['sym_ticks'] == mean['sym_ticks'] and back['label'] == 'T100K'
    # the reference cell has no temperature
    a['temperature_k'] = None
    back = th.load_thermal_spectral(th.save_thermal_spectral(a, str(tmp_path / 'ref.npz')))
    assert back['temperature_k'] is None
    with pytest.raises(ValueError):
        th.ensemble_spectral_function([a, dict(a, kdist=kdist + 1.0)])


def test_static_linewidth_einstein_spectrum(tmp_path):
    omega0, lam = 0.010, 1.5  # eV, lambda: a2F = lam omega0 / 2 delta(w - w0)
    omega = np.linspace(1.0e-4, 0.03, 3001)
    a2f = (
        lam
        * omega0
        / 2.0
        * np.exp(-0.5 * ((omega - omega0) / 2.0e-4) ** 2)
        / (np.sqrt(2.0 * np.pi) * 2.0e-4)
    )
    path = str(tmp_path / 'eliashberg.npz')
    np.savez(path, omega=omega, a2F=a2f, **{'lambda': lam})
    gamma = th.static_linewidth(path, [0.0, 300.0])
    assert gamma[0] == pytest.approx(np.pi * lam * omega0 / 2.0, rel=1.0e-3)
    n = 1.0 / np.expm1(omega0 / (KB_EV * 300.0))
    assert gamma[1] == pytest.approx(np.pi * lam * omega0 / 2.0 * (2.0 * n + 1.0), rel=1.0e-3)
    # a meV table is recognised
    np.savez(path, omega=omega * 1.0e3, a2F=a2f, **{'lambda': lam})
    assert th.static_linewidth(path, [0.0])[0] == pytest.approx(gamma[0], rel=1.0e-3)


def test_zg_cells_and_run_script(tmp_path):
    from PAOFLOW.phonon.special_displacement import MANIFEST

    zg = tmp_path / 'zg'
    for seed, temps in ((0, [None, 0.0, 300.0]), (1, [0.0, 300.0])):
        d = zg / ('seed%d' % seed)
        d.mkdir(parents=True)
        entries = []
        for temp in temps:
            prefix = 'pb_zg_ref' if temp is None else 'pb_zg_T%gK' % temp
            (d / (prefix + '.in')).write_text('&CONTROL\n/\n')
            entries.append(
                {'temperature_k': temp, 'prefix': prefix, 'input': prefix + '.in', 'error': 0.0}
            )
        (d / MANIFEST).write_text(
            json.dumps(
                {
                    'dims': [3, 3, 3],
                    'kgrid': [4, 4, 4],
                    'nbnd': 324,
                    'natoms': 27,
                    'configurations': entries,
                }
            )
        )
    # one finished run (save with the schema file), one directory without it
    save = zg / 'seed1' / 'tmp_pb_zg_T300K' / 'pb_zg_T300K.save'
    save.mkdir(parents=True)
    (save / 'data-file-schema.xml').write_text('<qes/>')
    (zg / 'seed0' / 'tmp_pb_zg_T0K' / 'pb_zg_T0K.save').mkdir(parents=True)
    # a save whose run has not finished (schema file present, .out without JOB DONE) is not used
    running = zg / 'seed0' / 'tmp_pb_zg_T300K' / 'pb_zg_T300K.save'
    running.mkdir(parents=True)
    (running / 'data-file-schema.xml').write_text('<qes/>')
    (zg / 'seed0' / 'pb_zg_T300K.out').write_text('     iteration #  3\n')
    cells = th.zg_cells(str(zg))
    assert [(c['temperature_k'], c['seed']) for c in cells] == [
        (None, 0),
        (0.0, 0),
        (0.0, 1),
        (300.0, 0),
        (300.0, 1),
    ]
    assert (
        cells[0]['prefix'] == 'pb_zg_ref'
        and cells[0]['natoms'] == 27
        and cells[0]['dims'] == [3, 3, 3]
    )
    assert [c['save'] is not None for c in cells] == [False, False, False, False, True]
    assert cells[-1]['save'] == str(save)
    script = th.write_zg_run_script(str(zg))
    text = open(script).read()
    assert (
        os.access(script, os.X_OK)
        and 'seed0/pb_zg_ref.in' in text
        and 'seed1/pb_zg_T300K.in' in text
    )
    assert 'JOB DONE' in text and 'mpirun -np $NP $PW -nk $NK' in text
    assert th.zg_cells(str(tmp_path / 'nothing')) == []


def test_symmetry_points_explicit_and_from_the_save(tmp_path):
    class System:
        save_path = str(tmp_path / 'pb.save')

    explicit = th.symmetry_points(System, {'G': (0, 0, 0), 'X': np.array([0.5, 0.0, 0.5])})
    assert explicit == {'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.5]}
    os.makedirs(System.save_path)
    alat = 9.27
    xml = (
        '<?xml version="1.0"?><qes:espresso xmlns:qes="http://www.quantum-espresso.org/ns/qes/qes-1.0">'
        '<output><atomic_structure nat="1" alat="%g" bravais_index="2"><cell>'
        '<a1>%g 0.0 %g</a1><a2>0.0 %g %g</a2><a3>%g %g 0.0</a3></cell></atomic_structure></output>'
        '</qes:espresso>' % (alat, -alat / 2, alat / 2, alat / 2, alat / 2, -alat / 2, alat / 2)
    )
    (tmp_path / 'pb.save' / 'data-file-schema.xml').write_text(xml)
    points = th.symmetry_points(System)
    assert 'G' in points and 'gG' not in points and 'X' in points and 'L' in points
    np.testing.assert_allclose(points['G'], [0.0, 0.0, 0.0])
    # members of the X and L stars (Cartesian, 2 pi / alat): |k| components (1, 0, 0) and (1/2, 1/2, 1/2)
    at = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
    bg = np.linalg.inv(at).T
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(points['X']) @ bg)), [0.0, 0.0, 1.0], atol=1e-12
    )
    np.testing.assert_allclose(np.abs(np.asarray(points['L']) @ bg), [0.5, 0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(
        np.sort(np.abs(np.asarray(th.FCC_POINTS['X']) @ bg)), [0.0, 0.0, 1.0]
    )
    xml0 = xml.replace('bravais_index="2"', 'bravais_index="0"')
    (tmp_path / 'pb.save' / 'data-file-schema.xml').write_text(xml0)
    with pytest.raises(ValueError):
        th.symmetry_points(System)


def test_thermal_summary_round_trip_and_plot(tmp_path):
    rows = [
        dict(
            temperature_k=None,
            label='reference',
            configurations=1,
            band=6,
            kdist=0.3,
            peak_ev=0.002,
            fwhm_ev=0.007,
            weight=0.98,
            shift_ev=0.001,
            shift_rms_ev=0.001,
            spread_ev=0.003,
            sum_rule=18.0,
            gamma_static_ev=None,
        ),
        dict(
            temperature_k=0.0,
            label='T0K',
            configurations=5,
            band=6,
            kdist=0.3,
            peak_ev=0.02,
            peak_shift_ev=0.021,
            peak_vs_reference_ev=0.018,
            shift_vs_reference_ev=0.023,
            fwhm_ev=0.016,
            weight=0.9,
            shift_ev=0.024,
            shift_rms_ev=0.025,
            spread_ev=0.007,
            sum_rule=18.0,
            gamma_static_ev=0.0079,
        ),
        dict(
            temperature_k=300.0,
            label='T300K',
            configurations=5,
            band=6,
            kdist=0.3,
            peak_ev=0.1,
            fwhm_ev=0.12,
            weight=0.8,
            shift_ev=0.14,
            shift_rms_ev=0.15,
            spread_ev=0.05,
            sum_rule=18.0,
            gamma_static_ev=0.064,
        ),
    ]
    path = th.write_thermal_summary(rows, str(tmp_path / 'out' / 'thermal_summary.json'))
    assert th.load_thermal_summary(path) == rows
    table = open(os.path.splitext(path)[0] + '.dat').read().splitlines()
    assert table[0].startswith('#') and table[1].split()[0] == 'nan' and table[2].split()[1] == '5'
    pytest.importorskip('matplotlib')
    figure = th.plot_thermal_summary(rows, str(tmp_path / 'out' / 'thermal_summary.png'), name='Pb')
    assert os.path.getsize(figure) > 0
