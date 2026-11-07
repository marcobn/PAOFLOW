"""Unit tests of PAOFLOW.agentic.sscha: run adapter, diagnostics, temperature scans, dispersions, figures.

Synthetic run directories (``state.json``, population and Hessian summaries) and the delta-TaN dynamical matrices of
``tests/unit/phonon/data/sscha_tan`` (harmonic: X unstable at -223 cm^-1; SSCHA 2800 K: stable, X 100.3 cm^-1). No
python-sscha import is needed.
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pytest

from PAOFLOW.agentic.sscha import (
    SSCHARun,
    dispersion,
    find_runs,
    grid_minimum,
    next_temperature,
    scan_table,
    stability_temperature,
)

pytestmark = pytest.mark.unit

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'phonon',
                    'data', 'sscha_tan')  # fmt: skip
HARMONIC = os.path.join(DATA, 'harmonic', 'tan.dyn')
SSCHA = os.path.join(DATA, 'sscha_T2800', 'tan.dyn')
GAMMA, X = [0.0, 0.0, 0.0], [0.0, 0.5, 0.5]  # fcc, crystal coordinates of the reciprocal lattice


def _write(path, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(data, handle)


def _make_run(root, temperature, history, hessian=None, fc_source=None, electronic=None, failed=0, total=10,
              kong_liu_ratio=0.5, converged=None):  # fmt: skip
    """A run directory as PAOFLOW.sscha_* leaves it."""
    folder = os.path.join(root, 'sscha_T%g' % temperature)
    converged = history[-1]['converged'] if converged is None else converged
    state = dict(temperature=temperature, supercell=[2, 2, 2], nqirr=3, generated=len(history),
                 minimized=len(history), converged=converged, hessian_population=None,
                 electronic_temperature=electronic, population_sizes=[10] * 4, hessian_size=10,
                 kong_liu_ratio=kong_liu_ratio, min_step_dyn=0.05, meaningful_factor=1e-3)  # fmt: skip
    for row in history:
        pop = os.path.join(folder, 'pop%d' % row['population'])
        _write(os.path.join(pop, 'summary.json'), row)
        for index in range(total):
            open(os.path.join(pop, 'cfg_%04d.in' % index), 'w').write('input')
            if index >= failed:
                open(os.path.join(pop, 'cfg_%04d.out' % index), 'w').write('Total force\nJOB DONE.')
    if hessian is not None:
        state.update(hessian_population=len(history) + 1, generated=len(history) + 1, hessian_done=True,
                     stable=hessian['stable'], exported=['auxiliary', 'hessian'])  # fmt: skip
        _write(os.path.join(folder, 'hessian', 'summary.json'), hessian)
    if fc_source is not None:
        target = os.path.join(folder, 'force_constants_hessian')
        os.makedirs(target)
        for index in range(4):
            shutil.copy(fc_source + str(index), os.path.join(target, 'tan.dyn%d' % index))
    _write(os.path.join(folder, 'state.json'), state)
    return folder


def _row(population, f, df, kl, converged, lowest):
    return dict(population=population, size=10, free_energy_ry=f, free_energy_error_ry=df, kong_liu=kl,
                converged=converged, lowest_cm1=[lowest] * 6)  # fmt: skip


def test_run_adapter_status_history_and_hessian(tmp_path):
    folder = _make_run(str(tmp_path), 2800.0,
                       [_row(1, -0.10, 1e-4, 0.5, False, 90.0), _row(2, -0.11, 1e-4, 0.9, True, 100.3)],
                       hessian={'stable': True, 'lowest_cm1': [[[0, 0, 0], 124.4], [[0, 0.5, 0.5], 100.3]]},
                       fc_source=SSCHA, electronic=2800.0)  # fmt: skip
    run = SSCHARun(folder)
    assert (
        run.temperature == 2800.0
        and run.electronic_temperature == 2800.0
        and run.supercell == (2, 2, 2)
    )
    assert run.status()['stage'] == 'done' and len(run.history()) == 2
    assert run.lowest_hessian() == ([0, 0.5, 0.5], 100.3)
    assert run.force_constants('hessian').endswith('force_constants_hessian/tan.dyn')
    assert run.force_constants('auxiliary') is None
    assert run.diagnostics(fine_mesh=(4, 4, 4)) == []
    text = run.describe()
    assert 'Fermi-Dirac 2800 K' in text and 'Hessian: stable, softest 100.3' in text
    with pytest.raises(FileNotFoundError, match='state.json'):
        SSCHARun(str(tmp_path))


def test_diagnostics_find_the_problems(tmp_path):
    history = [_row(p, -0.1 - 0.01 * p, 1e-4, 0.5, False, 60.0 + 15 * p) for p in (1, 2, 3, 4)]
    folder = _make_run(str(tmp_path), 1500.0, history, failed=2,
                       hessian={'stable': True, 'lowest_cm1': [[[0, 0.5, 0.5], 30.0]]},
                       fc_source=HARMONIC, converged=False)  # fmt: skip
    rows = SSCHARun(folder).diagnostics(fine_mesh=(2, 2, 2))
    codes = [row['code'] for row in rows]
    assert codes.count('failed_configurations') == 4 and rows[0]['level'] == 'warning'
    assert {
        'kong_liu_every_population',
        'free_energy_drift',
        'frequencies_moving',
        'cold_electrons',
        'unstable_between_grid_points',
    } <= set(codes)
    unstable = _make_run(str(tmp_path / 'u'), 1000.0, [_row(1, -0.1, 1e-4, 0.9, True, 50.0)],
                         hessian={'stable': False, 'lowest_cm1': [[[0, 0.5, 0.5], -80.0]]})  # fmt: skip
    rows = SSCHARun(unstable).diagnostics()
    assert [row['code'] for row in rows] == [
        'unstable',
        'cold_electrons',
    ] and '-80.0 cm^-1' in rows[0]['message']
    failed = _make_run(
        str(tmp_path / 'f'), 900.0, [_row(p, -0.1, 1e-4, 0.5, False, 50.0) for p in (1, 2, 3, 4)]
    )
    assert SSCHARun(failed).status()['stage'] == 'failed'
    assert SSCHARun(failed).diagnostics()[0]['code'] == 'not_converged'


def test_find_runs_and_scan_table(tmp_path):
    for temperature, w in ((2000.0, -40.0), (1000.0, -120.0), (2800.0, 100.3)):
        _make_run(str(tmp_path), temperature, [_row(1, -0.1, 1e-4, 0.9, True, abs(w))],
                  hessian={'stable': w > 0, 'lowest_cm1': [[[0, 0.5, 0.5], w]]},
                  fc_source=SSCHA if w > 0 else HARMONIC)  # fmt: skip
    runs = find_runs(str(tmp_path))
    assert [run.temperature for run in runs] == [1000.0, 2000.0, 2800.0]
    rows = scan_table(runs, fine_mesh=(2, 2, 2))
    assert [row['stable'] for row in rows] == [False, False, True]
    assert rows[0]['lowest_cm1'] == -120.0 and rows[0]['fine_lowest_cm1'] == pytest.approx(
        -223.4, abs=0.2
    )
    assert rows[2]['fine_lowest_cm1'] == pytest.approx(100.3, abs=0.1)


def test_stability_temperature_of_a_soft_mode():
    temperatures = np.array([1000.0, 1400.0, 1800.0, 2200.0])
    w2 = 2.0 * (temperatures - 1500.0)
    lowest = np.sign(w2) * np.sqrt(np.abs(w2))
    result = stability_temperature(temperatures[::-1], lowest[::-1])
    assert result['t_stability'] == pytest.approx(1500.0) and result['t_fit'] == pytest.approx(
        1500.0
    )
    assert result['bracket'] == (1400.0, 1800.0) and result['direction'] == 'stabilised'
    assert result['slope'] == pytest.approx(2.0) and result['rms'] == pytest.approx(0.0, abs=1e-9)
    assert next_temperature(temperatures, lowest) == 1500.0
    assert (
        next_temperature(
            [1400.0, 1550.0], lowest[[1]].tolist() + [np.sqrt(100.0)], tolerance_k=200.0
        )
        is None
    )
    cold = stability_temperature(
        [600.0, 800.0, 1000.0], -np.sqrt(2.0 * (1500.0 - np.array([600.0, 800, 1000])))
    )
    assert cold['extrapolated'] and cold['t_stability'] == pytest.approx(1500.0)
    assert next_temperature([600.0, 800.0, 1000.0], [-42.4, -37.4, -31.6]) == pytest.approx(
        1500.0, abs=60.0
    )
    with pytest.raises(ValueError, match='two temperatures'):
        stability_temperature([1000.0], [-10.0])


def test_dispersion_and_grid_minimum():
    curve = dispersion(SSCHA, [GAMMA, X], ['G', 'X'], npoints=10)
    assert curve['frequencies_cm1'].shape == (11, 6) and curve['ticks'] == pytest.approx([0.0, 1.0])
    assert curve['frequencies_cm1'][-1, :3] == pytest.approx([100.3, 103.6, 103.6], abs=0.1)
    assert curve['frequencies_cm1'][0, :3] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)
    q, w = grid_minimum(HARMONIC, (2, 2, 2))
    assert w == pytest.approx(-223.4, abs=0.2) and np.count_nonzero(np.abs(q) == 0.5) == 2
    with pytest.raises(ValueError, match='one label per vertex'):
        dispersion(SSCHA, [GAMMA, X], ['G'])


def test_figures(tmp_path):
    pytest.importorskip('matplotlib')
    import matplotlib

    matplotlib.use('Agg')
    from PAOFLOW.agentic.sscha.plotting import (
        plot_convergence,
        plot_dispersions,
        plot_scan,
        publication_style,
        save_figure,
    )

    folder = _make_run(str(tmp_path), 2800.0, [_row(1, -0.10, 1e-4, 0.5, False, 90.0),
                                               _row(2, -0.11, 1e-4, 0.9, True, 100.3)])  # fmt: skip
    path = [GAMMA, X, [0.5, 0.5, 0.5]]
    with publication_style():
        figure, _ = plot_convergence(SSCHARun(folder))
        assert all(p.exists() for p in save_figure(figure, tmp_path / 'convergence'))
        rows = [
            dict(temperature=t, lowest_cm1=w)
            for t, w in ((1000.0, -60.0), (2000.0, -20.0), (2800.0, 40.0))
        ]
        result = stability_temperature([1000.0, 2000.0, 2800.0], [-60.0, -20.0, 40.0])
        figure, _ = plot_scan(rows, result, experiment={'temperature': 2094.0, 'low': 2000.0, 'high': 2200.0,
                                                         'label': 'eutectoid'})  # fmt: skip
        assert all(p.exists() for p in save_figure(figure, tmp_path / 'scan'))
        figure, _ = plot_dispersions({2800.0: dispersion(SSCHA, path, ['G', 'X', 'L'], 8)},
                                     reference=dispersion(HARMONIC, path, ['G', 'X', 'L'], 8))  # fmt: skip
        assert all(p.exists() for p in save_figure(figure, tmp_path / 'dispersions'))
