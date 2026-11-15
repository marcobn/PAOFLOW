#!/usr/bin/env python3
"""Model vs experiment (template: copy to <study>/analysis.py).

    python analysis.py      # -> data/model_curves.csv, tables/metrics.csv, tables/*.csv, tables/summary.md

Reads data/*.csv (canonical datasets) and the transport output (read-only, through the
adapters); writes every model curve it compares to data/model_curves.csv, and one metrics
row per comparison x dataset x condition x window, each with a baseline.  Two blocks are
sketched: a semiconductor mobility mu(T) and a metal resistivity rho(T); delete the one you
do not need.  Playbook stages 4-5: PAOFLOW/agentic/transport/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, read_dataset, write_dataset
from PAOFLOW.agentic.transport import MobilityModel, ResistivityModel
from PAOFLOW.agentic.transport import baselines as bl

HERE = Path(__file__).resolve().parent
SYSTEM_DIR = Path(
    '<electron-phonon system folder>'
)  # read-only (system.py, transport/, superconductivity/)
OUTPUT = 'output'  # the transport/<tag> run compared
DATA, TABLES = HERE / 'data', HERE / 'tables'
SOURCE_MODEL = 'PAOFLOW <method> (this work), %s/transport/%s' % (SYSTEM_DIR.name, OUTPUT)
HALL_FACTOR = {
    'e': 1.0,
    'h': 1.0,
}  # r_H of the measured Hall mobilities (state the source), 1.0 = drift data
RHO0_UOHM_CM = (
    0.0  # residual resistivity of the measured sample (state the source), 0 = already subtracted
)
WINDOWS = {'phonon-limited': (100.0, 400.0)}  # temperature windows (K) of the comparison


def load_experiments():
    return pd.concat([read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')],
                     ignore_index=True)  # fmt: skip


def mobility_block(exp, curves, rows):
    """mu(T) of each measured dataset against the MRTA drift mobility (power-law interpolation)."""
    model = MobilityModel.from_system(SYSTEM_DIR, output=OUTPUT)
    for name, group in exp[exp.quantity.str.startswith('mobility_')].groupby('dataset', sort=False):
        carrier = group.quantity.iloc[0][-1]  # 'e' or 'h'
        hall = HALL_FACTOR[carrier] if 'hall' in group.quantity.iloc[0] else 1.0
        temps = group.x.to_numpy(dtype=float)  # x = temperature (K)
        inside = (temps >= model.temperatures.min()) & (temps <= model.temperatures.max())
        mu_model = np.array(
            [model.mobility(t, carrier, 'mrta', interpolate=True) * hall for t in temps[inside]]
        )
        curves.append(make_dataset('model_%s' % name, group.quantity.iloc[0], temps[inside], mu_model, 'K', 'cm2/Vs',
                                   'calculated', SOURCE_MODEL, 'mobility.dat', np.nan,
                                   note='MRTA, power-law interpolation in T; Hall factor %g' % hall))  # fmt: skip
        for window_name, window in WINDOWS.items():
            stats = mt.curve_deviation(temps, group.y, temps[inside], mu_model, 'log10', window)
            p_exp = bl.fit_power_law(temps, group.y, window)['exponent']
            p_model = -model.power_law_exponent(carrier, 'mrta', window)
            rows.append({'experiment': name, 'quantity': group.quantity.iloc[0], 'window': window_name,
                         **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'exponent_exp': p_exp, 'exponent_model': p_model,
                         'baseline': '<published calculation>', 'baseline_rms': np.nan})  # fmt: skip


def resistivity_block(exp, curves, rows):
    """rho(T) of each measured dataset against the model with the sample's rho0 added (Matthiessen)."""
    model = ResistivityModel.from_system(SYSTEM_DIR, output=OUTPUT)
    for name, group in exp[exp.quantity == 'resistivity'].groupby('dataset', sort=False):
        temps = group.x.to_numpy(dtype=float)
        inside = (temps >= model.temperatures.min()) & (temps <= model.temperatures.max())
        rho_model = np.array([model.resistivity(t, rho0_uohm_cm=RHO0_UOHM_CM, interpolate=True)
                              for t in temps[inside]])  # fmt: skip
        curves.append(make_dataset('model_%s' % name, 'resistivity', temps[inside], rho_model, 'K', 'uOhm cm',
                                   'calculated', SOURCE_MODEL, 'resistivity.dat', np.nan,
                                   note='linear interpolation in T; rho0 = %g uOhm cm added' % RHO0_UOHM_CM))  # fmt: skip
        bg = bl.fit_bloch_gruneisen(
            temps, group.y.to_numpy(dtype=float)
        )  # the trivial model of the experiment
        offset = bl.matthiessen_offset(temps, group.y, model.temperatures, model.table['total'])
        for window_name, window in WINDOWS.items():
            stats = mt.curve_deviation(temps, group.y, temps[inside], rho_model, 'linear', window)
            rows.append({'experiment': name, 'quantity': 'resistivity', 'window': window_name,
                         **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'slope_exp': bl.fit_linear(temps, group.y, window)['slope'],
                         'slope_model': model.slope(window)['slope'],
                         'matthiessen_rho0': offset['rho0'], 'rms_after_offset': offset['rms_after'],
                         'baseline': 'Bloch-Gruneisen fit (rho0 %.3g, theta %.0f K)' % (bg['rho0'], bg['theta_k']),
                         'baseline_rms': bg['rms']})  # fmt: skip


def main():
    exp = load_experiments()
    TABLES.mkdir(exist_ok=True)
    curves, rows, described = [], [], []
    if (SYSTEM_DIR / 'transport' / OUTPUT / 'mobility.dat').exists():
        mobility_block(exp, curves, rows)
        described.append(MobilityModel.from_system(SYSTEM_DIR, output=OUTPUT).describe())
    if (SYSTEM_DIR / 'transport' / OUTPUT / 'resistivity.dat').exists():
        resistivity_block(exp, curves, rows)
        described.append(ResistivityModel.from_system(SYSTEM_DIR, output=OUTPUT).describe())
    write_dataset(DATA / 'model_curves.csv', pd.concat(curves, ignore_index=True),
                  ['Model curves of %s at the experimental conditions (analysis.py).' % SOURCE_MODEL])  # fmt: skip
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / 'metrics.csv', index=False)
    (TABLES / 'summary.md').write_text(
        '# Metrics\n\n' + '\n\n'.join(described) + '\n\n' + metrics.to_string() + '\n'
    )
    print(metrics.to_string())


if __name__ == '__main__':
    main()
