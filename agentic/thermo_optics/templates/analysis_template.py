#!/usr/bin/env python3
"""Model vs experiment (template: copy to <study>/analysis.py).

    python analysis.py      # -> data/model_curves.csv, tables/metrics.csv, tables/*.csv, tables/summary.md

Reads data/*.csv (canonical datasets) and the model output (read-only, through an
adapter); writes every model curve it compares to data/model_curves.csv, and one
metrics row per comparison x dataset x condition x window, each with a baseline.
Playbook stages 4-5: PAOFLOW/agentic/thermo_optics/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, read_dataset, write_dataset
from PAOFLOW.agentic.thermo_optics.paoflow_optics import (
    OpticsModel,
)  # or: from model_adapter import ModelAdapter
from PAOFLOW.agentic.thermo_optics.units import convert_photon

HERE = Path(__file__).resolve().parent
MODEL_DIR = Path('<calculation output folder>')  # read-only
DATA, TABLES = HERE / 'data', HERE / 'tables'
SOURCE_MODEL = 'PAOFLOW <method> (this work), %s' % MODEL_DIR.name
TOLERANCE_K = 40.0  # largest |T_exp - T_model| used without interpolation
WINDOWS = {
    '<physics window 1>': (0.0, 1.0),
    '<physics window 2>': (1.0, 2.0),
}  # in the experimental x unit


def load_experiments():
    return pd.concat([read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')],
                     ignore_index=True)  # fmt: skip


def main():
    model = OpticsModel(MODEL_DIR)
    exp = load_experiments()
    TABLES.mkdir(exist_ok=True)
    curves, rows = [], []
    for name, group in exp[exp.quantity == '<quantity>'].groupby('dataset', sort=False):
        temp = float(group.temperature_k.iloc[0])
        ((_, t_model, dt),) = mt.match_conditions([temp], model.temperatures, TOLERANCE_K)
        interpolate = t_model is None
        t_eval = temp if interpolate else t_model
        y_model = model.absorption_coefficient(
            t_eval, interpolate=interpolate
        )  # the observable at the geometry
        x_exp = convert_photon(group.x.to_numpy(), group.x_unit.iloc[0], 'eV')
        label = 'interpolated %g-%g K' % model.bracket(temp) if interpolate else '%g K' % t_model
        curves.append(make_dataset('model_<quantity>_%gK' % temp, '<quantity>', model.omega_ev, y_model, 'eV', '<unit>',
                                   'calculated', SOURCE_MODEL, '<how evaluated>', temp, note=label))  # fmt: skip
        reference = exp[exp.dataset == '<published theory or second measurement dataset>']
        for window_name, window in WINDOWS.items():
            stats = mt.curve_deviation(x_exp, group.y, model.omega_ev, y_model, 'log10', window)
            base = (
                mt.curve_deviation(x_exp, group.y, reference.x, reference.y, 'log10', window)
                if len(reference)
                else None
            )
            rows.append({'experiment': name, 'condition': '%g K' % temp, 'model_condition': label, 'window': window_name,
                         **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'baseline': '<reference name>' if base else '', 'baseline_rms': base['rms'] if base else np.nan})  # fmt: skip
    write_dataset(DATA / 'model_curves.csv', pd.concat(curves, ignore_index=True),
                  ['Model curves of %s at the experimental conditions (analysis.py).' % MODEL_DIR])  # fmt: skip
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / 'metrics.csv', index=False)
    (TABLES / 'summary.md').write_text(
        '# Metrics\n\n' + model.describe() + '\n\n' + metrics.to_string() + '\n'
    )
    print(metrics.to_string())


if __name__ == '__main__':
    main()
