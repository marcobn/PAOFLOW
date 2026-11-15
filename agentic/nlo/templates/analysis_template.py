#!/usr/bin/env python3
"""Model vs experiment (template: copy to <study>/analysis.py).

    python analysis.py      # -> tables/symmetry.csv, data/model_curves.csv, tables/metrics.csv, tables/summary.md

Reads data/*.csv (canonical datasets of SHG coefficients / spectra / photocurrents) and the
output folder of PAOFLOW.nonlinear_optics (read-only, through NonlinearModel); does the symmetry
audit first, then evaluates the model at every measured condition with the stated conventions
and writes one metrics row per comparison with a baseline and the symmetry error bar.
Playbook stages 1, 4, 5: PAOFLOW/agentic/nlo/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.nlo import NonlinearModel
from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, read_dataset, write_dataset

HERE = Path(__file__).resolve().parent
OUTPUT = Path('<examples/nlo_examples/<System>/output>')  # read-only
RUN = {
    'nk': '<40^3>',
    'eta_ev': 0.05,
    'scissor_ev': 0.86,
    'basis': '<...>',
    'nosym': False,
    'tol_ev': 0.01,
}
POINT_GROUP = '<-43m | mm2_x | 3m_x | ...>'  # axis convention of symmetry.GENERATORS
TWO_D, SLAB_NM, THICKNESS_NM = (
    False,
    None,
    None,
)  # sheet runs: slab height of the run, literature thickness
DATA, TABLES = HERE / 'data', HERE / 'tables'
SOURCE_MODEL = 'PAOFLOW nonlinear_optics (this work), %s' % OUTPUT
AUDIT_ENERGIES = (0.0, 1.0, 2.0, 3.0)


def load_experiments():
    return pd.concat([read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')],
                     ignore_index=True)  # fmt: skip


def symmetry_audit(model):
    rows = []
    for kind in ('shg', 'shift', 'injectc'):
        key = 'shgr' if kind == 'shg' else kind
        if not any(k == key for k, _ in model.data):
            continue
        for e in AUDIT_ENERGIES:
            check = model.symmetry_check(POINT_GROUP, kind, e)
            rows.append({'kind': kind, 'energy_ev': e, 'forbidden_present': ','.join(check['forbidden_present']),
                         'leakage': check['leakage'],
                         'symmetry_error': check['error']['relative'] if check['error'] else np.nan})  # fmt: skip
    rows.append({'kind': 'shg', 'energy_ev': model.energy_ev[0], 'forbidden_present': 'Kleinman',
                 'leakage': np.nan, 'symmetry_error': model.kleinman_check()['relative']})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    model = NonlinearModel(OUTPUT, two_d=TWO_D, slab_height_nm=SLAB_NM, system=RUN)
    exp = load_experiments()
    TABLES.mkdir(exist_ok=True)
    audit = symmetry_audit(model)
    audit.to_csv(TABLES / 'symmetry.csv', index=False)
    error_bar = float(np.nanmax(audit[['leakage', 'symmetry_error']].to_numpy()))
    curves, rows = [], []
    # --- static / single-energy SHG coefficients (d or chi)
    for name, group in exp[exp.quantity.isin(['shg_d', 'shg_chi'])].groupby('dataset', sort=False):
        comp = group.conditions.iloc[0].split('component=')[1].split(';')[0].strip()
        e_exp = float(group.x.iloc[0])
        chi = model.value('shga', comp, max(e_exp, model.energy_ev[0]))
        value = NonlinearModel.d_coefficient(chi) if group.quantity.iloc[0] == 'shg_d' else chi
        if TWO_D and THICKNESS_NM:
            value = value / (1.0e3 * THICKNESS_NM)  # pm^2/V -> pm/V per layer thickness
        curves.append(make_dataset('model_' + name, group.quantity.iloc[0], [e_exp], [value], 'eV', group.y_unit.iloc[0],
                                   'calculated', SOURCE_MODEL, 'shga_%s.dat' % comp, np.nan,
                                   note='value at %.3g eV (run minimum %.3g eV); %s' % (max(e_exp, model.energy_ev[0]), model.energy_ev[0], RUN)))  # fmt: skip
        rows.append({'experiment': name, 'quantity': group.quantity.iloc[0], 'component': comp, 'energy_ev': e_exp,
                     'experiment_value': float(group.y.iloc[0]), 'model_value': value,
                     'relative_deviation': value / float(group.y.iloc[0]) - 1.0,
                     'baseline': '<published calculation / second measurement>', 'baseline_value': np.nan,
                     'symmetry_error_bar': error_bar})  # fmt: skip
    # --- spectra (|chi(w)| or shift current): deviation per window
    for name, group in exp[exp.quantity.isin(['shift_current'])].groupby('dataset', sort=False):
        comp = group.conditions.iloc[0].split('component=')[1].split(';')[0].strip()
        y_model = model.spectrum('shift', comp)
        if TWO_D and THICKNESS_NM and 'nm' not in group.y_unit.iloc[0]:
            y_model = model.effective_3d('shift', comp, THICKNESS_NM)
        curves.append(make_dataset('model_' + name, 'shift_current', model.energy_ev, y_model, 'eV', group.y_unit.iloc[0],
                                   'calculated', SOURCE_MODEL, 'shift_%s.dat' % comp, np.nan, note=str(RUN)))  # fmt: skip
        stats = mt.curve_deviation(group.x, group.y, model.energy_ev, y_model, 'linear', None)
        rows.append({'experiment': name, 'quantity': 'shift_current', 'component': comp, 'energy_ev': np.nan,
                     'experiment_value': float(np.max(np.abs(group.y))), 'model_value': model.peak('shift', comp)['value'],
                     'relative_deviation': stats['bias'] / max(float(np.max(np.abs(group.y))), 1e-30),
                     'baseline': '<...>', 'baseline_value': np.nan, 'symmetry_error_bar': error_bar, **{k: stats[k] for k in ('n', 'rms', 'max_abs')}})  # fmt: skip
    write_dataset(DATA / 'model_curves.csv', pd.concat(curves, ignore_index=True),
                  ['Model values of %s at the experimental conditions (analysis.py).' % SOURCE_MODEL])  # fmt: skip
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / 'metrics.csv', index=False)
    (TABLES / 'summary.md').write_text('# Symmetry audit\n\n' + audit.to_string() + '\n\n# Model\n\n' + model.describe()
                                       + '\n\n# Metrics\n\n' + metrics.to_string() + '\n')  # fmt: skip
    print(audit.to_string())
    print(metrics.to_string())


if __name__ == '__main__':
    main()
