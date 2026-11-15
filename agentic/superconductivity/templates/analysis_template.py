#!/usr/bin/env python3
"""Model vs experiment (template: copy to <study>/analysis.py).

    python analysis.py      # -> data/model_curves.csv, tables/tc.csv, tables/metrics.csv, tables/summary.md

Reads data/*.csv (canonical datasets: tc, gap, lambda, a2f, ...) and the superconductivity
output of the system folder (read-only, through the adapters); writes the Tc table with its
mu* sensitivity, every model curve compared (with the BCS baseline), and one metrics row per
comparison x dataset x condition x window.  Playbook stages 4-5:
PAOFLOW/agentic/superconductivity/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.superconductivity import AnisotropicGapModel, EliashbergModel, GapModel
from PAOFLOW.agentic.superconductivity import baselines as bl
from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, read_dataset, write_dataset

HERE = Path(__file__).resolve().parent
SYSTEM_DIR = Path('<electron-phonon system folder>')  # read-only (system.py, superconductivity/)
OUTPUT, ME, ME_ANISO = (
    'output',
    'output/me',
    'output/me_aniso',
)  # the runs compared ('' to skip the ME ones)
DATA, TABLES = HERE / 'data', HERE / 'tables'
SOURCE_MODEL = 'PAOFLOW e-ph superconductivity (this work), %s/superconductivity' % SYSTEM_DIR.name
MU_STARS = (0.10, 0.13)  # the mu* sensitivity quoted next to the run's value
GAP_KIND = (
    'imag'  # 'imag' (Delta(iw_0)), 'pade' or 'acon' (gap edge): match the measured definition
)
SHEETS = {}  # e.g. {'sigma': [6, 7], 'pi': [8]} band groups for a multi-gap metal
PHONON_WINDOWS = {
    'acoustic': (0.0, 0.006),
    'optical': (0.006, 0.02),
}  # eV, for alpha^2F comparisons


def load_experiments():
    return pd.concat([read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')],
                     ignore_index=True)  # fmt: skip


def tc_table(eli, gap, aniso, exp):
    rows = [{'side': 'model', 'definition': 'Allen-Dynes, mu* %g (run)' % eli.scalars['mu_star'], 'tc_k': eli.tc()},
            {'side': 'model', 'definition': 'McMillan, mu* %g (run)' % eli.scalars['mu_star'], 'tc_k': eli.tc(formula='mcmillan')}]  # fmt: skip
    rows += [
        {'side': 'model', 'definition': 'Allen-Dynes, mu* %g' % mu, 'tc_k': eli.tc(mu_star=mu)}
        for mu in MU_STARS
    ]
    if gap is not None:
        rows += [
            {'side': 'model', 'definition': 'isotropic ME, %s' % k, 'tc_k': v}
            for k, v in gap.tc.items()
        ]
    if aniso is not None:
        rows.append(
            {'side': 'model', 'definition': 'anisotropic ME, Delta_max^2 -> 0', 'tc_k': aniso.tc}
        )
    for name, group in exp[exp.quantity == 'tc'].groupby('dataset', sort=False):
        tc_exp = float(group.y.iloc[0])
        rows.append({'side': name, 'definition': group.conditions.iloc[0], 'tc_k': tc_exp,
                     'lambda_inverted': bl.lambda_from_tc(tc_exp, eli.scalars['omega_log'], eli.scalars['mu_star'],
                                                          eli.scalars['omega_2'])})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    eli = EliashbergModel.from_system(SYSTEM_DIR, output=OUTPUT)
    gap = GapModel.from_system(SYSTEM_DIR, output=ME) if ME else None
    aniso = AnisotropicGapModel.from_system(SYSTEM_DIR, output=ME_ANISO) if ME_ANISO else None
    exp = load_experiments()
    TABLES.mkdir(exist_ok=True)
    curves, rows = [], []
    tc = tc_table(eli, gap, aniso, exp)
    tc.to_csv(TABLES / 'tc.csv', index=False)
    # --- Delta(T) against the measured gaps, BCS as the baseline on both sides
    if gap is not None:
        temps = gap.temperatures
        curves.append(make_dataset('model_gap_%s' % GAP_KIND, 'gap', temps, gap.gaps(GAP_KIND), 'K', 'meV', 'calculated',
                                   SOURCE_MODEL, 'migdal_eliashberg.npz', np.nan, note='mu* %g, %s' % (gap.mu_star, GAP_KIND)))  # fmt: skip
        curves.append(make_dataset('model_gap_bcs', 'gap', temps, gap.bcs_gap(kind=GAP_KIND), 'K', 'meV', 'calculated',
                                   'BCS scaled to the model Delta(0) and Tc', 'baselines.bcs_gap_vs_t', np.nan))  # fmt: skip
        for name, group in exp[exp.quantity == 'gap'].groupby('dataset', sort=False):
            t_exp, d_exp = group.x.to_numpy(dtype=float), group.y.to_numpy(dtype=float)
            inside = (t_exp >= temps.min()) & (t_exp <= temps.max())
            stats = mt.curve_deviation(
                t_exp[inside], d_exp[inside], temps, gap.gaps(GAP_KIND), 'linear', None
            )
            tc_exp = (
                float(exp[exp.quantity == 'tc'].y.iloc[0])
                if (exp.quantity == 'tc').any()
                else gap.tc['gap']
            )
            bcs_exp = bl.bcs_gap_vs_t(t_exp, tc_exp, d_exp[np.argmin(t_exp)])
            base = mt.curve_deviation(t_exp, d_exp, t_exp, bcs_exp, 'linear', None)
            rows.append({'experiment': name, 'quantity': 'gap (meV)', 'window': 'T <= Tc', 'definition': GAP_KIND,
                         **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'baseline': 'BCS scaled to the measured Delta(0), Tc', 'baseline_rms': base['rms'],
                         'gap_ratio_model': gap.gap_ratio(GAP_KIND), 'gap_ratio_bcs': bl.BCS_GAP_RATIO,
                         'gap_ratio_strong_coupling': bl.strong_coupling_gap_ratio(gap.tc['gap'], eli.scalars['omega_log'])})  # fmt: skip
    # --- alpha^2F against a tunnelling-inverted spectrum, per phonon window
    for name, group in exp[exp.quantity == 'a2f'].groupby('dataset', sort=False):
        w_exp = group.x.to_numpy(dtype=float) * (1e-3 if group.x_unit.iloc[0] == 'meV' else 1.0)
        w_model, f_model = eli.function()
        curves.append(make_dataset('model_a2f', 'a2f', 1e3 * w_model, f_model, 'meV', '1', 'calculated', SOURCE_MODEL,
                                   'eliashberg.npz', np.nan, note='lambda %.3f' % eli.scalars['lambda']))  # fmt: skip
        for window_name, window in PHONON_WINDOWS.items():
            stats = mt.curve_deviation(w_exp, group.y, w_model, f_model, 'linear', window)
            lam_exp = bl.cumulative_lambda(w_exp, group.y.to_numpy(dtype=float))
            rows.append({'experiment': name, 'quantity': 'alpha^2F', 'window': window_name,
                         **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'lambda_window_model': eli.lambda_in_window(*window),
                         'lambda_window_exp': float(np.interp(window[1], w_exp, lam_exp) - np.interp(window[0], w_exp, lam_exp)),
                         'baseline': 'inversion lambda %s' % group.conditions.iloc[0], 'baseline_rms': np.nan})  # fmt: skip
    # --- multi-gap: distribution peaks and sheet gaps against measured sheet gaps
    if aniso is not None and SHEETS:
        for t in aniso.temperatures:
            sheets = aniso.sheet_gaps(t, SHEETS, GAP_KIND if GAP_KIND != 'imag' else 'gap0')
            for sheet, values in sheets.items():
                curves.append(make_dataset('model_gap_%s' % sheet, 'gap', [t], [values['gap_mean']], 'K', 'meV', 'calculated',
                                           SOURCE_MODEL, 'migdal_eliashberg_aniso.npz', np.nan,
                                           conditions={'sheet': sheet, 'bands': SHEETS[sheet]},
                                           note='weighted mean; range %.2f-%.2f; peaks %s' % (values['gap_min'], values['gap_max'], aniso.gap_peaks(t))))  # fmt: skip
    write_dataset(DATA / 'model_curves.csv', pd.concat(curves, ignore_index=True),
                  ['Model curves of %s at the experimental conditions (analysis.py).' % SOURCE_MODEL])  # fmt: skip
    metrics = pd.DataFrame(rows)
    metrics.to_csv(TABLES / 'metrics.csv', index=False)
    described = [m.describe() for m in (eli, gap, aniso) if m is not None]
    (TABLES / 'summary.md').write_text('# Metrics\n\n' + '\n\n'.join(described) + '\n\n## Tc\n\n' + tc.to_string()
                                       + '\n\n## Metrics\n\n' + metrics.to_string() + '\n')  # fmt: skip
    print(tc.to_string())
    print(metrics.to_string())


if __name__ == '__main__':
    main()
