#!/usr/bin/env python3
"""Model vs experiment (template: copy to <study>/analysis.py).

    python analysis.py      # -> data/model_curves.csv, tables/metrics.csv, tables/lambda.csv, tables/summary.md

Reads data/*.csv (canonical datasets: dispersion, re_self_energy, ...) and data/*_map.npz
(calibrated band-map images), and the ARPES output of the system folder (read-only, through
ArpesModel).  Aligns the dispersions at their own k_F, defines the bare band with ONE protocol
on both sides, extracts Sigma and lambda the same way, and writes one metrics row per
comparison x dataset x window with a baseline.  Playbook stages 4-5:
PAOFLOW/agentic/arpes/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from PAOFLOW.agentic.arpes import ArpesModel
from PAOFLOW.agentic.arpes import curves as cv
from PAOFLOW.agentic.thermo_optics import metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, read_dataset, write_dataset
from PAOFLOW.elphon.self_energy import eliashberg_self_energy, mass_enhancement

HERE = Path(__file__).resolve().parent
SYSTEM_DIR = Path(
    '<electron-phonon system folder>'
)  # read-only (system.py, arpes/, superconductivity/)
OUTPUT = 'output'  # the arpes/<output> run compared
KIND, TEMP = 'isotropic', 10.0  # the dressed map compared
DATA, TABLES = HERE / 'data', HERE / 'tables'
SOURCE_MODEL = 'PAOFLOW ARPES (this work), %s/arpes/%s' % (SYSTEM_DIR.name, OUTPUT)
BARE_WINDOW = (
    -0.10,
    -0.04,
)  # eV: the bare-band protocol (the authors' linear-fit window), used on BOTH sides
KINK_WINDOW = (-0.03, -0.005)  # eV: the dressed-velocity window near E_F
TUNNELLING_A2F = (
    None  # Path('<tunnelling alpha2F.dat: omega (meV), a2F>') for the baseline Sigma, or None
)


def load_experiments():
    return pd.concat([read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')],
                     ignore_index=True)  # fmt: skip


def analyse_dispersion(energy_ev, k):
    """Bare line, k_F, dressed velocity and lambda of a dispersion with ONE protocol."""
    bare = cv.linear_band(energy_ev, k, BARE_WINDOW)
    dressed = cv.linear_band(energy_ev, k, KINK_WINDOW)
    sigma = cv.self_energy_from_dispersion(
        energy_ev, k, np.zeros_like(k), lambda q: bare['velocity'] * (q - bare['k_fermi'])
    )
    return {'k_fermi': cv.fermi_momentum(energy_ev, k), 'v_bare': bare['velocity'], 'v_dressed': dressed['velocity'],
            'lambda_velocity': cv.mass_enhancement_from_velocities(bare['velocity'], dressed['velocity']),
            're_sigma': sigma['re_sigma_ev']}  # fmt: skip


def main():
    model = ArpesModel.from_system(SYSTEM_DIR, output=OUTPUT)
    exp = load_experiments()
    TABLES.mkdir(exist_ok=True)
    curves, rows, lam_rows = [], [], []
    # --- the model dispersion, extracted as an experimentalist would
    m = model.dispersion(KIND, TEMP, method='lorentzian', energy_window=(BARE_WINDOW[0], 0.0))
    m_an = analyse_dispersion(m['energy_ev'], m['k'])
    curves.append(make_dataset('model_dispersion_%gK' % TEMP, 'dispersion', m['k'] - m_an['k_fermi'],
                               1e3 * m['energy_ev'], '1/A', 'meV', 'calculated', SOURCE_MODEL, 'arpes_%s_%gK.npz' % (KIND, TEMP),
                               TEMP, note='MDC Lorentzian peaks vs k - k_F; protocol windows %s / %s' % (BARE_WINDOW, KINK_WINDOW)))  # fmt: skip
    lam_rows.append(
        {
            'side': 'model',
            'method': 'velocity ratio (%s)' % (BARE_WINDOW,),
            'lambda': m_an['lambda_velocity'],
        }
    )
    lam_slope, z = model.mass_enhancement(KIND, TEMP)
    lam_rows.append({'side': 'model', 'method': 'slope of Re Sigma at E_F', 'lambda': lam_slope})
    lam_rows.append(
        {
            'side': 'model',
            'method': 'kink energy (meV)',
            'lambda': 1e3 * model.kink_energy(KIND, TEMP),
        }
    )
    # --- each measured dispersion
    for name, group in exp[exp.quantity == 'dispersion'].groupby('dataset', sort=False):
        scale = 1e-3 if group.y_unit.iloc[0] == 'meV' else 1.0
        e_exp, k_exp = group.y.to_numpy(dtype=float) * scale, group.x.to_numpy(dtype=float)
        e_an = analyse_dispersion(e_exp, k_exp)
        lam_rows.append(
            {
                'side': name,
                'method': 'velocity ratio (%s)' % (BARE_WINDOW,),
                'lambda': e_an['lambda_velocity'],
            }
        )
        for window_name, window in (
            ('below kink', (BARE_WINDOW[0], KINK_WINDOW[0])),
            ('near E_F', KINK_WINDOW),
        ):
            stats = mt.curve_deviation(
                k_exp - e_an['k_fermi'],
                1e3 * e_exp,
                m['k'] - m_an['k_fermi'],
                1e3 * m['energy_ev'],
                'linear',
                None,
            )  # fmt: skip  (window on E: filter the points yourself)
            keep = (e_exp >= window[0]) & (e_exp <= window[1])
            dev = mt.curve_deviation(k_exp[keep] - e_an['k_fermi'], 1e3 * e_exp[keep], m['k'] - m_an['k_fermi'],
                                     1e3 * m['energy_ev'], 'linear', None) if keep.sum() > 1 else stats  # fmt: skip
            rows.append({'experiment': name, 'quantity': 'dispersion (meV vs k - k_F)', 'window': window_name,
                         **{k: dev.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                         'baseline': 'linear bare band', 'baseline_rms': e_an['v_bare'] * 0.0 + np.nan})  # fmt: skip
    # --- Re Sigma extracted by the authors vs the model's Sigma (and the tunnelling baseline)
    sigma_model = model.self_energy(KIND, TEMP)
    for name, group in exp[exp.quantity == 're_self_energy'].groupby('dataset', sort=False):
        w_exp = group.x.to_numpy(dtype=float) * (1e-3 if group.x_unit.iloc[0] == 'meV' else 1.0)
        stats = mt.curve_deviation(
            1e3 * w_exp,
            group.y,
            1e3 * sigma_model['w_ev'],
            1e3 * sigma_model['sigma'].real,
            'linear',
            None,
        )
        base = np.nan
        if TUNNELLING_A2F is not None:
            table = np.loadtxt(TUNNELLING_A2F)
            sig_t = eliashberg_self_energy(
                table[:, 0] * 1e-3, table[:, 1], sigma_model['w_ev'], TEMP
            )
            base = mt.curve_deviation(
                1e3 * w_exp, group.y, 1e3 * sigma_model['w_ev'], 1e3 * sig_t.real, 'linear', None
            )['rms']
            lam_rows.append(
                {
                    'side': 'tunnelling alpha^2F',
                    'method': 'slope of Re Sigma',
                    'lambda': mass_enhancement(sigma_model['w_ev'], sig_t)[0],
                }
            )
        rows.append({'experiment': name, 'quantity': 'Re Sigma (meV)', 'window': 'all',
                     **{k: stats.get(k, np.nan) for k in ('n', 'bias', 'rms', 'max_abs')},
                     'baseline': 'tunnelling alpha^2F Sigma' if TUNNELLING_A2F else '', 'baseline_rms': base})  # fmt: skip
    for name, group in exp[exp.quantity == 'lambda'].groupby('dataset', sort=False):
        lam_rows.append(
            {'side': name, 'method': group.conditions.iloc[0], 'lambda': float(group.y.iloc[0])}
        )
    write_dataset(DATA / 'model_curves.csv', pd.concat(curves, ignore_index=True),
                  ['Model curves of %s at the experimental conditions (analysis.py).' % SOURCE_MODEL])  # fmt: skip
    metrics, lam = pd.DataFrame(rows), pd.DataFrame(lam_rows)
    metrics.to_csv(TABLES / 'metrics.csv', index=False)
    lam.to_csv(TABLES / 'lambda.csv', index=False)
    (TABLES / 'summary.md').write_text(
        '# Metrics\n\n'
        + model.describe()
        + '\n\n'
        + metrics.to_string()
        + '\n\n## lambda\n\n'
        + lam.to_string()
        + '\n'
    )
    print(metrics.to_string())
    print(lam.to_string())


if __name__ == '__main__':
    main()
