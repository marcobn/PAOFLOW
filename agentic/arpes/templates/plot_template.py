#!/usr/bin/env python3
"""Publication figures (template: copy to <study>/plot_figures.py); reads data/ and tables/ only.

Panel (a): the model band map (grey, intensities normalised per MDC) with the measured
dispersion as markers and the model dispersion as a line versus k - k_F, the bare band dashed.
Panel (b): Re Sigma (and Im Sigma on its own axis if needed): experiment markers, model line,
tunnelling baseline gray dashed.  Vector PDF + 600 dpi PNG.  Look at every PNG.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from PAOFLOW.agentic.arpes import ArpesModel  # noqa: E402
from PAOFLOW.agentic.thermo_optics.datasets import read_dataset  # noqa: E402
from PAOFLOW.agentic.thermo_optics.plotting import (  # noqa: E402
    INK,
    REFERENCE_THEORY,
    TEMPERATURE_RAMP,
    panel_label,
    provenance_note,
    publication_style,
    save_figure,
)

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / 'data', HERE / 'figures'
SYSTEM_DIR, OUTPUT, KIND, TEMP = (
    Path('<system folder>'),
    'output',
    'isotropic',
    10.0,
)  # as in analysis.py (map only)
EXP_MARKER = dict(ls='none', marker='o', ms=3.0, mew=0.4, mec='white', color=TEMPERATURE_RAMP[4])
MODEL_LINE = dict(lw=1.3, color=TEMPERATURE_RAMP[2])
BARE_LINE = dict(lw=0.9, ls=(0, (4, 2)), color=INK['secondary'])
THEORY_LINE = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))


def figure_band_map(exp, model_curves):
    model = ArpesModel.from_system(SYSTEM_DIR, output=OUTPUT)
    amap = model.map(KIND, TEMP)
    k_f = model.fermi_momentum(KIND, TEMP)
    intensity = amap['intensity'] / np.maximum(amap['intensity'].max(axis=0, keepdims=True), 1e-30)
    with publication_style(3.4, 2.8):
        fig, axis = plt.subplots()
        axis.pcolormesh(
            amap['distance'] - k_f,
            1e3 * amap['w'],
            intensity.T,
            cmap='Greys',
            shading='auto',
            rasterized=True,
        )
        bare = model.bare_dispersion()
        axis.plot(bare['k'] - model.bare_fermi_momentum(), 1e3 * bare['energy_ev'], **BARE_LINE)
        m = model_curves[model_curves.dataset == 'model_dispersion_%gK' % TEMP].sort_values('y')
        axis.plot(m.x, m.y, **MODEL_LINE)
        for _, group in exp[exp.quantity == 'dispersion'].groupby('dataset', sort=False):
            scale = 1.0 if group.y_unit.iloc[0] == 'meV' else 1e3
            axis.plot(
                group.x - float(group.conditions.iloc[0].count('k_reference=kF') and 0.0),
                scale * group.y,
                **EXP_MARKER,
            )
        axis.set(
            xlabel='$k - k_F$ (1/Å)',
            ylabel='$E - E_F$ (meV)',
            ylim=(1e3 * amap['w'][0], 1e3 * amap['w'][-1]),
        )
        handles = [Line2D([], [], **EXP_MARKER, label='<Experiment, year>'),
                   Line2D([], [], **MODEL_LINE, label='This work (dressed)'),
                   Line2D([], [], **BARE_LINE, label='bare band')]  # fmt: skip
        axis.legend(handles=handles, loc='lower right')
        panel_label(axis, '(a)')
        provenance_note(
            fig, '<Measured dispersion: digitised / MDC fits; aligned at k_F; bare-band protocol.>'
        )
        return save_figure(fig, FIGURES / 'fig1_band_map')


def figure_self_energy(exp, model_curves):
    model = ArpesModel.from_system(SYSTEM_DIR, output=OUTPUT)
    sigma = model.self_energy(KIND, TEMP)
    with publication_style(3.4, 2.6):
        fig, axis = plt.subplots()
        axis.plot(1e3 * sigma['w_ev'], 1e3 * sigma['sigma'].real, **MODEL_LINE)
        for _, group in exp[exp.quantity == 're_self_energy'].groupby('dataset', sort=False):
            axis.plot(group.x, group.y, **EXP_MARKER)
        axis.set(xlabel='$\\omega$ (meV)', ylabel='Re $\\Sigma$ (meV)')
        handles = [Line2D([], [], **EXP_MARKER, label='<Experiment, year> (bare band: <protocol>)'),
                   Line2D([], [], **MODEL_LINE, label='This work, %s' % KIND),
                   Line2D([], [], **THEORY_LINE, label='<tunnelling alpha^2F>')]  # fmt: skip
        axis.legend(handles=handles, loc='best')
        panel_label(axis, '(b)')
        provenance_note(
            fig, '<Extracted Sigma: authors protocol; model Sigma from alpha^2F at %g K.>' % TEMP
        )
        return save_figure(fig, FIGURES / 'fig2_self_energy')


if __name__ == '__main__':
    experiments = pd.concat(
        [read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')]
    )
    models = read_dataset(DATA / 'model_curves.csv')
    print('wrote', figure_band_map(experiments, models))
    if (experiments.quantity == 're_self_energy').any():
        print('wrote', figure_self_energy(experiments, models))
