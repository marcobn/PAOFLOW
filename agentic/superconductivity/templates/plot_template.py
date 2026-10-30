#!/usr/bin/env python3
"""Publication figures (template: copy to <study>/plot_figures.py); reads CSV files only.

(a) alpha^2F: this work solid, tunnelling alpha^2F markers / gray dashed, lambda(omega) on its own
panel; (b) Delta(T): experiment markers, this work solid, BCS dashed.  Vector PDF + 600 dpi PNG.
Look at every PNG.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from PAOFLOW.agentic.superconductivity import baselines as bl  # noqa: E402
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
EXP_MARKER = dict(ls='none', marker='o', ms=3.2, mew=0.4, mec='white', color=TEMPERATURE_RAMP[4])
MODEL_LINE = dict(lw=1.3, color=TEMPERATURE_RAMP[2])
BASELINE_LINE = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))


def figure_a2f(exp, model):
    m = model[model.dataset == 'model_a2f'].sort_values('x')
    with publication_style(3.4, 3.6):
        fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
        top.plot(m.x, m.y, **MODEL_LINE)
        for _, group in exp[exp.quantity == 'a2f'].groupby('dataset', sort=False):
            top.plot(group.x, group.y, **BASELINE_LINE)
            bottom.plot(
                group.x,
                bl.cumulative_lambda(group.x.to_numpy() * 1e-3, group.y.to_numpy()),
                **BASELINE_LINE,
            )
        bottom.plot(m.x, bl.cumulative_lambda(m.x.to_numpy() * 1e-3, m.y.to_numpy()), **MODEL_LINE)
        top.set(ylabel='$\\alpha^2F(\\omega)$')
        bottom.set(xlabel='$\\omega$ (meV)', ylabel='$\\lambda(\\omega)$')
        handles = [Line2D([], [], **MODEL_LINE, label='This work'),
                   Line2D([], [], **BASELINE_LINE, label='<tunnelling inversion, year>')]  # fmt: skip
        top.legend(handles=handles, loc='best')
        panel_label(top, '(a)')
        panel_label(bottom, '(b)')
        provenance_note(fig, '<alpha^2F digitised from ...; inversion mu* = ...>')
        return save_figure(fig, FIGURES / 'fig1_a2f')


def figure_gap(exp, model):
    with publication_style(3.4, 2.6):
        fig, axis = plt.subplots()
        m = model[
            model.dataset.str.startswith('model_gap_') & ~model.dataset.eq('model_gap_bcs')
        ].sort_values('x')
        b = model[model.dataset == 'model_gap_bcs'].sort_values('x')
        axis.plot(m.x, m.y, **MODEL_LINE)
        axis.plot(b.x, b.y, **BASELINE_LINE)
        for _, group in exp[exp.quantity == 'gap'].groupby('dataset', sort=False):
            axis.plot(group.x, group.y, **EXP_MARKER)
        axis.set(xlabel='T (K)', ylabel='$\\Delta$ (meV)', ylim=(0, None))
        handles = [Line2D([], [], **EXP_MARKER, label='<Experiment, year> (<definition>)'),
                   Line2D([], [], **MODEL_LINE, label='This work ($\\mu^*$ = <...>)'),
                   Line2D([], [], **BASELINE_LINE, label='BCS scaled to the model')]  # fmt: skip
        axis.legend(handles=handles, loc='lower left')
        panel_label(axis, '(c)')
        provenance_note(fig, '<gap definition on both sides; temperatures; mu*>')
        return save_figure(fig, FIGURES / 'fig2_gap')


if __name__ == '__main__':
    experiments = pd.concat(
        [read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')]
    )
    models = read_dataset(DATA / 'model_curves.csv')
    if (models.dataset == 'model_a2f').any():
        print('wrote', figure_a2f(experiments, models))
    if (experiments.quantity == 'gap').any():
        print('wrote', figure_gap(experiments, models))
    _ = (INK, np)  # keep the imports available for extra panels
