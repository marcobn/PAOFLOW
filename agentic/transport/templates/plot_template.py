#!/usr/bin/env python3
"""Publication figures (template: copy to <study>/plot_figures.py); reads CSV files only.

Experiment = markers (one shape per sample; filled: drift / direct, open: Hall or replotted),
this work = solid lines, published theory = gray dashed; mu(T) on log-log axes with the
exponent annotated, rho(T) linear with rho0 stated; direct labels + a style legend; vector
PDF + 600 dpi PNG.  Look at every PNG.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from PAOFLOW.agentic.thermo_optics.datasets import read_dataset  # noqa: E402
from PAOFLOW.agentic.thermo_optics.plotting import (  # noqa: E402
    INK,
    REFERENCE_THEORY,
    TEMPERATURE_RAMP,
    label_line,
    panel_label,
    provenance_note,
    publication_style,
    save_figure,
)

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / 'data', HERE / 'figures'
EXP_MARKER = dict(ls='none', marker='o', ms=3.2, mew=0.4, mec='white')
EXP_OPEN = dict(ls='none', marker='o', ms=3.2, mew=0.8, mfc='white')
MODEL_LINE = dict(lw=1.3, solid_capstyle='round')
THEORY_LINE = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))
CARRIER_COLOR = {
    'e': TEMPERATURE_RAMP[3],
    'h': TEMPERATURE_RAMP[1],
}  # two series: two steps of one ramp


def figure_mobility(exp, model):
    with publication_style(3.4, 2.6):
        fig, axis = plt.subplots()
        for name, group in exp[exp.quantity.str.startswith('mobility_')].groupby(
            'dataset', sort=False
        ):
            carrier = group.quantity.iloc[0][-1]
            style = EXP_OPEN if 'hall' in group.quantity.iloc[0] else EXP_MARKER
            axis.plot(group.x, group.y, color=CARRIER_COLOR[carrier], **style)
            m = model[model.dataset == 'model_%s' % name].sort_values('x')
            axis.plot(m.x, m.y, color=CARRIER_COLOR[carrier], **MODEL_LINE)
            label_line(axis, float(m.x.iloc[-1]), float(m.y.iloc[-1]), carrier, fontsize=6)
        axis.set(xscale='log', yscale='log', xlabel='T (K)', ylabel='$\\mu$ (cm$^2$/V s)')
        handles = [Line2D([], [], color=INK['secondary'], **EXP_MARKER, label='<Experiment, year> (drift)'),
                   Line2D([], [], color=INK['secondary'], **EXP_OPEN, label='<Experiment, year> (Hall)'),
                   Line2D([], [], color=INK['secondary'], **MODEL_LINE, label='This work (MRTA)'),
                   Line2D([], [], **THEORY_LINE, label='<Reference calculation>')]  # fmt: skip
        axis.legend(handles=handles, loc='best')
        panel_label(axis, '(a)')
        provenance_note(
            fig, '<Which data are digitised / Hall-corrected / interpolated, and from where.>'
        )
        return save_figure(fig, FIGURES / 'fig1_mobility')


def figure_resistivity(exp, model):
    with publication_style(3.4, 2.6):
        fig, axis = plt.subplots()
        for name, group in exp[exp.quantity == 'resistivity'].groupby('dataset', sort=False):
            axis.plot(group.x, group.y, color=TEMPERATURE_RAMP[3], **EXP_MARKER)
            m = model[model.dataset == 'model_%s' % name].sort_values('x')
            axis.plot(m.x, m.y, color=TEMPERATURE_RAMP[3], **MODEL_LINE)
        axis.set(xlabel='T (K)', ylabel='$\\rho$ ($\\mu\\Omega$ cm)')
        handles = [Line2D([], [], color=INK['secondary'], **EXP_MARKER, label='<Experiment, year>'),
                   Line2D([], [], color=INK['secondary'], **MODEL_LINE, label='This work + $\\rho_0$')]  # fmt: skip
        axis.legend(handles=handles, loc='best')
        panel_label(axis, '(b)')
        provenance_note(fig, '<rho0 = ... added to the phonon-limited model; source of rho0.>')
        return save_figure(fig, FIGURES / 'fig2_resistivity')


if __name__ == '__main__':
    experiments = pd.concat(
        [read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')]
    )
    models = read_dataset(DATA / 'model_curves.csv')
    if (experiments.quantity.str.startswith('mobility_')).any():
        print('wrote', figure_mobility(experiments, models))
    if (experiments.quantity == 'resistivity').any():
        print('wrote', figure_resistivity(experiments, models))
