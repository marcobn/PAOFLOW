#!/usr/bin/env python3
"""Publication figures (template: copy to <study>/plot_figures.py); reads CSV files only.

Experiment = markers (filled: direct; open: indirect or replotted), this work = solid
lines, published theory = gray dashed; temperature on the ordinal ramp (<= 5 per panel);
direct labels + a style legend; vector PDF + 600 dpi PNG.  Look at every PNG.
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
    label_line,
    panel_label,
    provenance_note,
    publication_style,
    save_figure,
    temperature_colors,
)

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / 'data', HERE / 'figures'
EXP_MARKER = dict(ls='none', marker='o', ms=3.2, mew=0.4, mec='white')
EXP_OPEN = dict(ls='none', marker='o', ms=3.2, mew=0.8, mfc='white')
MODEL_LINE = dict(lw=1.3, solid_capstyle='round')
THEORY_LINE = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))


def figure_comparison(exp, model):
    temps = sorted(exp[exp.quantity == '<quantity>'].temperature_k.unique())[
        :5
    ]  # facet if more than 5
    colors = temperature_colors(temps)
    with publication_style(3.4, 2.6):  # 7.0 in for double column
        fig, axis = plt.subplots()
        for temp in temps:
            c = colors[float(temp)]
            e = exp[(exp.quantity == '<quantity>') & (exp.temperature_k == temp)].sort_values('x')
            m = model[model.dataset == 'model_<quantity>_%gK' % temp].sort_values('x')
            axis.plot(m.x, m.y, color=c, **MODEL_LINE)
            axis.plot(e.x, e.y, color=c, **EXP_MARKER)
            label_line(axis, float(m.x.iloc[-1]), float(m.y.iloc[-1]), '%g K' % temp, fontsize=6)
        axis.set(xlabel='<x label (unit)>', ylabel='<y label (unit)>')
        handles = [Line2D([], [], color=INK['secondary'], **EXP_MARKER, label='<Experiment, year>'),
                   Line2D([], [], color=INK['secondary'], **MODEL_LINE, label='This work'),
                   Line2D([], [], **THEORY_LINE, label='<Reference theory>')]  # fmt: skip
        axis.legend(handles=handles, loc='best')
        panel_label(axis, '(a)')
        provenance_note(
            fig, '<Which data are digitised / vector-extracted / interpolated, and from where.>'
        )
        return save_figure(fig, FIGURES / 'fig1_<name>')


if __name__ == '__main__':
    experiments = pd.concat(
        [read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')]
    )
    models = read_dataset(DATA / 'model_curves.csv')
    print('wrote', figure_comparison(experiments, models))
