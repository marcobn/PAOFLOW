#!/usr/bin/env python3
"""Publication figures (template: copy to <study>/plot_figures.py); reads data/ and tables/ only.

|chi|, Re chi, Im chi per component (model lines, experiments markers, published calculations
gray dashed), the shift current per component with the forbidden ones thin; sheet units stated;
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
    TEMPERATURE_RAMP_ALT,
    panel_label,
    provenance_note,
    publication_style,
    save_figure,
)

HERE = Path(__file__).resolve().parent
DATA, FIGURES = HERE / 'data', HERE / 'figures'
COLORS = [TEMPERATURE_RAMP[4], TEMPERATURE_RAMP_ALT[3], TEMPERATURE_RAMP[2]]
EXP_MARKER = dict(ls='none', marker='o', ms=3.2, mew=0.4, mec='white', color=INK['secondary'])
MODEL_LINE = dict(lw=1.2, solid_capstyle='round')
THEORY_LINE = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))


def figure_shg(exp, model):
    with publication_style(3.4, 2.6):
        fig, axis = plt.subplots()
        for i, (_, group) in enumerate(
            model[model.quantity.isin(['shg_chi', 'shg_d'])].groupby('dataset', sort=False)
        ):
            axis.plot(group.x, group.y, color=COLORS[i % 3], **MODEL_LINE)
        for _, group in exp[exp.quantity.isin(['shg_chi', 'shg_d'])].groupby('dataset', sort=False):
            style = THEORY_LINE if group.provenance.iloc[0] == 'calculated' else EXP_MARKER
            axis.plot(group.x, group.y, **style)
        axis.set(xlabel='photon energy (eV)', ylabel='<$\\chi^{(2)}$ or d (pm/V)>')
        handles = [Line2D([], [], **EXP_MARKER, label='<Experiment, year> (<convention>)'),
                   Line2D([], [], color=COLORS[0], **MODEL_LINE, label='This work (<grid, eta, scissor>)'),
                   Line2D([], [], **THEORY_LINE, label='<Reference calculation>')]  # fmt: skip
        axis.legend(handles=handles, loc='best')
        panel_label(axis, '(a)')
        provenance_note(fig, '<conventions, photon energies, scissor, symmetry error bar>')
        return save_figure(fig, FIGURES / 'fig1_shg')


if __name__ == '__main__':
    experiments = pd.concat(
        [read_dataset(p) for p in sorted(DATA.glob('*.csv')) if not p.name.startswith('model_')]
    )
    models = read_dataset(DATA / 'model_curves.csv')
    print('wrote', figure_shg(experiments, models))
