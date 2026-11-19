#!/usr/bin/env python3
"""Figures of an SSCHA campaign (template: copy to <study>/plot_figures.py).

    python plot_figures.py      # -> figures/convergence_T<T>.{pdf,png}, figures/scan.{pdf,png},
                                #    figures/dispersions.{pdf,png}

Reads the run directories (read-only) and tables/ written by analysis.py. Conventions: temperature is a
single-hue ramp (at most five per panel), the harmonic reference a dashed gray line, computed points markers.
Look at every figure before using it. Playbook stage 8: PAOFLOW/agentic/sscha/AGENT_WORKFLOW.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import pandas as pd  # noqa: E402

from PAOFLOW.agentic.sscha import dispersion, find_runs, stability_temperature  # noqa: E402
from PAOFLOW.agentic.sscha.plotting import (  # noqa: E402
    plot_convergence,
    plot_dispersions,
    plot_scan,
    publication_style,
    save_figure,
)

HERE = Path(__file__).resolve().parent
RUN_ROOT = Path('<run folder>/output')  # read-only
HARMONIC_DYN = '<run folder>/harmonic/<prefix>.dyn'
PATH = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.25, 0.75, 0.5], [0.375, 0.75, 0.375], [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]]  # fcc G-X-W-K-G-L (crystal coordinates of the reciprocal lattice)  # fmt: skip
LABELS = ['G', 'X', 'W', 'K', 'G', 'L']
EXPERIMENT = None  # {'temperature': K, 'low': K, 'high': K, 'label': '<what, source>'}
FIGURES = HERE / 'figures'


def main() -> None:
    runs = find_runs(str(RUN_ROOT))
    with publication_style():
        for run in runs:
            if run.history():
                figure, _ = plot_convergence(run)
                save_figure(figure, FIGURES / ('convergence_T%g' % run.temperature))
        scan = HERE / 'tables' / 'scan.csv'
        if scan.exists():
            table = pd.read_csv(scan)
            rows = [dict(temperature=t, lowest_cm1=w)
                    for t, w in zip(table['temperature_k'], table['softest_fine_cm1'])]  # fmt: skip
            result = (
                stability_temperature(table['temperature_k'], table['softest_fine_cm1'])
                if len(rows) > 1
                else None
            )
            figure, _ = plot_scan(rows, result, experiment=EXPERIMENT)
            save_figure(figure, FIGURES / 'scan')
        curves = {run.temperature: dispersion(run.force_constants('hessian'), PATH, LABELS, 30)
                  for run in runs if run.force_constants('hessian')}  # fmt: skip
        if curves:
            chosen = dict(list(sorted(curves.items()))[-5:])  # at most five temperatures per panel
            figure, _ = plot_dispersions(
                chosen, reference=dispersion(HARMONIC_DYN, PATH, LABELS, 30)
            )
            save_figure(figure, FIGURES / 'dispersions')


if __name__ == '__main__':
    main()
