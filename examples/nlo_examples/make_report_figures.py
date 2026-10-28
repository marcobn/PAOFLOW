#!/usr/bin/env python3
"""Figures of REPORT.md from the outputs of the three examples (PDF + 600 dpi PNG).

    python make_report_figures.py [--graphene DIR] [--gaas DIR] [--gaas-conv DIR] [--ges DIR] [--out figures]

Each DIR is an output folder of PAOFLOW.nonlinear_optics (defaults: the examples' output/).
The model is drawn as lines (one hue per component, thin), literature values as gray markers
or bands with their source in the legend; one axis per panel.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from PAOFLOW.agentic.nlo import NonlinearModel  # noqa: E402
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
COLORS = [
    TEMPERATURE_RAMP[4],
    TEMPERATURE_RAMP_ALT[3],
    TEMPERATURE_RAMP[2],
    TEMPERATURE_RAMP_ALT[1],
    INK['muted'],
]
LINE = dict(lw=1.2, solid_capstyle='round')
REF = dict(color=REFERENCE_THEORY, lw=0.9, ls=(0, (4, 2)))


def graphene(model, out):
    with publication_style(7.0, 2.6):
        fig, (a, b) = plt.subplots(1, 2)
        e = model.energy_ev
        for i, (comp, label) in enumerate(
            (('yyy', 'yyy'), ('yxx', 'yxx'), ('xxy', 'xxy'), ('xxx', 'xxx'))
        ):
            a.plot(e, model.spectrum('shift', comp), color=COLORS[i], label=label, **LINE)
        a.set(xlabel='photon energy (eV)', ylabel='shift current (nm $\\mu$A/V$^2$)', xlim=(0, 4))
        a.legend(title='$\\sigma^{abc}$', ncol=2)
        panel_label(a, '(a)')
        for i, comp in enumerate(('yyy', 'yxx')):
            b.plot(
                e,
                model.spectrum('shga', comp) / 1e6,
                color=COLORS[i],
                label='|$\\chi^{%s}$|' % comp,
                **LINE,
            )
        b.plot(
            e,
            model.spectrum('shgr', 'yyy') / 1e6,
            color=COLORS[0],
            ls=(0, (2, 1)),
            lw=0.9,
            label='Re $\\chi^{yyy}$',
        )
        b.plot(
            e,
            model.spectrum('shgi', 'yyy') / 1e6,
            color=COLORS[0],
            ls=(0, (1, 1)),
            lw=0.9,
            label='Im $\\chi^{yyy}$',
        )
        b.set(xlabel='photon energy (eV)', ylabel='$\\chi^{(2)}$ (10$^6$ pm$^2$/V)', xlim=(0, 3))
        b.legend()
        panel_label(b, '(b)')
        provenance_note(
            fig,
            'Gapped graphene tight-binding model (t = 2.7 eV, delta = 1 eV), 300x300 k, eta 0.05 eV; sheet units with the 10 alat cell. yyy = -yxx = -xxy, xxx = 0 (D3h).',
        )
        return save_figure(fig, out / 'fig1_graphene')


def gaas(model, conv, out):
    with publication_style(7.0, 2.6):
        fig, (a, b) = plt.subplots(1, 2)
        e = model.energy_ev
        a.plot(
            e,
            model.spectrum('shga', 'xyz'),
            color=COLORS[0],
            label='|$\\chi_{xyz}$| 40$^3$',
            **LINE,
        )
        a.plot(
            e, model.spectrum('shgr', 'xyz'), color=COLORS[0], ls=(0, (2, 1)), lw=0.9, label='Re'
        )
        a.plot(
            e, model.spectrum('shgi', 'xyz'), color=COLORS[0], ls=(0, (1, 1)), lw=0.9, label='Im'
        )
        a.plot(
            e,
            model.spectrum('shga', 'yzx'),
            color=COLORS[1],
            lw=0.8,
            label='|$\\chi_{yzx}$| 40$^3$',
        )
        if conv is not None:
            a.plot(
                conv.energy_ev,
                conv.spectrum('shga', 'xyz'),
                color=COLORS[2],
                lw=0.9,
                label='|$\\chi_{xyz}$| 60$^3$',
            )
        # low-frequency experiments (Levine-Bethea 180, Roberts 166, Eyres 172 pm/V at ~0.12 eV)
        a.errorbar(
            [0.12],
            [172.0],
            yerr=[10.0],
            fmt='o',
            ms=3.5,
            color=INK['secondary'],
            capsize=2,
            label='exp. 0.12 eV (166-180)',
        )
        a.axhspan(
            207.0,
            245.0,
            color=INK['grid'],
            alpha=0.6,
            lw=0,
            label='TDDFT+scissor static (Luppi 2010)',
        )
        a.set(xlabel='photon energy (eV)', ylabel='$\\chi^{(2)}$ (pm/V)', xlim=(0, 4))
        a.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.9)
        panel_label(a, '(a)')
        b.plot(
            e,
            model.spectrum('shift', 'xyz'),
            color=COLORS[0],
            label='$\\sigma_{xyz}$ 40$^3$',
            **LINE,
        )
        b.plot(
            e,
            model.spectrum('shift', 'yzx'),
            color=COLORS[1],
            lw=0.8,
            label='$\\sigma_{yzx}$ 40$^3$',
        )
        b.plot(
            e,
            model.spectrum('shift', 'xxx'),
            color=COLORS[4],
            lw=0.8,
            label='$\\sigma_{xxx}$ (forbidden)',
        )
        if conv is not None:
            b.plot(
                conv.energy_ev,
                conv.spectrum('shift', 'xyz'),
                color=COLORS[2],
                lw=0.9,
                label='$\\sigma_{xyz}$ 60$^3$',
            )
        b.set(xlabel='photon energy (eV)', ylabel='shift current ($\\mu$A/V$^2$)', xlim=(0, 4))
        b.legend(fontsize=6, loc='lower left')
        panel_label(b, '(b)')
        provenance_note(
            fig,
            'GaAs, PAW 16^3 nscf (symmetry-reduced), 62-orbital PAO basis, scissor 0.86 eV, eta 0.05 eV, 10 meV degeneracy tolerance. Experimental static values from the literature summary in REPORT.md.',
        )
        return save_figure(fig, out / 'fig2_gaas')


def ges(model, out):
    with publication_style(7.0, 2.6):
        fig, (a, b) = plt.subplots(1, 2)
        e = model.energy_ev
        for i, comp in enumerate(('xxx', 'xyy', 'yxy', 'yyy')):
            a.plot(
                e,
                model.spectrum('shift', comp),
                color=COLORS[i],
                label='$\\sigma^{%s}$' % comp,
                **LINE,
            )
        a.axhspan(
            100.0 * 0.6,
            150.0 * 0.6,
            color=INK['grid'],
            alpha=0.6,
            lw=0,
            label='lit. 100-150 $\\mu$A/V$^2$ x 0.6 nm',
        )
        a.set(
            xlabel='photon energy (eV)', ylabel='shift current (nm $\\mu$A/V$^2$)', xlim=(1.0, 4.0)
        )
        a.legend(fontsize=6, ncol=2, loc='upper right', framealpha=0.9)
        panel_label(a, '(a)')
        for i, comp in enumerate(('xxx', 'xyy', 'yxy')):
            b.plot(
                e,
                model.spectrum('shga', comp) / 1e6,
                color=COLORS[i],
                label='|$\\chi^{%s}$|' % comp,
                **LINE,
            )
        b.set(xlabel='photon energy (eV)', ylabel='$\\chi^{(2)}$ (10$^6$ pm$^2$/V)', xlim=(0, 4))
        b.legend(fontsize=6)
        panel_label(b, '(b)')
        provenance_note(
            fig,
            'Monolayer GeS (Pmn2_1, a = 4.40, b = 3.68 A, PBE gap 1.68 eV), nosym 18x18x1 nscf, BASIS_PS, 200x200 k, eta 0.05 eV, no scissor; sheet units with the 18 A cell. yyy and xxy vanish (mirror).',
        )
        return save_figure(fig, out / 'fig3_ges')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--graphene', default=str(HERE / 'Graphene' / 'output'))
    parser.add_argument('--gaas', default=str(HERE / 'GaAs' / 'output'))
    parser.add_argument('--gaas-conv', default=None, help='a second GaAs output (denser grid)')
    parser.add_argument('--ges', default=str(HERE / 'GeS' / 'output'))
    parser.add_argument('--out', default=str(HERE / 'figures'))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if Path(args.graphene).is_dir():
        written += graphene(NonlinearModel(args.graphene, two_d=True), out)
    if Path(args.gaas).is_dir():
        conv = (
            NonlinearModel(args.gaas_conv)
            if args.gaas_conv and Path(args.gaas_conv).is_dir()
            else None
        )
        written += gaas(NonlinearModel(args.gaas), conv, out)
    if Path(args.ges).is_dir():
        written += ges(NonlinearModel(args.ges, two_d=True), out)
    print('\n'.join(str(w) for w in written))
    _ = Line2D, np


if __name__ == '__main__':
    main()
