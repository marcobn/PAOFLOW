"""Reproduce the summary figures of a SKEAF-vs-experiment comparison (collaborator package).

Figure 1  fig1_sweep_overlay.png   calculated extremal orbits (NumOrbCopy >= fig1_min_copies) vs
                                   the experimental sweep peaks, with branch assignments.
Figure 2  fig2_network_orbits.png  magnetic-breakdown network orbits vs the measured peaks that no
                                   direct orbit explains (only if data/network_orbits.csv exists).

Requires only Python 3, numpy and matplotlib:  python plot_figures.py
Inputs: data/calc_orbits.csv, data/exp_sweep_peaks.csv, data/meta.json,
        optional data/branch_assignments.csv and data/network_orbits.csv (see README.md).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
META = json.loads((DATA / 'meta.json').read_text())
T = META['thresholds']
GREEK = [
    ('epsilon', 'ε'),
    ('kappa', 'κ'),
    ('sigma', 'σ'),
    ('gamma', 'γ'),
    ('delta', 'δ'),
    ('alpha', 'α'),
    ('beta', 'β'),
    ('mu', 'μ'),
    (' deg', '°'),
]


def read_csv(path):
    with open(path, newline='') as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith('#')]
    return {h: [r[i] for r in rows[1:]] for i, h in enumerate(rows[0])}


def num(t, k):
    return np.array([float(v) if v not in ('', 'nan') else np.nan for v in t[k]])


def txt(t, k):
    return np.array(t[k])


def greek(s):
    for w, g in GREEK:
        s = s.replace(w, g)
    return s


def load():
    e = read_csv(DATA / 'exp_sweep_peaks.csv')
    c = read_csv(DATA / 'calc_orbits.csv')
    exp = dict(angle=num(e, 'angle_deg'), F=num(e, 'freq_T'), mag=num(e, 'rel_mag'))
    calc = dict(
        angle=num(c, 'angle_deg'),
        F=num(c, 'freq_T'),
        amp=num(c, 'amp_factor'),
        copies=num(c, 'num_orbit_copies'),
        group=txt(c, 'group'),
    )
    net = None
    if (DATA / 'network_orbits.csv').exists():
        n = read_csv(DATA / 'network_orbits.csv')
        net = dict(
            angle=num(n, 'angle'),
            F=num(n, 'freq_T'),
            copies=num(n, 'copies'),
            R=num(n, 'R_ref'),
            open=txt(n, 'uses_open') == 'True',
        )
    ann = (
        read_csv(DATA / 'branch_assignments.csv')
        if (DATA / 'branch_assignments.csv').exists()
        else None
    )
    return exp, calc, net, ann


def exp_scatter(ax, e):
    weak, sig = e['mag'] < T['sig_mag'], e['mag'] >= T['sig_mag']
    ax.scatter(
        e['angle'][weak],
        e['F'][weak],
        s=10,
        facecolors='none',
        edgecolors='0.75',
        lw=0.6,
        label=f'exp. peak, weak (< {T["sig_mag"]:.2f})',
        zorder=1,
    )
    ax.scatter(
        e['angle'][sig],
        e['F'][sig],
        s=25 + 160 * e['mag'][sig],
        facecolors='none',
        edgecolors='k',
        lw=1.1,
        label='exp. peak (size = FFT magnitude)',
        zorder=3,
    )


def style(ax, title, ylim=(5, 10000)):
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    ax.set_xlabel(META.get('axis_label', 'angle (deg)'))
    ax.set_ylabel('F (T)')
    ax.set_title(title, fontsize=11)


def figure1(e, c, ann):
    keep = c['copies'] >= T['fig1_min_copies']
    amax = np.quantile(c['amp'][keep], 0.98)
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    exp_scatter(ax, e)
    for g in sorted(set(c['group'])):
        m = keep & (c['group'] == g)
        ax.scatter(
            c['angle'][m],
            c['F'][m],
            s=4 + 26 * np.clip(c['amp'][m] / amax, 0, 1),
            color=META['colors'].get(g, 'C0'),
            alpha=0.75,
            lw=0,
            zorder=2,
            label=META['labels'].get(g, f'calc. {g}'),
        )
    if ann:
        for i, label in enumerate(ann['label']):
            tx, ty = float(ann['text_angle_deg'][i]), float(ann['text_freq_T'][i])
            g = ann['group'][i]
            if ann['status'][i] == 'unassigned' or not g:
                ax.text(
                    tx,
                    ty,
                    greek(label),
                    fontsize=8,
                    style='italic',
                    color='0.35',
                    ha='center',
                    zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75),
                )
                continue
            col = META['colors'].get(g, 'k')
            ax.annotate(
                greek(label),
                xy=(float(ann['arrow_angle_deg'][i]), float(ann['arrow_freq_T'][i])),
                xytext=(tx, ty),
                fontsize=12,
                fontweight='bold',
                color=col,
                ha='center',
                va='center',
                zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=0.8, alpha=0.85),
                arrowprops=dict(arrowstyle='-|>', color=col, lw=1.0, shrinkA=2, shrinkB=1),
            )
    style(
        ax,
        f'{META["title"]}: calculated orbits (NumOrbCopy $\\geq$ {T["fig1_min_copies"]}) vs experiment',
    )
    ax.legend(fontsize=7.5, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(HERE / 'fig1_sweep_overlay.png')
    plt.close(fig)


def unexplained(e, c):
    sig = e['mag'] >= T['sig_mag']
    direct = c['copies'] >= T['direct_min_copies']
    out = np.zeros(len(e['F']), bool)
    for i in np.flatnonzero(sig):
        m = direct & (np.abs(c['angle'] - e['angle'][i]) <= T['angle_win'])
        out[i] = (np.min(np.abs(c['F'][m] - e['F'][i]) / e['F'][i]) if m.any() else np.inf) > T[
            'direct_tol'
        ]
    return out


def matched(e, n, mask, robust, freqs=None):
    freqs = e['F'] if freqs is None else freqs
    hit = np.zeros(len(freqs), bool)
    for i in np.flatnonzero(mask):
        m = robust & (n['angle'] == e['angle'][i])
        hit[i] = bool(m.any() and np.min(np.abs(n['F'][m] - freqs[i]) / freqs[i]) <= T['match_tol'])
    return hit


def figure2(e, c, n):
    un = unexplained(e, c)
    robust = (n['copies'] >= T['net_min_copies']) & (n['R'] >= T['net_min_R'])
    hit = matched(e, n, un, robust)
    rng = np.random.default_rng(3)
    rates = [
        matched(e, n, un, robust, np.exp(rng.uniform(np.log(15), np.log(8000), len(e['F']))))[
            un
        ].mean()
        for _ in range(300)
    ]
    print(
        f'Figure 2: {un.sum()} significant peaks without a direct orbit; {hit.sum()} matched by a robust '
        f'network orbit ({100 * hit.sum() / max(un.sum(), 1):.0f} %); chance {100 * np.mean(rates):.0f} +/- '
        f'{100 * np.std(rates):.0f} %'
    )
    direct = c['copies'] >= T['direct_min_copies']
    weak_open = n['open'] & ~robust & (n['copies'] >= 3)
    sig = e['mag'] >= T['sig_mag']
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    ax.scatter(c['angle'][direct], c['F'][direct], s=3, color='0.82', lw=0, label='direct orbits')
    for flag, col, lab in (
        (False, '#1f77b4', 'network orbit, closed pieces only'),
        (True, '#d62728', 'network orbit using open-sheet pieces'),
    ):
        m = robust & (n['open'] == flag)
        ax.scatter(
            n['angle'][m],
            n['F'][m],
            s=8 + 60 * n['R'][m],
            color=col,
            alpha=0.75,
            lw=0,
            label=f'{lab} (size ~ R)',
        )
    ax.scatter(
        n['angle'][weak_open],
        n['F'][weak_open],
        s=6,
        marker='s',
        facecolors='none',
        edgecolors='#d62728',
        lw=0.5,
        alpha=0.6,
        label='open-sheet network orbit, below threshold',
    )
    ax.scatter(
        e['angle'][sig],
        e['F'][sig],
        s=25 + 160 * e['mag'][sig],
        facecolors='none',
        edgecolors='0.45',
        lw=0.9,
        label='exp. peak',
    )
    ax.scatter(
        e['angle'][un],
        e['F'][un],
        s=25 + 160 * e['mag'][un],
        facecolors='none',
        edgecolors='k',
        lw=1.4,
        label='exp. peak, no direct orbit',
    )
    ax.scatter(
        e['angle'][hit],
        e['F'][hit],
        marker='x',
        s=45,
        color='k',
        lw=1.5,
        zorder=5,
        label=f'... matched by a network orbit ({T["match_tol"]:.0%})',
    )
    style(ax, f'{META["title"]}: magnetic-breakdown network orbits', ylim=(10, 10000))
    ax.legend(fontsize=7, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(HERE / 'fig2_network_orbits.png')
    plt.close(fig)


def main():
    e, c, n, ann = load()
    figure1(e, c, ann)
    if n is not None:
        figure2(e, c, n)
    print('figures written')


if __name__ == '__main__':
    main()
