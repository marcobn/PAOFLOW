"""Figure and checks of the how-to "How to compare pyskeaf orbits with measured quantum-oscillation
frequencies" (docs/learn/howtos/quantum-oscillations.md).

Run it after ``python main.py > main.log``, in the same directory. It reads the tables that
``main.py`` writes to ``qo_analysis/sweep/``, its printed output ``main.log``, and the pyskeaf
tables of ``quantum-oscillations/pyskeaf/`` for the calculated orbits, draws the figure of the
page and asserts the numbers the page quotes.
"""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PAOFLOW.agentic.qo_analysis.orbit_tables import robust_orbits
from PAOFLOW.agentic.qo_analysis.skeaf_output import read_orbits

palette = {'11,12': '#004488', '13,14': '#D55E00', '15,16': '#FFC20A', '17,18': '#009E73'}
pairs = {b: f'{b - 1 + b % 2},{b + b % 2}' for b in range(11, 19)}  # Kramers pairs, as in main.py
orbits = robust_orbits(read_orbits('quantum-oscillations/pyskeaf', list(pairs), '0', pairs), 50)
table = pd.read_csv('qo_analysis/sweep/sweep_match_table.csv', dtype={'group': str})
table_100 = pd.read_csv('qo_analysis/sweep/sweep_match_table_N100.csv')

# Calculated orbits over the measured peaks, matched peaks open, unmatched crossed.
fig, ax = plt.subplots(figsize=(6.4, 5.2))
for group, members in orbits.groupby('group'):
    ax.scatter(members.polar_deg, members.freq_T, s=3, lw=0, color=palette[group],
               label=f'calc. bands {group}')  # fmt: skip
hit = table.matched.to_numpy()
ax.scatter(table.calc_angle[hit], table.F_exp[hit], s=36, facecolors='none', edgecolors='#000000',
           lw=1.0, label='measured, matched within 5 %')  # fmt: skip
ax.scatter(table.calc_angle[~hit], table.F_exp[~hit], s=30, marker='x', color='#000000', lw=1.0,
           label='measured, no orbit within 5 %')  # fmt: skip
ax.set(yscale='log', ylim=(8, 1e4), xlim=(-2, 92), ylabel='$F$ (T)',
       xlabel='angle from [001] towards [110] (deg)')  # fmt: skip
ax.legend(fontsize=7.5, loc='lower right', markerscale=1.5, framealpha=0.9)
fig.tight_layout()
fig.savefig('nbsn2_sweep_overlay.png', dpi=150)

# The tables of direction_analysis as main.py printed them: {(first, second): {label: row}}
log = open('main.log').read()
chance = [int(p) for p in re.findall(r'log-uniform (\d+) \+/- \d+ %, angle-shuffled (\d+)', log)[0]]
chance_100 = int(re.findall(r'log-uniform (\d+) \+/-', log)[1])
anisotropy, pair = {}, None
for line in log.splitlines():
    if m := re.fullmatch(r'(\S+) -> (\S+)', line):
        pair = anisotropy.setdefault(m.groups(), {})
    elif pair is not None and re.match(r'\s*\w+ \d+,\d+\s', line):
        label, group, *values = line.split()[:9]
        pair[label] = dict(zip(['F_ref_exp', 'F_ref_calc', 'F_cmp_exp', 'F_cmp_calc', 'ratio_exp',
                                'ratio_calc', '1+lambda_ref'], map(float, values)))  # fmt: skip

# Numbers the page quotes, at the precision it quotes them.
rate = table.matched.mean()
assert len(table) == 160 and abs(rate - 0.44) < 0.005, f'rate, min_copies=50: {rate}'
rates = [chance[0], chance[1], round(100 * table_100.matched.mean()), chance_100]
assert rates == [41, 39, 38, 35], f'baselines and rate N100 (%) {rates}'
mu_5 = table.loc[(table.angle_deg == 5) & (table.F_exp > 3000)].iloc[0]
assert mu_5.group == '13,14' and abs(mu_5.F_calc - 3669) < 1, f'orbit matched at 5 deg {mu_5}'
best = anisotropy['[001]', '[11y]']
ratios = [best[label]['ratio_calc'] for label in ('delta', 'mu', 'sigma1', 'sigma2')]
assert np.allclose(ratios, [1.256, 1.583, 0.981, 0.981], atol=1e-3), f'calc. ratios {ratios}'
lam = [best[label]['1+lambda_ref'] for label in ('mu', 'sigma2', 'delta')]
lam.append(anisotropy['[001]', '[110]']['beta1']['1+lambda_ref'])
assert np.allclose(lam, [2.48, 2.75, 1.09, 2.85], atol=0.005), f'1 + lambda {lam}'
