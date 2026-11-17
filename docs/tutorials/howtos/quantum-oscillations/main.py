"""Code of the how-to "How to compare pyskeaf orbits with measured quantum-oscillation frequencies"
(docs/tutorials/howtos/quantum-oscillations.md).

Run it from the directory that holds ``quantum-oscillations/``, the folder of
``howto_assets.tar.gz`` of the release tutorial-assets-v2. It writes the tables of the analysis to
``qo_analysis/``, the figure of the page to the working directory, and ends with asserts of the
numbers the page quotes.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')


def save(*names: str) -> None:
    """Write the last ``len(names)`` figures, in the order they were drawn, and close all."""
    for num, name in zip(plt.get_fignums()[-len(names) :], names):
        plt.figure(num).savefig(name, dpi=150)
    plt.close('all')


# [setup]
from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis

qo = QuantumOscillationAnalysis(
    skeaf_dir='quantum-oscillations/pyskeaf',  # qo_EF_0_freqvsangle_<band>.out[.gz]
    output_dir='qo_analysis',
    bands=[11, 12, 13, 14, 15, 16, 17, 18],
    degenerate_groups=2,  # spin-orbit run: Kramers pairs (11,12), (13,14), ...
    sweep_angle='polar',
    fixed_angle=29.95,
    primitive_cell=[[0.5, 0.0, 1.690651], [0.5, 0.867802, 0.0], [0.0, 0.867802, 1.690651]],
    conventional_from_primitive=[[1, 1, -1], [-1, 1, 1], [1, -1, 1]],
    field_convention='reciprocal',
    system_name='NbSn2',
)
inventory = qo.fermi_surface_inventory()
print(inventory['copy_histogram'].to_string(index=False))
# [/setup]

# [directions]
print(
    qo.field_directions(
        [
            dict(label='[001]', hkl=[0, 0, 1], azimuthal=29.95, polar=0.0, day=3),
            dict(label='[11y]', hkl=[1, 1, 3.902], azimuthal=29.95, polar=45.0),
            dict(label='[110]', hkl=[1, 1, 0], azimuthal=29.95, polar=90.0, day=3),
        ]
    )
)
# [/directions]

# [experiment]
qo.read_experiment(
    sweep_peaks='quantum-oscillations/experiment/exp_sweep_peaks.csv',
    lk_table='quantum-oscillations/experiment/exp_LK_table.csv',
    angle_scale=1.0,  # calculated angle = angle_scale * measured angle + angle_offset
    angle_offset=0.0,
)
# [/experiment]

# [sweep]
palette = {'11,12': '#004488', '13,14': '#D55E00', '15,16': '#FFC20A', '17,18': '#009E73'}
sweeps = {}
for min_copies in (50, 100):
    sweeps[min_copies] = qo.sweep_comparison(
        min_copies=min_copies,
        angle_window=2.6,  # deg around each measured angle
        tolerance=0.05,  # |F_calc - F_exp| / F_exp
        weak_threshold=0.15,  # peaks below 15 % of the largest FFT magnitude are not tested
        colors=palette,
    )
    print(sweeps[min_copies]['text'])
# [/sweep]

# [anisotropy]
directions = qo.direction_analysis(min_copies=50, tolerance=0.05, anisotropy_window=0.35)
columns = ['label', 'group', 'F_ref_exp', 'F_ref_calc', 'F_cmp_exp', 'F_cmp_calc', 'ratio_exp',
           'ratio_calc', '1+lambda_ref', 'verdict']  # fmt: skip
for (first, second), best in directions['anisotropy'].items():
    print(f'{first} -> {second}')
    print(best[columns].round(3).to_string(index=False))
# [/anisotropy]

# Figure: calculated orbits over the measured peaks, matched peaks open, unmatched crossed.
import numpy as np

orbits = qo.orbits(min_copies=50)
table = sweeps[50]['table']
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
save('nbsn2_sweep_overlay.png')

# Numbers the page quotes, at the precision it quotes them.
n50, n100 = sweeps[50], sweeps[100]
assert n50['n'] == 160 and abs(n50['rate'] - 0.44) < 0.005, f'rate, min_copies=50: {n50["rate"]}'
rates = [n50['loguniform'][0], n50['shuffled'][0], n100['rate'], n100['loguniform'][0]]
assert np.allclose(rates, [0.41, 0.39, 0.38, 0.35], atol=0.005), f'baselines and rate N100 {rates}'
mu_5 = table.loc[(table.angle_deg == 5) & (table.F_exp > 3000)].iloc[0]
assert mu_5.group == '13,14' and abs(mu_5.F_calc - 3669) < 1, f'orbit matched at 5 deg {mu_5}'
best = directions['anisotropy'][('[001]', '[11y]')].set_index('label')
ratios = best.loc[['delta', 'mu', 'sigma1', 'sigma2'], 'ratio_calc'].to_numpy()
assert np.allclose(ratios, [1.256, 1.583, 0.981, 0.981], atol=1e-3), f'calc. ratios {ratios}'
beta1 = directions['anisotropy'][('[001]', '[110]')].set_index('label').loc['beta1']
lam = [*best.loc[['mu', 'sigma2', 'delta'], '1+lambda_ref'], beta1['1+lambda_ref']]
assert np.allclose(lam, [2.48, 2.75, 1.09, 2.85], atol=0.005), f'1 + lambda {lam}'
