"""Schematic of extremal orbits for the theory page quantum-oscillations.md.

A warped cylindrical Fermi sheet with radius r(k_H) = r0 + d cos(2 pi k_H / G) along the field.
Panel (a): the sheet cut by planes perpendicular to B at a fixed spacing, with the belly (maximal
cross-section) and the neck (minimal cross-section). Panel (b): the area A(k_H) of the
cross-sections, sampled on the slices, and the slices where a sampled area is a local extremum.

Run from anywhere: python docs/learn/theory/quantum-oscillations/figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[3] / '_static' / 'images' / 'theory' / 'quantum-oscillations'
BLUE, VERMILLION, AMBER, BLACK = '#004488', '#D55E00', '#FFC20A', '#000000'

r0, d, G = 1.0, 0.25, 2.0  # radius, warping amplitude, period along k_H (arbitrary units)
kh = np.linspace(-0.8, 1.6, 400)
radius = r0 + d * np.cos(2 * np.pi * kh / G)
slices = np.arange(-0.75, 1.6, 0.22)  # slice positions, off the true extrema at 0 and 1
area_s = np.pi * (r0 + d * np.cos(2 * np.pi * slices / G)) ** 2
local = [i for i in range(1, len(slices) - 1)
         if (area_s[i] - area_s[i - 1]) * (area_s[i + 1] - area_s[i]) < 0]  # fmt: skip

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={'width_ratios': [1, 1.3]})
for sign in (1, -1):
    ax1.plot(sign * radius, kh, color=BLUE, lw=2)
for i, s in enumerate(slices):
    r = r0 + d * np.cos(2 * np.pi * s / G)
    ax1.plot([-r, r], [s, s], color=VERMILLION if i in local else '#56B4E9',
             lw=2.0 if i in local else 0.8)  # fmt: skip
ax1.annotate('', xy=(1.55, 1.4), xytext=(1.55, 0.8), arrowprops=dict(arrowstyle='->', color=BLACK))
ax1.text(1.6, 1.1, r'$\mathbf{B}$', va='center')
ax1.text(0, -0.03, 'belly', ha='center', va='top', fontsize=8)
ax1.text(0, 0.97, 'neck', ha='center', va='top', fontsize=8)
ax1.set(xlim=(-1.8, 1.9), ylim=(-0.9, 1.7), xlabel=r'$k_\perp$', ylabel=r'$k_H$',
        title='(a) slices perpendicular to the field')  # fmt: skip
ax1.set_xticks([])
ax1.set_yticks([])

ax2.plot(kh, np.pi * radius**2, color=BLUE, lw=2, label=r'$A(k_H)$')
ax2.plot(slices, area_s, 'o', color='#56B4E9', ms=5, label='area on a slice')
ax2.plot(slices[local], area_s[local], 'o', color=VERMILLION, ms=8,
         label='local extremum of the sampled areas')  # fmt: skip
ax2.axhline(np.pi * (r0 + d) ** 2, color=BLACK, lw=0.6, ls='--')
ax2.axhline(np.pi * (r0 - d) ** 2, color=BLACK, lw=0.6, ls='--')
ax2.set(xlabel=r'$k_H$', ylabel=r'$A$ (arb. units)', title='(b) cross-section area along the field')
ax2.set_xticks([])
ax2.legend(fontsize=7.5, loc='upper right', framealpha=0.9)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'extremal_orbits.png', dpi=150)
