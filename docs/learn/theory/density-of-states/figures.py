"""Schematic of adaptive smearing for the theory page density-of-states.md.

A one-dimensional nearest-neighbour band eps(k) = -2t cos(k a) sampled on a mesh of N points.
Panel (a): the band, the mesh points and the energy range each mesh cell spans, next to the
adaptive Gaussian width. Panel (b): the exact DOS against the fixed-width and the adaptive DOS
on the same mesh, both from the smearing functions PAOFLOW uses.

Run from anywhere: python docs/learn/theory/density-of-states/figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW.spectrum.do_adaptive_smearing import adaptive_widths
from PAOFLOW.utils.smearing import gaussian

OUT = Path(__file__).resolve().parents[3] / '_static' / 'images' / 'theory' / 'density-of-states'
BLUE, VERMILLION, BLACK = '#004488', '#D55E00', '#000000'

t, nmesh, afac, min_width, fixed = 1.0, 24, 0.7, 0.02, 0.15  # eV, points, a, eV, eV
k = 2 * np.pi * (np.arange(nmesh) / nmesh - 0.5)  # k a on the mesh
dk = 2 * np.pi / nmesh
eps = -2 * t * np.cos(k)
vel = 2 * t * np.sin(k)  # d eps / d(k a)
sigma = adaptive_widths(vel[None, :], afac, dk, min_width=min_width)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), gridspec_kw={'width_ratios': [1, 1.2]})

# (a) band, mesh points, energy span of each mesh cell and the adaptive width
kf = np.linspace(-np.pi, np.pi, 400)
ax1.plot(kf, -2 * t * np.cos(kf), color=BLUE, lw=1.5, label=r'$\varepsilon(k)$')
for ki, ei, si in zip(k, eps, sigma):
    lo, hi = -2 * t * np.cos(ki - dk / 2), -2 * t * np.cos(ki + dk / 2)
    ax1.plot([ki - 0.06] * 2, [min(lo, hi), max(lo, hi)], color=BLACK, lw=1.2)
    ax1.plot(
        [ki + 0.06] * 2, [ei - si / np.sqrt(2), ei + si / np.sqrt(2)], color=VERMILLION, lw=1.2
    )
ax1.plot(k, eps, 'o', color=BLUE, ms=3.5)
ax1.plot([], [], color=BLACK, lw=1.2, label='energy span of the cell')
ax1.plot([], [], color=VERMILLION, lw=1.2, label=r'Gaussian, $\pm$ one standard deviation')
ax1.set_xlabel(r'$k a$')
ax1.set_ylabel(r'$\varepsilon$ (eV)')
ax1.set_xlim(-np.pi, np.pi)
ax1.set_xticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
ax1.legend(fontsize=7, loc='upper center', frameon=False)
ax1.set_title('(a)', loc='left', fontsize=10)

# (b) exact DOS against the fixed-width and the adaptive DOS of the same mesh
ene = np.linspace(-2.6, 2.6, 2081)
inside = np.abs(ene) < 2 * t
exact = np.zeros_like(ene)
exact[inside] = 1 / (np.pi * np.sqrt(4 * t**2 - ene[inside] ** 2))
dos_fixed = gaussian(eps[None, :], ene[:, None], fixed).sum(axis=1) / nmesh
dos_adapt = gaussian(eps[None, :], ene[:, None], sigma[None, :]).sum(axis=1) / nmesh
ax2.plot(ene, exact, color=BLACK, lw=1.0, label='exact')
ax2.plot(ene, dos_fixed, color=VERMILLION, lw=1.3, label=rf'fixed width, $\delta$ = {fixed} eV')
ax2.plot(ene, dos_adapt, color=BLUE, lw=1.3, label=rf'adaptive, $a$ = {afac}, floor {min_width} eV')
ax2.set_xlabel(r'$\varepsilon$ (eV)')
ax2.set_ylabel(r'$g(\varepsilon)$ (states/eV per cell)')
ax2.set_ylim(0, 1.6)
ax2.set_xlim(ene[0], ene[-1])
ax2.legend(fontsize=7, loc='upper center', frameon=False)
ax2.set_title('(b)', loc='left', fontsize=10)

fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'adaptive_smearing_1d.png', dpi=150)

# the mesh sums count every state once
assert abs(np.trapezoid(dos_adapt, ene) - 1) < 1e-3, 'adaptive DOS does not integrate to 1'
assert abs(np.trapezoid(dos_fixed, ene) - 1) < 1e-3, 'fixed DOS does not integrate to 1'
