"""Figures and checks of the how-to "How to run PAOFLOW on a tight-binding model".

The page is docs/tutorials/howtos/tight-binding-model.md. Run it after ``python main.py`` in the
same directory: it reads the output folders of ``main.py``, writes the figures of the page to the
working directory and ends with asserts of the numbers the page quotes.
"""

import matplotlib

matplotlib.use('Agg')  # GPAO calls plt.show(); keep the figures open for savefig instead
import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import GPAO

plotter = GPAO.GPAO()
plotter.plot_dos_beside_bands(
    'kane_mele/dosdk_0.dat',
    'kane_mele/bands_0.dat',
    sym_points='kane_mele/kpath_points.txt',
    title=r'Kane–Mele model, $t = 1$ eV, $\lambda_{SO} = 0.1$ eV',
    band_label=r'$E$ (eV)',
    y_lim=(-3.4, 3.4),
    col='#004488',
)
plt.gcf().savefig('km_bands_dos.png', dpi=150)
plt.close('all')

ax = plotter.plot_bands(
    ['graphene_0/bands_0.dat', 'graphene_1/bands_0.dat'],
    sym_points='graphene_0/kpath_points.txt',
    title=r'Graphene, $t = 2.7$ eV',
    label=r'$E$ (eV)',
    labels=[r'$\delta = 0$', r'$\delta = 1$ eV'],
    cols=['#004488', '#D55E00'],
    show=False,
)
ax.figure.savefig('graphene_bands.png', dpi=150)
plt.close('all')

ax = plotter.plot_bands(
    'silicon/bands_0.dat',
    sym_points='silicon/kpath_points.txt',
    title=r'Si, nearest-neighbour $sp^3$ Slater–Koster model',
    label=r'$E$ (eV)',
    cols='#004488',
    show=False,
)
ax.figure.savefig('si_sk_bands.png', dpi=150)
plt.close('all')

# The numbers the page quotes, at the precision it quotes them.
km = np.loadtxt('kane_mele/bands_0.dat')[:, 1:]
km_gap = km[:, 2].min() - km[:, 1].max()  # four bands, each twofold degenerate
assert abs(km_gap - 1.0392) < 1e-4, f'Kane-Mele gap {km_gap:.4f} eV'
energy, dos = np.loadtxt('kane_mele/dosdk_0.dat', unpack=True)
electrons = np.trapezoid(dos[energy <= 0], energy[energy <= 0])
assert abs(electrons - 2.00) < 0.01, f'Kane-Mele states below E = 0: {electrons:.3f}'

flat, gapped = (np.loadtxt(f'graphene_{d}/bands_0.dat')[:, 1:] for d in ('0', '1'))
assert abs(flat[:, 1].max() - 8.1) < 0.05, f'graphene band top at Gamma {flat[:, 1].max():.3f} eV'
graphene_gap = gapped[:, 1].min() - gapped[:, 0].max()
assert abs(graphene_gap - 1.000) < 1e-3, f'gapped graphene gap {graphene_gap:.4f} eV'

si = np.loadtxt('silicon/bands_0.dat')[:, 1:]
si_vbm, si_cbm = si[:, 3].max(), si[:, 4].min()
assert abs(si_vbm + 0.229) < 1e-3, f'Si Slater-Koster VBM {si_vbm:.4f} eV'
assert abs(si_cbm - si_vbm - 2.309) < 1e-3, f'Si Slater-Koster gap {si_cbm - si_vbm:.4f} eV'

hal = np.loadtxt('haldane/bands_0.dat')[:, 1:]
hal_gap = hal[:, 1].min() - hal[:, 0].max()
assert abs(hal_gap - 6 * np.sqrt(3) * 0.15) < 1e-4, f'Haldane gap {hal_gap:.4f} eV'
