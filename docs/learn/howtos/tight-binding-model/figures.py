"""Figure and checks of the how-to "How to run PAOFLOW on a tight-binding model".

The page is docs/learn/howtos/tight-binding-model.md. Run it after ``python tight_binding_model.py`` in the folder
of that run, by the path of this script (``python
<PAOFLOW>/docs/learn/howtos/tight-binding-model/figures.py``): it reads ``kane_mele/``, writes the
figure of the page to the working directory and ends with asserts of the numbers the page quotes.
"""

import matplotlib

matplotlib.use('Agg')  # GPAO calls plt.show(); keep the figure open for savefig instead
import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import GPAO

GPAO.GPAO().plot_dos_beside_bands(
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

# The numbers the page quotes, at the precision it quotes them.
bands = np.loadtxt('kane_mele/bands_0.dat')[:, 1:]
gap = bands[:, 2].min() - bands[:, 1].max()  # four bands, each twofold degenerate
assert abs(gap - 1.0392) < 1e-4, f'Kane-Mele gap {gap:.4f} eV'
energy, dos = np.loadtxt('kane_mele/dosdk_0.dat', unpack=True)
electrons = np.trapezoid(dos[energy <= 0], energy[energy <= 0])
assert abs(electrons - 2.00) < 0.005, f'Kane-Mele states below E = 0: {electrons:.3f}'
energy, shc = np.loadtxt('kane_mele/shcEf_z_xy.dat', unpack=True)
plateau = shc[np.abs(energy).argmin()]
assert abs(plateau - 0.158987) < 2e-6, f'spin Hall plateau at E = 0 {plateau:.6f}'
