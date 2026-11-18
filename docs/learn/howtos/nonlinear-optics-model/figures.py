"""Figures and checks of the how-to "How to compute the non-linear optical response of a
tight-binding model".

Run it after ``python main.py > main.log``, in the same directory: it reads the spectra in
``nlo/`` and ``nlo_inv/``, draws ``nlo_shift.png`` and ``nlo_shg.png`` and asserts the numbers
the page quotes.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # GPAO calls show(): write the figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import GPAO

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')

plotter = GPAO.GPAO()
plotter.plot_optical(
    'shift',
    path='nlo',
    component=['yyy', 'yxx', 'xxy', 'xxx'],
    x_lim=(0.0, 4.0),
    title=r'Gapped graphene, $\delta = 1$ eV: shift current',
    labels=[r'$\sigma$'],
    cols=['#004488', '#D55E00', '#FFC20A', '#009E73'],
    y_label=r'$\sigma$ (nm $\mu$A/V$^2$)',
)
plt.gcf().savefig('nlo_shift.png', dpi=150, bbox_inches='tight')  # bbox: keep the y label
plotter.plot_optical(
    ['shgr', 'shgi', 'shga'],
    path='nlo',
    component='yyy',
    x_lim=(0.0, 3.0),
    title=r'Gapped graphene, $\delta = 1$ eV: $\chi^{yyy}$',
    cols=['#004488', '#D55E00', '#000000'],
    y_label=r'$\chi^{(2)}$ (pm$^2$/V)',
)
plt.gcf().savefig('nlo_shg.png', dpi=150, bbox_inches='tight')
plt.close('all')


def spectrum(name: str, path: str = 'nlo') -> tuple[np.ndarray, np.ndarray]:
    """Photon energy (eV) and value of one output file."""
    return np.loadtxt(f'{path}/{name}.dat', unpack=True)


energy, yyy = spectrum('shift_yyy')
yxx, xxy, xxx = (spectrum(f'shift_{c}')[1] for c in ('yxx', 'xxy', 'xxx'))
symmetry = max(np.abs(yyy + yxx).max(), np.abs(yyy + xxy).max())
_, chi = spectrum('shga_yyy')
_, chi_re = spectrum('shgr_yyy')
gap = (energy > 0.8) & (energy < 1.5)  # the one-photon resonance at the 1 eV gap
re_gap = chi_re[gap].min()
inv_shift, inv_chi = spectrum('shift_yyy', 'nlo_inv')[1], spectrum('shga_yyy', 'nlo_inv')[1]

# The numbers the page quotes, at the precision it quotes them.
assert abs(yyy.max() - 14.74) < 0.01, f'shift-current peak {yyy.max():.3f} nm muA/V^2'
assert abs(energy[yyy.argmax()] - 1.07) < 0.005, f'shift-current peak at {energy[yyy.argmax()]} eV'
assert symmetry < 1e-3 and np.abs(xxx).max() < 1e-10, f'D3h relations {symmetry:.1e}'
assert abs(chi.max() / 8.85e5 - 1) < 1e-3, f'two-photon SHG peak {chi.max():.4e} pm^2/V'
assert abs(energy[chi.argmax()] - 0.51) < 0.005, f'two-photon SHG peak at {energy[chi.argmax()]} eV'
assert abs(re_gap / -3.33e5 - 1) < 2e-3, f'Re chi at the gap {re_gap:.4e} pm^2/V'
assert np.abs(inv_shift).max() < 1e-6 and inv_chi.max() < 1e-4, 'responses with inversion'
