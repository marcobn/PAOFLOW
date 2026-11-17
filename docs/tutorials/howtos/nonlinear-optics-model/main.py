"""Code of the how-to "How to compute the non-linear optical response of a tight-binding model".

The page is docs/tutorials/howtos/nonlinear-optics-model.md. The model is built in code, so the
script needs no input files. It writes the figures of the page to the working directory and ends
with asserts of the numbers the page quotes.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')


def save(*names: str) -> None:
    """Write the last ``len(names)`` figures, in the order they were drawn, and close all."""
    for num, name in zip(plt.get_fignums()[-len(names) :], names):
        plt.figure(num).savefig(name, dpi=150, bbox_inches='tight')  # keep the y label
    plt.close('all')


# [nlo]
from PAOFLOW import PAOFLOW

model = {'label': 'graphene', 't': 2.7, 'delta': 1.0}  # sublattice potential: no inversion
paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='nlo')
paoflow.nonlinear_optics(
    nk=(300, 300, 1),
    emin=0.0,
    emax=6.0,
    ne=601,
    eta=0.05,
    components=['yyy', 'yxx', 'xxy', 'xxx'],
    nelec=2,  # a model has no electron count of its own: fill the lower band
    twoD=True,
)
paoflow.finish_execution()
# [/nlo]

# [plot]
from PAOFLOW import GPAO

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
plotter.plot_optical(
    ['shgr', 'shgi', 'shga'],
    path='nlo',
    component='yyy',
    x_lim=(0.0, 3.0),
    title=r'Gapped graphene, $\delta = 1$ eV: $\chi^{yyy}$',
    cols=['#004488', '#D55E00', '#000000'],
    y_label=r'$\chi^{(2)}$ (pm$^2$/V)',
)
# [/plot]
save('nlo_shift.png', 'nlo_shg.png')

# [inversion]
centrosymmetric = PAOFLOW.PAOFLOW(
    model={'label': 'graphene', 't': 2.7, 'delta': 0.0}, outputdir='nlo_inv'
)
centrosymmetric.nonlinear_optics(
    nk=(120, 120, 1), emin=0.0, emax=6.0, ne=601, eta=0.05, components=['yyy'], nelec=2, twoD=True
)
centrosymmetric.finish_execution()
# [/inversion]


# [checks]
def spectrum(name: str, path: str = 'nlo') -> tuple[np.ndarray, np.ndarray]:
    """Photon energy (eV) and value of one output file."""
    return np.loadtxt(f'{path}/{name}.dat', unpack=True)


energy, yyy = spectrum('shift_yyy')
yxx, xxy, xxx = (spectrum(f'shift_{c}')[1] for c in ('yxx', 'xxy', 'xxx'))
symmetry = max(np.abs(yyy + yxx).max(), np.abs(yyy + xxy).max())
_, chi = spectrum('shga_yyy')
_, chi_re = spectrum('shgr_yyy')
gap = (energy > 0.8) & (energy < 1.5)  # the one-photon resonance at the 1 eV gap
e_gap, re_gap = energy[gap][chi_re[gap].argmin()], chi_re[gap].min()
print(f'shift current: peak {yyy.max():.2f} nm muA/V^2 at {energy[yyy.argmax()]:.2f} eV')
print(f'  |yyy + yxx|, |yyy + xxy| <= {symmetry:.1e}, |xxx| <= {np.abs(xxx).max():.1e}')
print(f'|chi_yyy|: {chi[0]:.3e} pm^2/V at 0, peak {chi.max():.3e} at {energy[chi.argmax()]:.2f} eV')
print(f'Re chi_yyy: minimum {re_gap:.3e} pm^2/V at {e_gap:.2f} eV')
for kind in ('shiftm', 'injectm', 'injectc'):  # odd under time reversal; circular injection
    maxima = [
        f'{c} {np.abs(spectrum(f"{kind}_{c}")[1]).max():.1e}' for c in ('yyy', 'yxx', 'xxy', 'xxx')
    ]
    print(f'max |{kind}|: ' + ', '.join(maxima))
inv_shift, inv_chi = spectrum('shift_yyy', 'nlo_inv')[1], spectrum('shga_yyy', 'nlo_inv')[1]
print(f'delta = 0: max |shift| {np.abs(inv_shift).max():.1e}, max |chi| {inv_chi.max():.1e}')
# [/checks]

# The numbers the page quotes, at the precision it quotes them.
assert abs(yyy.max() - 14.74) < 0.01, f'shift-current peak {yyy.max():.3f} nm muA/V^2'
assert abs(energy[yyy.argmax()] - 1.07) < 0.005, f'shift-current peak at {energy[yyy.argmax()]} eV'
assert symmetry < 1e-3 and np.abs(xxx).max() < 1e-10, f'D3h relations {symmetry:.1e}'
assert abs(chi.max() / 8.85e5 - 1) < 1e-3, f'two-photon SHG peak {chi.max():.4e} pm^2/V'
assert abs(energy[chi.argmax()] - 0.51) < 0.005, f'two-photon SHG peak at {energy[chi.argmax()]} eV'
assert abs(re_gap / -3.33e5 - 1) < 2e-3, f'Re chi at the gap {re_gap:.4e} pm^2/V'
assert np.abs(inv_shift).max() < 1e-6 and inv_chi.max() < 1e-4, 'responses with inversion'
