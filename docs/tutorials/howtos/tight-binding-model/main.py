"""Code of the how-to "How to run PAOFLOW on a tight-binding model" (tight-binding-model.md).

Every model is built in code, so the script needs no input files. It writes the figures of the
page to the working directory and ends with asserts of the numbers the page quotes. The PythTB
part needs the ``pythtb`` extra of PAOFLOW.
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
        plt.figure(num).savefig(name, dpi=150)
    plt.close('all')


# [kane-mele]
from PAOFLOW import PAOFLOW

model = {'label': 'Kane_Mele', 't': 1.0, 'soc_par': 0.1, 'r_par': 0.0, 'v_par': 0.0}
paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='kane_mele')

# k-path in crystal coordinates of the reciprocal vectors of the model; ibrav=4 is hexagonal
points = {'G': [0.0, 0.0, 0.0], 'M': [0.5, 0.0, 0.0], 'K': [2 / 3, 1 / 3, 0.0]}
paoflow.bands(ibrav=4, nk=100, band_path='G-M-K-G', high_sym_points=points)

paoflow.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=1)  # a 2D model: one k-point along z
paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing(min_width=0.02)  # floor: twice the 0.01 eV spacing of the DOS grid
paoflow.dos(emin=-3.5, emax=3.5, ne=701, do_pdos=False)
paoflow.finish_execution()
# [/kane-mele]

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
save('km_bands_dos.png')

# [graphene]
for delta in (0.0, 1.0):
    graphene = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': 2.7, 'delta': delta}, outputdir=f'graphene_{delta:g}'
    )
    graphene.bands(ibrav=4, nk=100, band_path='G-M-K-G', high_sym_points=points)
    graphene.finish_execution()

plotter.plot_bands(
    ['graphene_0/bands_0.dat', 'graphene_1/bands_0.dat'],
    sym_points='graphene_0/kpath_points.txt',
    title=r'Graphene, $t = 2.7$ eV',
    label=r'$E$ (eV)',
    labels=[r'$\delta = 0$', r'$\delta = 1$ eV'],
    cols=['#004488', '#D55E00'],
)
# [/graphene]
save('graphene_bands.png')

# [slater-koster]
onsite = {'s': -5.19278, 'px': 1.05825, 'py': 1.05825, 'pz': 1.05825}
silicon = {
    'label': 'Slater_Koster',
    'alat': 10.26,  # bohr
    'model': {
        'a_vectors': [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
        'atoms': {
            '0': {
                'name': 'Si',
                'tau': [0.0, 0.0, 0.0],
                'orbitals': ['s', 'px', 'py', 'pz'],
                **onsite,
            },
            '1': {
                'name': 'Si',
                'tau': [0.25, 0.25, 0.25],
                'orbitals': ['s', 'px', 'py', 'pz'],
                **onsite,
            },
        },
        'hoppings': {'sss': -2.36233, 'sps': 1.86401, 'pps': 2.85882, 'ppp': -0.94687},
    },
}
paoflow = PAOFLOW.PAOFLOW(model=silicon, outputdir='silicon')
fcc_points = {'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.5, 0.0], 'L': [0.5, 0.5, 0.5]}
paoflow.bands(ibrav=2, nk=500, band_path='L-G-X', high_sym_points=fcc_points)
paoflow.finish_execution()
# [/slater-koster]

plotter.plot_bands(
    'silicon/bands_0.dat',
    sym_points='silicon/kpath_points.txt',
    title=r'Si, nearest-neighbour $sp^3$ Slater–Koster model',
    label=r'$E$ (eV)',
    cols='#004488',
)
save('si_sk_bands.png')

# [pythtb]
from pythtb import Lattice, TBModel

lattice = Lattice(
    [[1, 0], [1 / 2, np.sqrt(3) / 2]], [[1 / 3, 1 / 3], [2 / 3, 2 / 3]], periodic_dirs=[0, 1]
)
haldane = TBModel(lattice=lattice, spinful=False)
haldane.set_onsite([0.0, 0.0])
for R in ([0, 0], [-1, 0], [0, -1]):  # first neighbours, A to B
    haldane.set_hop(-1.0, 0, 1, R)
t2 = 0.15 * np.exp(1j * np.pi / 2)  # second neighbours, phase phi = pi/2
for R in ([1, 0], [-1, 1], [0, -1]):
    haldane.set_hop(t2, 0, 0, R)
for R in ([-1, 0], [1, -1], [0, 1]):
    haldane.set_hop(t2, 1, 1, R)

paoflow = PAOFLOW.PAOFLOW(model=haldane, outputdir='haldane')
paoflow.bands(ibrav=4, nk=600, band_path='G-M-K-G', high_sym_points=points)
paoflow.finish_execution()
# [/pythtb]

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
