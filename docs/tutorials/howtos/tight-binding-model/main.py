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

# Two parameter sets of graphene
from PAOFLOW import GPAO

for delta in (0.0, 1.0):
    graphene = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': 2.7, 'delta': delta}, outputdir=f'graphene_{delta:g}'
    )
    graphene.bands(ibrav=4, nk=100, band_path='G-M-K-G', high_sym_points=points)
    graphene.finish_execution()

plotter = GPAO.GPAO()
plotter.plot_bands(
    ['graphene_0/bands_0.dat', 'graphene_1/bands_0.dat'],
    sym_points='graphene_0/kpath_points.txt',
    title=r'Graphene, $t = 2.7$ eV',
    label=r'$E$ (eV)',
    labels=[r'$\delta = 0$', r'$\delta = 1$ eV'],
    cols=['#004488', '#D55E00'],
)

# A Slater-Koster model of silicon
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

# The Haldane model built with PythTB
import numpy as np
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
