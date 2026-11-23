import numpy as np

from PAOFLOW.models.band_unfold import unfold_bands
from PAOFLOW.models.edtb_params import EDTBModel

spd = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
si_params = {
    'edtb_version': '1.0',
    'basis': {'Si': {'orbitals': spd, 'l_channels': ['s', 'p', 'd']}},
    'onsite': {'Si': {'s': -3.76717, 'p': 1.88404, 't2g': 5.55238, 'eg': 5.49235}},
    'hoppings': {
        'Si-Si': [  # one entry per neighbour shell, r_ref = shell radius (bohr)
            {
                'r_ref': 4.58968,
                'params': {
                    'sss': -1.43863, 'sps': -2.19284, 'sds': 1.39344, 'pps': 2.45055,
                    'ppp': -0.71398, 'pds': -0.18045, 'pdp': 0.91304, 'dds': 0.14564,
                    'ddp': 0.32818, 'ddd': -0.20873,
                },
            },
            {
                'r_ref': 7.494997,
                'params': {
                    'sss': -0.09207, 'sps': -0.19994, 'sds': 0.25732, 'pps': 0.14479,
                    'ppp': -0.03920, 'pds': -0.58344, 'pdp': 0.05952, 'dds': -0.21677,
                    'ddp': 0.22660, 'ddd': -0.03560,
                },
            },
            {
                'r_ref': 8.788663,
                'params': {
                    'sss': -0.10306, 'sps': -0.02571, 'sds': 0.11758, 'pps': -0.10026,
                    'ppp': 0.06022, 'pds': -0.10671, 'pdp': 0.04463, 'dds': -0.22592,
                    'ddp': 0.08134, 'ddd': 0.00171,
                },
            },
        ]
    },
    'screening': {
        'r_cut': 8.0,
        'gamma': {
            'Si-Si': {
                'ss': 0.02701, 'sp': 0.00661, 'sd': 0.02200, 'pp': -0.01298, 'pd': 0.01929,
                'dd': 0.06764,
            }
        },
    },
}  # fmt: skip

# The primitive cell and the conventional cube of silicon
alat = 10.599478  # bohr
fcc = {
    'alat': alat,
    'a_vectors': [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]],
    'atoms': [
        {'species': 'Si', 'tau': [0.0, 0.0, 0.0]},
        {'species': 'Si', 'tau': [0.25, 0.25, 0.25]},
    ],
}
fcc_sites = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
cube = {
    'alat': alat,
    'a_vectors': np.eye(3).tolist(),
    'atoms': [
        {'species': 'Si', 'tau': list(np.add(s, b))}
        for b in ([0, 0, 0], [0.25, 0.25, 0.25])
        for s in fcc_sites
    ],
}

# Unfolding the perfect supercell
pc = EDTBModel(si_params, fcc).to_model_dict()
sc = EDTBModel(si_params, cube).to_model_dict()
points = {'G': [0, 0, 0], 'X': [0.5, 0.0, 0.5], 'L': [0.5, 0.5, 0.5]}
result = unfold_bands(pc, sc, sym_points=points, path_str='L-G-X', nk_per_seg=80)

# A substitutional impurity
impurity = EDTBModel(si_params, cube).to_model_dict()
site = impurity['model']['atoms']['0']
for orbital in site['orbitals']:
    site[orbital] += 1.0  # eV, on all nine orbitals of atom 0
result_imp = unfold_bands(pc, impurity, sym_points=points, path_str='L-G-X', nk_per_seg=80)

# Saving the bands and weights
np.savez(
    'si_unfolded.npz', kdist=result.kdist, kpath=result.kpath_cart,
    ticks=[t[0] for t in result.sym_ticks], E_pc=result.E_pc, E_sc=result.E_sc, W=result.W,
    E_imp=result_imp.E_sc, W_imp=result_imp.W,
)  # fmt: skip
