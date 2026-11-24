import numpy as np

from PAOFLOW.models.band_unfold import unfold_bands_sparse
from PAOFLOW.models.edtb_params import EDTBModel

spd = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
carbon = {
    'edtb_version': '1.0',
    'basis': {'C': {'orbitals': spd, 'l_channels': ['s', 'p', 'd']}},
    'onsite': {'C': {'s': -4.52191, 'p': -0.84835, 't2g': 6.33815, 'eg': 6.34247}},
    'hoppings': {
        'C-C': {
            'type': 'distance_dependent', 'r_0': 2.684, 'r_c': 10.0, 'n_c': 2.59873,
            'channels': {
                'sss': {'V0': -4.28565, 'n': 4.25117}, 'sps': {'V0': 5.03970, 'n': 3.50549},
                'sds': {'V0': 2.96708, 'n': 1.15475}, 'pps': {'V0': 4.50301, 'n': 2.56064},
                'ppp': {'V0': -2.08356, 'n': 4.38254}, 'pds': {'V0': 1.93938, 'n': -0.31188},
                'pdp': {'V0': -6.42159, 'n': 1.33501}, 'dds': {'V0': -0.00009, 'n': -3.40900},
                'ddp': {'V0': 0.00040, 'n': 1.78124}, 'ddd': {'V0': -0.00166, 'n': -1.17807},
            },
        }
    },
    'screening': {
        'r_cut': 8.0,
        'gamma': {
            'C-C': {
                'ss': 0.00450, 'sp': 0.00743, 'sd': 0.60988, 'pp': 0.00670, 'pd': 1.70099,
                'dd': 1.53698,
            }
        },
    },
}  # fmt: skip


# The (n, m) supercell of AA bilayer graphene
def aa_bilayer(n, m, d=3.35, vacuum=15.0, a_cc=1.42):
    """AA bilayer graphene in the (n, m) supercell; lengths in angstrom, alat = graphene a."""
    a = a_cc * np.sqrt(3)
    a1, a2 = np.array([1.0, 0, 0]), np.array([0.5, np.sqrt(3) / 2, 0])
    cell = np.array([n * a1 + m * a2, -m * a1 + (n + m) * a2, [0, 0, (d + vacuum) / a]])
    span = range(-2 * (n + m), 2 * (n + m) + 1)
    frac = [np.linalg.solve(cell.T, i * a1 + j * a2) for i in span for j in span]
    cells = [f @ cell for f in frac if np.all((f[:2] > -1e-8) & (f[:2] < 1 - 1e-8))]
    basis = [np.array([0, 0, z]) + s for z in (0, d / a) for s in (np.zeros(3), (a1 + a2) / 3)]
    atoms = [{'species': 'C', 'tau': (t + b).tolist()} for b in basis for t in cells]
    return {'alat': a / 0.529177249, 'a_vectors': cell.tolist(), 'atoms': atoms}


# Unfolding with the sparse solver
pc = EDTBModel(carbon, aa_bilayer(1, 0)).to_model_dict()
supercell = EDTBModel(carbon, aa_bilayer(4, 3))
sc = supercell.to_model_dict()
hexagonal = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [2 / 3, 1 / 3, 0]}  # a1, a2 at 60 degrees
result = unfold_bands_sparse(
    pc, sc, sym_points=hexagonal, path_str='G-M-K-G',
    n_eigs=40, sigma=-0.5, nk_per_seg=20, n_workers=-1,
)  # fmt: skip

# The edges of the window
bottom, top = result.E_sc.min(axis=1), result.E_sc.max(axis=1)  # window edges per k-point
print(f'window covers {bottom.max():.3f} to {top.min():.3f} eV at every k-point')

# Saving the bands, the weights and the supercell model
np.savez(
    'bilayer_unfolded.npz', kdist=result.kdist, ticks=[t[0] for t in result.sym_ticks],
    N=result.N, E_pc=result.E_pc, E_sc=result.E_sc, W=result.W,
)  # fmt: skip
supercell.save('carbon.json', 'bilayer_4_3.json')
