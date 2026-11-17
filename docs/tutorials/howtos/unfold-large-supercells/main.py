"""Code of the how-to "How to unfold the bands of large supercells with the sparse solver".

The page is docs/tutorials/howtos/unfold-large-supercells.md. The model is built in code, so the
script needs no input files. Run it with ``OMP_NUM_THREADS=1``. It writes the figure of the page
to the working directory and ends with asserts of the numbers the page quotes.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')

# [model]
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
# [/model]


# [cell]
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


# [/cell]


# [unfold]
pc = EDTBModel(carbon, aa_bilayer(1, 0)).to_model_dict()
sc = EDTBModel(carbon, aa_bilayer(4, 3)).to_model_dict()
hexagonal = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [2 / 3, 1 / 3, 0]}  # a1, a2 at 60 degrees
result = unfold_bands_sparse(
    pc, sc, sym_points=hexagonal, path_str='G-M-K-G',
    n_eigs=40, sigma=-0.5, nk_per_seg=20, n_workers=-1,
)  # fmt: skip
# [/unfold]

# [window]
bottom, top = result.E_sc.min(axis=1), result.E_sc.max(axis=1)  # window edges per k-point
print(f'window covers {bottom.max():.3f} to {top.min():.3f} eV at every k-point')
# [/window]

# Energy zero: the mean of the four primitive states at K nearest sigma (the two Dirac points).
ticks = [t[0] for t in result.sym_ticks]
iK = int(np.argmin(np.abs(result.kdist - ticks[2])))
eK = np.sort(result.E_pc[iK])
dirac = np.sort(eK[np.argsort(np.abs(eK + 0.5))[:4]])
zero = dirac.mean()
print(f'Dirac states at K {np.round(dirac, 4)} eV, E_D = {zero:.4f} eV')
print(f'window relative to E_D: {bottom.max() - zero:.3f} to {top.min() - zero:.3f} eV')

fig, ax = plt.subplots(figsize=(5.6, 4.6))
k = result.kdist
ax.plot(k, result.E_pc - zero, color='#004488', lw=0.9)
ax.plot(k, bottom - zero, color='#8A6800', lw=1.0, ls='--')
ax.plot(k, top - zero, color='#8A6800', lw=1.0, ls='--')
kk = np.repeat(k[:, None], result.E_sc.shape[1], axis=1)
shown = result.W > 0.02
ax.scatter(kk[shown], result.E_sc[shown] - zero, s=14 * result.W[shown], color='#D55E00',
           edgecolors='none', zorder=3)  # fmt: skip
ax.set_xticks(ticks, [r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
for t in ticks:
    ax.axvline(t, color='#000000', lw=0.6)
ax.set(xlim=(k[0], k[-1]), ylim=(-4, 4), ylabel=r'$E - E_\mathrm{D}$ (eV)')
fig.tight_layout()
fig.savefig('bilayer_window.png', dpi=150)
plt.close(fig)

# The numbers the page quotes, at the precision it quotes them.
nawf = 9 * len(sc['model']['atoms'])
dense_gb = nawf**2 * 16 * 125 / 1e9  # complex H(R) on the 5 x 5 x 5 grid of the dense route
print(f'(4, 3) cell: {nawf} orbitals, dense H(R) {dense_gb:.1f} GB')
assert result.N == 37 and nawf == 1332, f'N = {result.N}, {nawf} orbitals'
assert abs(zero + 0.519) < 1e-3, f'Dirac energy E_D {zero:.4f} eV'
split = (dirac[2:].mean() - dirac[:2].mean()) / 2
assert abs(split - 0.094) < 1e-3, f'AA Dirac points at E_D +- {split:.4f} eV'
window = (bottom.max() - zero, top.min() - zero)
assert np.allclose(window, (-1.71, 1.75), atol=5e-3), f'window edges {window} eV'

# The multiplet at the lower edge of the window at Gamma: returned states against its size.
from PAOFLOW.spectrum.sparse_bands import SparseEDTB

edge = result.E_sc[0].min()
kept = int(np.sum(np.abs(result.E_sc[0] - edge) < 1e-3))
wider = SparseEDTB(sc, verbose=False).eigvals_at_k(np.zeros(3), n_eigs=80, sigma=-0.5)
size = int(np.sum(np.abs(wider - edge) < 1e-3))
print(f'Gamma, lower edge at {edge - zero:.3f} eV: {kept} of {size} states of the multiplet')
assert (kept, size) == (4, 6), f'states of the edge multiplet at Gamma: {kept} of {size}'
