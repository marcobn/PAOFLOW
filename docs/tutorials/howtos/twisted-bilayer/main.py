"""Code of the how-to "How to compute the band structure of twisted bilayer graphene"
(docs/tutorials/howtos/twisted-bilayer.md).

Run it as ``OMP_NUM_THREADS=1 python main.py``. The bands of the (4, 3) and (6, 5) cells take
about 6 minutes on 4 cores; the script writes them to ``twisted-bilayer/`` and skips them when
that folder already holds them, as it does after you extract ``howto_assets.tar.gz`` of the
release tutorial-assets-v1 in the working directory. It writes the figures of the page to the
working directory and ends with asserts of the numbers the page quotes.
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from PAOFLOW.models.edtb_params import EDTBModel
from PAOFLOW.spectrum.sparse_bands import SparseEDTB

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')
palette = ['#004488', '#D55E00', '#FFC20A', '#8A6800', '#000000']  # blue, vermillion, amber


def save(*names: str) -> None:
    """Write the last ``len(names)`` figures, in the order they were drawn, and close all."""
    for num, name in zip(plt.get_fignums()[-len(names) :], names):
        plt.figure(num).savefig(name, dpi=150)
    plt.close('all')


# EDTB model of carbon with distance-dependent hoppings, as on the page on large supercells
spd = ['s', 'px', 'py', 'pz', 'dxy', 'dyz', 'dzx', 'dx2-y2', 'dz2']
carbon = {
    'edtb_version': '1.0',
    'basis': {'C': {'orbitals': spd, 'l_channels': ['s', 'p', 'd']}},
    'onsite': {'C': {'s': -4.52191, 'p': -0.84835, 't2g': 6.33815, 'eg': 6.34247}},
    'hoppings': {'C-C': {
        'type': 'distance_dependent', 'r_0': 2.684, 'r_c': 10.0, 'n_c': 2.59873,
        'channels': {
            'sss': {'V0': -4.28565, 'n': 4.25117}, 'sps': {'V0': 5.03970, 'n': 3.50549},
            'sds': {'V0': 2.96708, 'n': 1.15475}, 'pps': {'V0': 4.50301, 'n': 2.56064},
            'ppp': {'V0': -2.08356, 'n': 4.38254}, 'pds': {'V0': 1.93938, 'n': -0.31188},
            'pdp': {'V0': -6.42159, 'n': 1.33501}, 'dds': {'V0': -0.00009, 'n': -3.40900},
            'ddp': {'V0': 0.00040, 'n': 1.78124}, 'ddd': {'V0': -0.00166, 'n': -1.17807}}}},
    'screening': {'r_cut': 8.0, 'gamma': {'C-C': {
        'ss': 0.00450, 'sp': 0.00743, 'sd': 0.60988, 'pp': 0.00670, 'pd': 1.70099, 'dd': 1.53698}}},
}  # fmt: skip


# [cell]
def twisted_bilayer(n, m, d=2.68, vacuum=15.0, a_cc=1.42):
    """Commensurate twisted bilayer graphene (n, m); lengths in angstrom, alat = graphene a."""
    a = a_cc * np.sqrt(3)
    a1, a2 = np.array([1.0, 0]), np.array([0.5, np.sqrt(3) / 2])
    theta = np.arccos((n * n + m * m + 4 * n * m) / (2 * (n * n + n * m + m * m)))
    c, s = np.cos(theta), np.sin(theta)
    cell = np.array([n * a1 + m * a2, -m * a1 + (n + m) * a2])
    span = range(-2 * (n + m), 2 * (n + m) + 1)
    atoms = []
    for z, rot in ((0.0, np.eye(2)), (d / a, np.array([[c, s], [-s, c]]))):
        sites = {}
        for i in span:
            for j in span:
                for t in (0, 1 / 3):  # sublattices A and B
                    f = np.linalg.solve(cell.T, rot @ (i * a1 + j * a2 + t * (a1 + a2)))
                    f -= np.floor(f + 1e-8)
                    sites[tuple(np.round(f, 6))] = f @ cell
        atoms += [{'species': 'C', 'tau': [x, y, z]} for x, y in sites.values()]
    a_vectors = [[*cell[0], 0], [*cell[1], 0], [0, 0, (d + vacuum) / a]]
    return {'alat': a / 0.529177249, 'a_vectors': a_vectors, 'atoms': atoms}, np.degrees(theta)


# [/cell]


def moire_bands(n, m):
    """Bands of the (n, m) cell along K-G-M-K and near K, written to twisted-bilayer/."""
    # [bands]
    geom, _ = twisted_bilayer(n, m)
    bands = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
    moire = {'G': [0, 0, 0], 'M': [0.5, 0, 0], 'K': [2 / 3, 1 / 3, 0]}
    bands.compute_bands(
        'K-G-M-K',
        moire,
        nk=60,
        n_eigs=40,
        sigma=-0.52,
        outputdir=f'twisted-bilayer/tbg_{n}_{m}',
        n_workers=-1,
    )
    # [/bands]
    # [near-K]
    moire['P'] = [0.64, 0.32, 0]  # 4 % of the way from K to G
    bands.compute_bands(
        'K-P',
        moire,
        nk=8,
        n_eigs=8,
        sigma=-0.52,
        outputdir=f'twisted-bilayer/near_K_{n}_{m}',
        n_workers=-1,
    )
    # [/near-K]


for n, m in ((4, 3), (6, 5)):
    if not Path(f'twisted-bilayer/near_K_{n}_{m}/bands_0.dat').exists():  # in the release asset
        moire_bands(n, m)

# the cells of the table on the page
a = 1.42 * np.sqrt(3)  # graphene lattice constant (angstrom)
angle = {}
print('(n, m)   theta (deg)   atoms   orbitals   L (A)')
for n, m in ((2, 1), (4, 3), (6, 5), (10, 9), (15, 14), (17, 16), (20, 19), (31, 30)):
    cells = n * n + n * m + m * m
    angle[n, m] = np.degrees(np.arccos((n * n + m * m + 4 * n * m) / (2 * cells)))
    print(
        f'({n}, {m})  {angle[n, m]:8.2f}  {4 * cells:8d}  {36 * cells:8d}  {a * np.sqrt(cells):8.1f}'
    )

# top view of the (6, 5) cell
geom, theta = twisted_bilayer(6, 5)
cell = np.array(geom['a_vectors'])[:2, :2] * a
xyz = np.array([atom['tau'] for atom in geom['atoms']]) * a
upper = xyz[:, 2] > 1e-6
tiles = [i * cell[0] + j * cell[1] for i in range(-1, 3) for j in range(-1, 3)]
fig, ax = plt.subplots(figsize=(6.2, 5.6))
for layer, col, label in (
    (~upper, palette[0], 'layer 1'),
    (upper, palette[1], f'layer 2, rotated {theta:.2f}°'),
):
    sites = np.vstack([xyz[layer, :2] + t for t in tiles])
    ax.scatter(*sites.T, s=6, color=col, label=label)
corners = np.array([[0, 0], cell[0], cell[0] + cell[1], cell[1], [0, 0]])
ax.plot(*corners.T, color=palette[4], lw=1.4)
for corner in corners[:4]:
    ax.annotate(
        'AA',
        corner,
        xytext=(-8, 6),
        textcoords='offset points',
        fontsize=10,
        bbox=dict(fc='white', ec='none', alpha=0.8, pad=1),
    )
ax.set_aspect('equal')
ax.set_xlim(-6, 30)
ax.set_ylim(-4, 38)
ax.set_xlabel('x (Å)')
ax.set_ylabel('y (Å)')
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),
    ncol=2,
    fontsize=9,
    markerscale=4,
    frameon=False,
)
fig.tight_layout()
save('moire_6_5.png')

# Dirac points of the AA bilayer at K: split by the interlayer hopping
aa = {}
for d in (3.35, 2.68):
    geom, _ = twisted_bilayer(1, 0, d=d)
    lower = [atom for atom in geom['atoms'] if atom['tau'][2] == 0]
    geom['atoms'] = lower + [
        {'species': 'C', 'tau': [x, y, d / a]} for x, y, _ in (atom['tau'] for atom in lower)
    ]
    aa_bands = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
    e = aa_bands.eigvals_at_k([2 / 3, 1 / 3, 0], n_eigs=4, sigma=-0.52)
    aa[d] = (e[2:].mean() - e[:2].mean(), e.mean())
    print(f'AA bilayer, d = {d} A: Dirac points {aa[d][0]:.3f} eV apart, mean {aa[d][1]:.4f} eV')


# [fit]
def dirac_slope(k, E, n_dirac=4):
    """Slope (eV alat) of the lower of the bands that leave the n_dirac states at K upwards."""
    near = np.array([np.sort(e[np.argsort(np.abs(e + 0.52))[:n_dirac]]) for e in E])
    return np.polyfit(k[2:6], near[2:6, n_dirac // 2], 1)[0]  # skips the first point after K


# [/fit]


# one layer, on a segment 0.4 % of K-G long, so that |k - K| is as small as in the moire cells
geom, _ = twisted_bilayer(1, 0)
geom['atoms'] = [atom for atom in geom['atoms'] if atom['tau'][2] == 0]
layer = SparseEDTB(EDTBModel(carbon, geom).to_model_dict(), verbose=False)
points = {'K': [2 / 3, 1 / 3, 0], 'P': [0.996 * 2 / 3, 0.996 / 3, 0]}
near_K = layer.compute_bands('K-P', points, nk=8, n_eigs=8, sigma=-0.52)
v0 = dirac_slope(near_K['k_dist'], near_K['eigenvalues'], n_dirac=2)
hbar = 6.582119569e-16  # eV s
print(
    f'one layer: hbar v0 = {v0:.3f} eV alat = {v0 * a:.3f} eV A, v0 = {v0 * a * 1e-10 / hbar:.3g} m/s'
)

# the bands of the two cells, from twisted-bilayer/
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
v, E_D, edges = {}, {}, {}
for ax, (n, m) in zip(axs, ((4, 3), (6, 5))):
    data = np.loadtxt(f'twisted-bilayer/near_K_{n}_{m}/bands_0.dat')
    v[n, m] = dirac_slope(data[:, 0], data[:, 1:])
    at_K = data[0, 1:][np.argsort(np.abs(data[0, 1:] + 0.52))[:4]]
    data = np.loadtxt(f'twisted-bilayer/tbg_{n}_{m}/bands_0.dat')
    k, E = data[:, 0], data[:, 1:]
    ticks = np.loadtxt(f'twisted-bilayer/tbg_{n}_{m}/kpath_points.txt', usecols=0)
    E_D[n, m] = E[0][np.argsort(np.abs(E[0] + 0.52))[:4]].mean()  # the four states at K
    for label, tick in (('G', ticks[1]), ('M', ticks[2])):
        e = E[np.argmin(np.abs(k - tick))] - E_D[n, m]
        edges[n, m, label] = (e[e > 1e-3].min(), e[e < -1e-3].max())
    print(
        f'({n}, {m}): E_D = {E_D[n, m]:.4f} eV, the four states at K within '
        f'{1e3 * np.ptp(at_K):.2f} meV, hbar v* = {v[n, m]:.3f} eV alat, v*/v0 = {v[n, m] / v0:.3f}'
    )
    print(
        f'  window at every k-point: E_D {E.min(1).max() - E_D[n, m]:+.3f} eV to '
        f'E_D {E.max(1).min() - E_D[n, m]:+.3f} eV'
    )
    print(
        '  bands above / below E_D at G and M (eV):',
        ', '.join(
            f'{lab} {up:.3f} / {down:.3f}'
            for (nn, mm, lab), (up, down) in edges.items()
            if (nn, mm) == (n, m)
        ),
    )
    ax.scatter(
        np.repeat(k, E.shape[1]),
        (E - E_D[n, m]).ravel(),
        s=5,
        color=palette[0],
        edgecolors='none',
        label='twisted bilayer',
    )
    for K in (ticks[0], ticks[-1]):  # the Dirac cone of one layer, from both K points
        dk = np.linspace(-0.25, 0.25, 51)
        for sign in (1, -1):
            ax.plot(
                K + dk,
                sign * v0 * np.abs(dk),
                color=palette[1],
                lw=1.0,
                ls='--',
                label='one layer' if (K, sign) == (ticks[0], 1) else None,
            )
    for tick in ticks:
        ax.axvline(tick, color=palette[4], lw=0.6)
    ax.set_xticks(ticks, ['K', r'$\Gamma$', 'M', 'K'])
    ax.set_xlim(k[0], k[-1])
    ax.set_ylim(-0.6, 0.6)
    ax.set_title(f'({n}, {m}), $\\theta$ = {angle[n, m]:.2f}°', fontsize=11)
axs[0].set_ylabel(r'$E - E_\mathrm{D}$ (eV)')
axs[0].legend(fontsize=9, loc='lower left')
fig.tight_layout()
save('bands.png')

# continuum model of Bistritzer and MacDonald, alpha fitted to the (4, 3) cell, alpha ~ 1/theta
angles = np.array([angle[4, 3], angle[6, 5]])
ratio = np.array([v[4, 3], v[6, 5]]) / v0
alpha = np.sqrt((1 - ratio[0]) / (3 + 6 * ratio[0]))
predicted = (1 - 3 * (alpha * angles[0] / angles[1]) ** 2) / (
    1 + 6 * (alpha * angles[0] / angles[1]) ** 2
)
magic = angles[0] * alpha * np.sqrt(3)
magic_65 = angles[1] * np.sqrt(3 * (1 - ratio[1]) / (3 + 6 * ratio[1]))  # fitted at 6.01 instead
print(
    f'alpha({angles[0]:.2f} deg) = {alpha:.3f}, predicted v*/v0 at {angles[1]:.2f} deg = {predicted:.3f}, '
    f'magic angle {magic:.2f} deg ({magic_65:.2f} deg if fitted at {angles[1]:.2f} deg)'
)
theta = np.linspace(1.6, 10.5, 300)
fig, ax = plt.subplots(figsize=(5.4, 3.8))
x = alpha * angles[0] / theta
ax.plot(
    theta,
    (1 - 3 * x**2) / (1 + 6 * x**2),
    color=palette[0],
    lw=1.2,
    label=f'continuum model, fitted at {angles[0]:.2f}°',
)
ax.scatter(
    angles, ratio, s=40, color=palette[1], zorder=3, label='tight-binding cells (4, 3), (6, 5)'
)
ax.axhline(0, color=palette[4], lw=0.6)
ax.axvline(magic, color=palette[3], lw=1.0, ls='--')
ax.annotate(
    f'{magic:.2f}°', (magic, 0.85), xytext=(5, 0), textcoords='offset points', color=palette[3]
)
ax.set_xlabel(r'twist angle $\theta$ (deg)')
ax.set_ylabel(r'$v^*/v_0$')
ax.set_xlim(1.6, 10.5)
ax.set_ylim(-0.3, 1.0)
ax.legend(fontsize=8, loc='lower right')
fig.tight_layout()
save('velocity.png')

# the numbers the page quotes
assert abs(aa[2.68][0] - 0.684) < 1e-3, f'AA Dirac-point splitting at 2.68 A {aa[2.68][0]:.4f} eV'
assert abs(ratio[0] - 0.858) < 1e-3, f'v*/v0 of the (4, 3) cell {ratio[0]:.4f}'
assert abs(ratio[1] - 0.734) < 1e-3, f'v*/v0 of the (6, 5) cell {ratio[1]:.4f}'
assert abs(predicted - 0.693) < 1e-3, f'continuum-model v*/v0 at 6.01 deg {predicted:.4f}'
assert abs(magic - 2.16) < 5e-3, f'angle where the continuum-model v* vanishes {magic:.3f} deg'
