"""Figures and checks of the how-to "How to compute the band structure of twisted bilayer graphene"
(docs/tutorials/howtos/twisted-bilayer.md).

Run it after ``OMP_NUM_THREADS=1 python main.py > main.log``, in the same directory, or inside the
folder ``twisted-bilayer/`` of ``howto_assets.tar.gz`` of the release tutorial-assets-v2, which
holds the outputs of ``main.py`` and its ``main.log``. It reads the files ``main.py`` writes to
``tbg_<n>_<m>/`` and ``near_K_<n>_<m>/`` and the numbers it prints, draws the figures of the page
and asserts the numbers the page quotes.
"""

import json
import re

import matplotlib.pyplot as plt
import numpy as np

palette = ['#004488', '#D55E00', '#FFC20A', '#8A6800', '#000000']  # blue, vermillion, amber
log = open('main.log').read()


def printed(pattern: str) -> float:
    """The number that ``main.py`` printed in place of the group of ``pattern``."""
    return float(re.search(pattern, log).group(1))


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
geom = json.load(open('tbg_6_5/geometry.json'))
cell = np.array(geom['a_vectors'])[:2, :2] * a
xyz = np.array([atom['tau'] for atom in geom['atoms']]) * a
upper = xyz[:, 2] > 1e-6
tiles = [i * cell[0] + j * cell[1] for i in range(-1, 3) for j in range(-1, 3)]
fig, ax = plt.subplots(figsize=(6.2, 5.6))
for layer, col, label in (
    (~upper, palette[0], 'layer 1'),
    (upper, palette[1], f'layer 2, rotated {angle[6, 5]:.2f}°'),
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
fig.savefig('moire_6_5.png', dpi=150)

# the bands of the two cells, with the Dirac cone of one layer (slope hbar v0 from main.log)
v0 = printed(r'hbar v0 = (\S+) eV alat')
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.4), sharey=True)
E_D, edges = {}, {}
for ax, (n, m) in zip(axs, ((4, 3), (6, 5))):
    data = np.loadtxt(f'near_K_{n}_{m}/bands_0.dat')
    at_K = data[0, 1:][np.argsort(np.abs(data[0, 1:] + 0.52))[:4]]
    data = np.loadtxt(f'tbg_{n}_{m}/bands_0.dat')
    k, E = data[:, 0], data[:, 1:]
    ticks = np.loadtxt(f'tbg_{n}_{m}/kpath_points.txt', usecols=0)
    E_D[n, m] = E[0][np.argsort(np.abs(E[0] + 0.52))[:4]].mean()  # the four states at K
    for label, tick in (('G', ticks[1]), ('M', ticks[2])):
        e = E[np.argmin(np.abs(k - tick))] - E_D[n, m]
        edges[n, m, label] = (e[e > 1e-3].min(), e[e < -1e-3].max())
    print(
        f'({n}, {m}): E_D = {E_D[n, m]:.4f} eV, the four states at K within '
        f'{1e3 * np.ptp(at_K):.2f} meV'
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
fig.savefig('bands.png', dpi=150)

# the velocities of the two cells and the continuum model fitted by main.py
angles = np.array([angle[4, 3], angle[6, 5]])
ratio = np.array([printed(rf'\({n}, {m}\): .* v\*/v0 = (\S+)') for n, m in ((4, 3), (6, 5))])
alpha = printed(r'alpha\(\S+ deg\) = (\S+),')
predicted = printed(r'predicted v\*/v0 at \S+ deg = (\S+),')
magic = printed(r'magic angle (\S+) deg')
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
fig.savefig('velocity.png', dpi=150)

# the numbers the page quotes
aa = printed(r'd = 2.68 A: Dirac points (\S+) eV apart')
assert abs(aa - 0.684) < 1e-3, f'AA Dirac-point splitting at 2.68 A {aa:.4f} eV'
assert abs(ratio[0] - 0.858) < 1e-3, f'v*/v0 of the (4, 3) cell {ratio[0]:.4f}'
assert abs(ratio[1] - 0.734) < 1e-3, f'v*/v0 of the (6, 5) cell {ratio[1]:.4f}'
assert abs(predicted - 0.693) < 1e-3, f'continuum-model v*/v0 at 6.01 deg {predicted:.4f}'
assert abs(magic - 2.16) < 5e-3, f'angle where the continuum-model v* vanishes {magic:.3f} deg'
