"""Figures and checks of the how-to "How to unfold supercell bands onto the primitive Brillouin zone".

The page is docs/learn/howtos/band-unfolding.md. Run it after ``python main.py`` in the same
directory: it reads ``si_unfolded.npz``, writes the figures of the page to the working directory
and ends with asserts of the numbers the page quotes.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from scipy.spatial import Voronoi

run = np.load('si_unfolded.npz')
k, ticks, E_pc, E_sc, W = run['kdist'], run['ticks'], run['E_pc'], run['E_sc'], run['W']
E_imp, W_imp = run['E_imp'], run['W_imp']

# Energy zero of the figures: the valence-band maximum of the primitive cell at Gamma.
ig = int(np.argmin(np.linalg.norm(run['kpath'], axis=1)))
vbm = E_pc[ig, 3]  # four valence bands per two Si atoms


def draw_weights(ax, E, W, zero, scale=14.0):
    """Supercell states as markers at their energies, with area proportional to the weight."""
    kk = np.repeat(k[:, None], E.shape[1], axis=1)
    shown = W > 0.02  # leave out the states with negligible weight
    ax.scatter(
        kk[shown], E[shown] - zero, s=scale * W[shown], color='#D55E00', edgecolors='none', zorder=3
    )


# Figure 1: geometry of folding, drawn for one layer of graphene in the (2, 1) supercell.
def ws_cell(b):
    """Vertices of the Wigner-Seitz cell of the 2D lattice spanned by the rows of ``b``."""
    pts = np.array([i * b[0] + j * b[1] for i in range(-3, 4) for j in range(-3, 4)])
    vor = Voronoi(pts)
    poly = vor.vertices[vor.regions[vor.point_region[len(pts) // 2]]]
    return poly[np.argsort(np.arctan2(poly[:, 1], poly[:, 0]))]


def closed(poly):
    return np.vstack([poly, poly[:1]]).T


n, m, a = 2, 1, 2.46  # supercell indices and graphene lattice constant (angstrom)
a1, a2 = np.array([1.0, 0.0]) * a, np.array([0.5, np.sqrt(3) / 2]) * a
A = np.array([n * a1 + m * a2, -m * a1 + (n + m) * a2])  # supercell vectors
honeycomb = np.array(
    [i * a1 + j * a2 + s for i in range(-6, 8) for j in range(-4, 8) for s in (0, (a1 + a2) / 3)]
)
frac = [np.linalg.solve(A.T, i * a1 + j * a2) for i in range(-6, 7) for j in range(-6, 7)]
R_l = np.array([f @ A for f in frac if np.all((f > -1e-8) & (f < 1 - 1e-8))])
N_graphene = len(R_l)
theta = np.arccos((n * n + m * m + 4 * n * m) / (2 * (n * n + n * m + m * m)))
rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])  # by -theta

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))
ax = axs[0]
ax.scatter(*honeycomb.T, s=6, color='#56B4E9', zorder=1)
ax.plot(*closed(np.array([[0, 0], A[0], A[0] + A[1], A[1]])), color='#000000', lw=1.4)
cell = np.array([[0, 0], a1, a1 + a2, a2])
ax.fill(*cell.T, color='#FFC20A', alpha=0.5, lw=0)
ax.plot(*closed(cell), color='#8A6800', lw=1.0)
ax.scatter(*R_l.T, s=40, marker='s', color='#D55E00', zorder=3)
for i, r in enumerate(R_l):
    ax.annotate(
        rf'$\mathbf{{R}}_{i + 1}$', r, xytext=(4, 4), textcoords='offset points',
        fontsize=9, color='#D55E00',
    )  # fmt: skip
ax.set(xlim=(-4, 9.5), ylim=(-1.5, 9), xlabel='x (Å)', ylabel='y (Å)', aspect='equal')
ax.set_title(f'(a) ({n}, {m}) supercell, N = {N_graphene}', fontsize=11)

ax = axs[1]
b = np.linalg.inv(np.array([a1, a2]) / a).T  # reciprocal vectors in units of 2 pi / a
B = np.linalg.inv(A / a).T
bz, sbz = ws_cell(b), ws_cell(B)
ax.fill(*closed(bz), color='#004488', alpha=0.08)
ax.plot(*closed(bz), color='#004488', lw=1.4)
G = np.array([i * B[0] + j * B[1] for i in range(-6, 7) for j in range(-6, 7)])
G = G[Path(bz * 1.0001).contains_points(G)]  # supercell reciprocal vectors in the primitive zone
for g in G:
    ax.plot(*closed(sbz + g), color='#000000', lw=0.6, alpha=0.5)
ax.plot(*closed(sbz), color='#000000', lw=1.4)
ax.scatter(*(G + 0.35 * sbz[0]).T, s=30, color='#D55E00', zorder=3)
ax.annotate(
    'primitive zone', bz[1], xytext=(-10, 8), textcoords='offset points',
    color='#004488', fontsize=9,
)  # fmt: skip
ax.set(aspect='equal', xlabel=r'$k_x$ ($2\pi/a$)', ylabel=r'$k_y$ ($2\pi/a$)')
ax.set_title(r'(b) $N$ wavevectors $\mathbf{k}$ that fold onto one point', fontsize=11)

ax = axs[2]
layer2 = honeycomb @ rot.T
ax.scatter(*honeycomb.T, s=28, facecolors='none', edgecolors='#004488', lw=1.0,
           label='layer 1 (primitive sites)')  # fmt: skip
ax.scatter(*layer2.T, s=10, color='#D55E00', label=rf'layer 2, rotated {np.degrees(theta):.2f}°')
ax.plot(*closed(np.array([[0, 0], A[0], A[0] + A[1], A[1]])), color='#000000', lw=1.4)
ax.set(xlim=(-4, 9.5), ylim=(-1.5, 9), xlabel='x (Å)', aspect='equal')
ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
ax.set_title(f'(c) twisted ({n}, {m}) cell', fontsize=11)
fig.tight_layout()
fig.savefig('folding_geometry.png', dpi=150)

# Figure 2: perfect and impurity supercell, full path.
fig, axs = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
axs[0].plot(k, E_sc - vbm, color='#56B4E9', lw=0.8)
axs[1].plot(k, E_pc - vbm, color='#004488', lw=0.9)
for ax, (E, w) in ((axs[0], (E_sc, W)), (axs[1], (E_imp, W_imp))):
    draw_weights(ax, E, w, vbm)
    ax.set_xticks(ticks, ['L', r'$\Gamma$', 'X'])
    for t in ticks:
        ax.axvline(t, color='#000000', lw=0.6)
    ax.set_xlim(k[0], k[-1])
axs[0].set(ylim=(-12.5, 4), ylabel=r'$E - E_\mathrm{VBM}$ (eV)')
axs[0].set_title('8-atom cell, perfect', fontsize=11)
axs[1].set_title('8-atom cell, +1 eV on one atom', fontsize=11)
fig.tight_layout()
fig.savefig('si_unfolded.png', dpi=150)

# Figure 3: zoom on the avoided crossing along L-Gamma and on the split multiplet at X.
E1 = E_imp - vbm
crossing, ix = (36, 53), len(k) - 1  # path indices of the zoom on L-Gamma, and of X


def zoom(ax, lo, hi, ylim):
    sl = slice(lo, hi)
    ax.plot(k[sl], E_sc[sl] - vbm, color='#56B4E9', lw=0.8)
    ax.plot(k[sl], E_pc[sl] - vbm, color='#004488', lw=1.0, ls='--')
    draw_weights(ax, E_imp, W_imp, vbm, scale=60.0)
    ax.set_xlim(k[lo], k[hi - 1])
    ax.set_ylim(*ylim)


def bright(ik, lo=-4.5, hi=-2.7):
    """Impurity-cell states at path index ``ik`` with weight above 0.1 in the zoom window."""
    s = (E1[ik] > lo) & (E1[ik] < hi) & (W_imp[ik] > 0.1)
    return E1[ik][s], W_imp[ik][s]


fig, axs = plt.subplots(1, 2, figsize=(9, 4.0))
zoom(axs[0], *crossing, (-4.5, -2.7))
for ik, dy in ((42, 1), (44, 1), (46, -1)):
    for e, w in zip(*bright(ik)):
        axs[0].annotate(f'{w:.2f}', (k[ik], e), xytext=(5, 6 * dy), textcoords='offset points',
                        fontsize=8)  # fmt: skip
axs[0].set_xticks([])
axs[0].set(xlabel=r'halfway along L–$\Gamma$', ylabel=r'$E - E_\mathrm{VBM}$ (eV)')
axs[0].set_title('(a) avoided crossing', fontsize=11)
zoom(axs[1], ix - 20, ix + 1, (-3.2, -2.4))
x_states = sorted({round(e, 3) for e in E1[ix][(E1[ix] > -3.2) & (E1[ix] < -2.4)]})
x_weights = [W_imp[ix][np.abs(E1[ix] - e) < 1e-3].sum() for e in x_states]
for e, w in zip(x_states, x_weights):
    axs[1].annotate(f'{w:.2f}', (k[ix], e), xytext=(-30, 4), textcoords='offset points', fontsize=8)
axs[1].set_xticks([k[ix]], ['X'])
axs[1].set_title('(b) split multiplet at X', fontsize=11)
fig.tight_layout()
fig.savefig('si_impurity_zoom.png', dpi=150)

# The numbers the page quotes, at the precision it quotes them.
deviation = np.abs(W.sum(axis=1) - E_pc.shape[1]).max()  # sum rule: nawf of the primitive cell
assert deviation < 1e-12, f'sum-rule deviation {deviation:.1e}'

unfolded = np.sort(E_sc[ig][W[ig] > 0.5])
match = np.abs(unfolded - E_pc[ig]).max()
assert match < 1e-12, f'perfect cell: weight-1 states vs primitive bands at Gamma {match:.1e} eV'

shift_vbm = E_imp[ig, 15] - vbm  # 16 valence bands in the 8-atom cell
shift_low = E_imp[ig, 0] - E_pc[ig, 0]
print(f'impurity: VBM shift {shift_vbm:.4f} eV, lowest band at Gamma {shift_low:.4f} eV')
assert abs(shift_vbm - 0.118) < 1e-3, f'impurity VBM shift {shift_vbm:.4f} eV'
assert abs(shift_low - 0.109) < 1e-3, f'impurity shift of the lowest band {shift_low:.4f} eV'

gap = np.diff(np.sort(bright(44)[0]))
print(f'avoided crossing at path index 44: states {np.sort(bright(44)[0])}, gap {gap}')
assert gap.size == 1 and abs(gap[0] - 0.24) < 5e-3, f'avoided-crossing gap {gap} eV'

print(f'X: states {x_states} eV, weights {np.round(x_weights, 3)}')
assert np.allclose(x_states, [-2.90, -2.68], atol=5e-3), f'split states at X {x_states} eV'
assert np.allclose(x_weights, 1.0, atol=5e-3), f'weights of the split states at X {x_weights}'
