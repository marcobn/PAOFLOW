"""Schematic of the unfolding weights for the theory page band-unfolding.md.

A one-dimensional chain with one orbital per site and hopping t to its neighbours is described
in a supercell of N = 3 sites. At every primitive wavevector k the supercell Hamiltonian is
built in the lattice gauge, H(k) = sum_T H(T) exp(i k T), and the weight of each eigenstate is
w_n(k) = (1/N) |sum_l exp(-i k R_l) C_l^n(k)|^2, the form the page derives. Panel (a) is the
perfect chain, panel (b) the chain with the on-site energy of one site in three raised by V.

Run from anywhere: python docs/learn/theory/band-unfolding/figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[3] / '_static' / 'images' / 'theory' / 'band-unfolding'
BLUE, VERMILLION, SKY = '#004488', '#D55E00', '#56B4E9'

N, t, V = 3, -1.0, 1.0  # sites per supercell, hopping, impurity potential (units of |t|)
k = np.linspace(-np.pi, np.pi, 241)  # primitive wavevector, k a in radians
R = np.arange(N)  # translations inside the supercell, units of a


def unfold(onsite):
    """Supercell eigenvalues and weights at every primitive k."""
    E, W = np.zeros((k.size, N)), np.zeros((k.size, N))
    for i, kk in enumerate(k):
        H = np.diag(onsite).astype(complex) + t * (np.eye(N, k=1) + np.eye(N, k=-1))
        H[N - 1, 0] += t * np.exp(1j * kk * N)  # bond from the last site to the next supercell
        H[0, N - 1] += t * np.exp(-1j * kk * N)
        E[i], C = np.linalg.eigh(H)
        W[i] = np.abs(np.exp(-1j * kk * R) @ C) ** 2 / N
    return E, W


perfect, impurity = unfold(np.zeros(N)), unfold(np.r_[V, np.zeros(N - 1)])
assert np.allclose(perfect[1].sum(axis=1), 1.0) and np.allclose(impurity[1].sum(axis=1), 1.0)
bright = perfect[0][perfect[1] > 0.5].reshape(k.size)  # one state of weight 1 per k
assert np.allclose(
    bright, 2 * t * np.cos(k)
), 'perfect chain: weight-1 states on the primitive band'

fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
for ax, (E, W), title in zip(
    axs, (perfect, impurity), ('(a) perfect chain', f'(b) one site in {N} shifted by $V = |t|$')
):
    ax.plot(k, E, color=SKY, lw=0.8)
    ax.plot(k, 2 * t * np.cos(k), color=BLUE, lw=1.0, ls='--')
    kk = np.repeat(k[:, None], N, axis=1)
    show = W > 0.01
    ax.scatter(
        kk[show][::2], E[show][::2], s=14 * W[show][::2], color=VERMILLION, edgecolors='none'
    )
    for x in (-np.pi / N, np.pi / N):
        ax.axvline(x, color='#000000', lw=0.6, ls=':')
    ax.set_xticks([-np.pi, -np.pi / N, 0, np.pi / N, np.pi])
    ax.set_xticklabels([r'$-\pi/a$', r'$-\pi/3a$', '0', r'$\pi/3a$', r'$\pi/a$'])
    ax.set(xlim=(-np.pi, np.pi), xlabel=r'primitive wavevector $k$', title=title)
axs[0].set_ylabel(r'$E$ (units of $|t|$)')
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'chain_weights.png', dpi=150)
