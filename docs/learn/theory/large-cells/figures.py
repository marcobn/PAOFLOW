"""Schematic of band folding for the theory page large-cells.md.

A one-dimensional chain with two orbitals per cell (on-site energies +D and -D, hopping t inside
the cell and t2 to the next cell) has two bands separated by a gap. The supercell of N cells is
built from the hoppings of the original cell by the doubling rule of the page,
H^(N)_{(a s),(b s')}(M) = H_ab(N M + s' - s), and its eigenvalues at K are checked against the
original bands at K + G_j. Each panel shades the states below an energy in the gap.

Run from anywhere: python docs/learn/theory/large-cells/figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[3] / '_static' / 'images' / 'theory' / 'large-cells'
BLUE, VERMILLION, AMBER, AMBER_TEXT = '#004488', '#D55E00', '#FFC20A', '#8A6800'

D, t, t2, top = 1.0, 1.0, 0.6, 0.4  # on-site, hopping in the cell, to the next cell, window top
H = {0: np.array([[D, t], [t, -D]]), 1: np.array([[0.0, 0.0], [t2, 0.0]])}
H[-1] = H[1].T  # H(-R) = H(R)^dagger


def bands(k):
    """Eigenvalues of the original cell at k (k a in radians)."""
    return np.linalg.eigvalsh(sum(np.exp(1j * k * R) * h for R, h in H.items()))


def supercell_bands(K, N):
    """Eigenvalues of the N-cell supercell at K (K a in radians), from the doubling rule."""
    HK = np.zeros((2 * N, 2 * N), dtype=complex)
    for M in (-1, 0, 1):
        for s in range(N):
            for sp in range(N):
                R = N * M + sp - s
                if R in H:
                    HK[2 * s : 2 * s + 2, 2 * sp : 2 * sp + 2] += np.exp(1j * K * N * M) * H[R]
    return np.linalg.eigvalsh(HK)


fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4), sharey=True)
for ax, N in zip(axes, (1, 2, 4)):
    K = np.linspace(-np.pi / N, np.pi / N, 201)
    for j in range(N):
        branch = np.array([bands(k + 2 * np.pi * j / N) for k in K])
        color = BLUE if j == 0 else VERMILLION
        label = r'$\varepsilon_{n,K}$' if j == 0 else r'$\varepsilon_{n,K+G_j}$'
        ax.plot(K / np.pi, branch[:, 0], color=color, lw=1.5, label=label if j < 2 else None)
        ax.plot(K / np.pi, branch[:, 1], color=color, lw=1.5)
    for Ki in K[::20]:  # the supercell spectrum is the union of the folded branches
        folded = np.sort([e for j in range(N) for e in bands(Ki + 2 * np.pi * j / N)])
        assert np.allclose(supercell_bands(Ki, N), folded, atol=1e-12), (N, Ki)
    ax.axhspan(-2.2, top, color=AMBER, alpha=0.25, lw=0)
    ax.axhline(top, color=AMBER_TEXT, lw=1.0, ls='--')
    ax.text(0, top + 0.12, f'solved: {N} of {2 * N}', color=AMBER_TEXT, ha='center', fontsize=9)
    ax.set_xlim(-1 / N, 1 / N)
    ax.set_xticks([-1 / N, 0, 1 / N])
    ax.set_xticklabels([rf'$-1/{N}$' if N > 1 else '$-1$', '0', rf'$1/{N}$' if N > 1 else '1'])
    ax.set_xlabel(r'$K$ ($\pi/a$)')
    ax.set_title(f'{N} cell' + ('s' if N > 1 else ''), fontsize=10)
axes[0].set_ylim(-2.2, 2.2)
axes[0].set_ylabel(r'$\varepsilon$ ($t$)')
axes[1].legend(loc='center', fontsize=8, frameon=False)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / 'folding.png', dpi=150)
