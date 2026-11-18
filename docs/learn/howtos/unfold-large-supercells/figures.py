"""Figure and checks of the how-to "How to unfold the bands of large supercells with the sparse solver".

The page is docs/learn/howtos/unfold-large-supercells.md. Run it after ``python main.py`` in
the same directory: it reads ``bilayer_unfolded.npz`` and the saved supercell model, writes the
figure of the page to the working directory and ends with asserts of the numbers the page quotes.
"""

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW.models.edtb_params import EDTBModel
from PAOFLOW.spectrum.sparse_bands import SparseEDTB

run = np.load('bilayer_unfolded.npz')
k, ticks, E_pc, E_sc, W = run['kdist'], run['ticks'], run['E_pc'], run['E_sc'], run['W']
bottom, top = E_sc.min(axis=1), E_sc.max(axis=1)  # window edges per k-point

# Energy zero: the mean of the four primitive states at K nearest sigma (the two Dirac points).
iK = int(np.argmin(np.abs(k - ticks[2])))
eK = np.sort(E_pc[iK])
dirac = np.sort(eK[np.argsort(np.abs(eK + 0.5))[:4]])
zero = dirac.mean()
print(f'Dirac states at K {np.round(dirac, 4)} eV, E_D = {zero:.4f} eV')
print(f'window relative to E_D: {bottom.max() - zero:.3f} to {top.min() - zero:.3f} eV')

fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot(k, E_pc - zero, color='#004488', lw=0.9)
ax.plot(k, bottom - zero, color='#8A6800', lw=1.0, ls='--')
ax.plot(k, top - zero, color='#8A6800', lw=1.0, ls='--')
kk = np.repeat(k[:, None], E_sc.shape[1], axis=1)
shown = W > 0.02
ax.scatter(kk[shown], E_sc[shown] - zero, s=14 * W[shown], color='#D55E00',
           edgecolors='none', zorder=3)  # fmt: skip
ax.set_xticks(ticks, [r'$\Gamma$', 'M', 'K', r'$\Gamma$'])
for t in ticks:
    ax.axvline(t, color='#000000', lw=0.6)
ax.set(xlim=(k[0], k[-1]), ylim=(-4, 4), ylabel=r'$E - E_\mathrm{D}$ (eV)')
fig.tight_layout()
fig.savefig('bilayer_window.png', dpi=150)

# The numbers the page quotes, at the precision it quotes them.
sc = EDTBModel.from_files('carbon.json', 'bilayer_4_3.json').to_model_dict()
nawf = 9 * len(sc['model']['atoms'])
dense_gb = nawf**2 * 16 * 125 / 1e9  # complex H(R) on the 5 x 5 x 5 grid of the dense route
print(f'(4, 3) cell: {nawf} orbitals, dense H(R) {dense_gb:.1f} GB')
assert run['N'] == 37 and nawf == 1332, f'N = {run["N"]}, {nawf} orbitals'
assert abs(zero + 0.519) < 1e-3, f'Dirac energy E_D {zero:.4f} eV'
split = (dirac[2:].mean() - dirac[:2].mean()) / 2
assert abs(split - 0.094) < 1e-3, f'AA Dirac points at E_D +- {split:.4f} eV'
window = (bottom.max() - zero, top.min() - zero)
assert np.allclose(window, (-1.71, 1.75), atol=5e-3), f'window edges {window} eV'

# The multiplet at the lower edge of the window at Gamma: returned states against its size.
edge = E_sc[0].min()
kept = int(np.sum(np.abs(E_sc[0] - edge) < 1e-3))
wider = SparseEDTB(sc, verbose=False).eigvals_at_k(np.zeros(3), n_eigs=80, sigma=-0.5)
size = int(np.sum(np.abs(wider - edge) < 1e-3))
print(f'Gamma, lower edge at {edge - zero:.3f} eV: {kept} of {size} states of the multiplet')
assert (kept, size) == (4, 6), f'states of the edge multiplet at Gamma: {kept} of {size}'
