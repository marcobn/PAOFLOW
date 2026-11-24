"""Figures and checks of the how-to "How to choose the truncation of the sparse Hamiltonian".

Run it after ``python sparse_truncation.py > sparse_truncation.log``, in the folder of that run, by
the path of this script (``python <PAOFLOW>/docs/learn/howtos/sparse-truncation/figures.py``): it
reads the band files, ``sparse.log`` and ``hamiltonian.npz`` of the runs and
``sparse_truncation.log``, draws ``hoppings.png``, ``degeneracy.png`` and ``convergence.png`` and
asserts the numbers the page quotes.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

palette = ['#004488', '#D55E00', '#FFC20A', '#009E73', '#8A6800']
orders = [5, 10, 15, 20, 25, 30, 33, 36, 40, 44, 47]


def bands(outputdir: str) -> np.ndarray:
    """Bands of one run, one row per k-point."""
    return np.loadtxt(f'{outputdir}/bands_0.dat')[:, 1:]


def log_value(outputdir: str, key: str) -> str:
    """The value of one line of ``sparse.log``."""
    text = Path(outputdir, 'sparse.log').read_text()
    return re.search(rf'^  {re.escape(key)}\s+(.*)$', text, re.M).group(1)


def save(fig: plt.Figure, name: str) -> None:
    """Write one figure of the page at 150 dpi and close it."""
    fig.savefig(name, dpi=150, bbox_inches='tight')
    plt.close(fig)


full = bands('output_full')
near_gap = (full > -2.0) & (full < 2.0)  # states within 2 eV of the valence-band maximum
truncated = {n: bands(f'output_bo{n}') for n in orders}
truncated[1e-4], truncated[1e-3] = bands('output_t1e-04'), bands('output_t1e-03')
# largest band shift (eV) within 2 eV of the VBM, against the full H(R)
shift = {key: np.nanmax(np.abs(b - full)[near_gap]) for key, b in truncated.items()}
print('rcut=30.0:', log_value('output_rcut30', 'cutoff radius (Bohr)'))
print('largest band shift within 2 eV of the VBM (meV):')
for key, value in sorted(shift.items(), key=lambda kv: -kv[0]):
    print(f'  {key!s:>8}  {1000 * value:6.1f}')
log = Path('sparse_truncation.log').read_text()
too_long = re.search(r'^bond_order=48: (.*)$', log, re.M).group(1)

# ---- figure: every hopping against its bond length, inside the range the grid represents
archive = np.load('output_full/hamiltonian.npz')
safe_radius = json.loads(str(archive['metadata_json']))['aliasing_safe_radius']
length, value = archive['bond_distance'], np.abs(archive['bond_value'][:, 0])
shown = (length > 1e-6) & (length < safe_radius + 2.0)
r36 = float(log_value('output_bo36', 'cutoff radius (Bohr)').split()[0])
r10 = float(log_value('output_bo10', 'cutoff radius (Bohr)').split()[0])
print(
    f'nearest-neighbour shell: {length[shown].min():.2f} Bohr, largest hopping {value[shown].max():.2f} eV'
)
print(
    f'largest hopping beyond 30 Bohr inside the grid: {value[shown & (length > 30.0)].max():.1e} eV'
)
dropped = (length > r36) & (length < safe_radius) & (value > 1e-3)
print(f'bond_order=36 drops {dropped.sum()} hoppings above 1e-3 eV inside the grid')
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.semilogy(length[shown], value[shown], '.', ms=1.0, color=palette[0], rasterized=True)
ax.axvline(r10, color=palette[3], ls='-', lw=1.2, label=f'bond_order=10 ({r10:.1f} Bohr)')
ax.axvline(r36, color=palette[3], ls='-.', lw=1.2, label=f'bond_order=36 ({r36:.1f} Bohr)')
ax.axvline(
    safe_radius, color='#000000', ls=':', lw=1.2, label=f'longest cutoff ({safe_radius:.1f} Bohr)'
)
ax.axhline(1e-3, color=palette[1], ls='--', lw=1.2, label=r'hopping_threshold $10^{-3}$ eV')
ax.axhline(1e-4, color=palette[2], ls='--', lw=1.2, label=r'hopping_threshold $10^{-4}$ eV')
top = ax.secondary_xaxis('top')  # one tick per neighbour shell
top.set_xticks(archive['shell_distances'][archive['shell_distances'] < safe_radius + 2.0])
top.set_xticklabels([])
ax.set_xlim(0.0, safe_radius + 2.0)
ax.set_ylim(1e-8, 10.0)
ax.set_xlabel(r'bond length $|a\mathbf{R} + \tau_\alpha - \tau_\beta|$ (Bohr)')
ax.set_ylabel(r'$|H_{\alpha\beta}(\mathbf{R})|$ (eV)')
ax.legend(fontsize=7, loc='lower left')
save(fig, 'hoppings.png')

# ---- figure: the doubly degenerate conduction pair on Gamma-L
counts = Path('output_full/kpath_points.txt').read_text().split('\n\n')[0].split()[1::2]
i_l = sum(int(n) for n in counts[:5])  # the fcc path is Gamma-X-W-K-Gamma-L-...
i_gamma = i_l - int(counts[4])
pair = [5, 6]  # the lowest doubly degenerate conduction bands on Gamma-L
cases = {'bond_order=36': truncated[36], r'hopping_threshold $10^{-3}$ eV': truncated[1e-3]}
split = {key: abs(b[i_l, pair[1]] - b[i_l, pair[0]]) for key, b in cases.items()}
move = {key: b[i_l, pair].mean() - full[i_l, pair].mean() for key, b in cases.items()}
print(f'pair at L: {full[i_l, pair].mean():.3f} eV above the VBM')
for key in cases:
    print(f'  {key}: split {1000 * split[key]:.2f} meV, moved {1000 * move[key]:.1f} meV')
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
x = np.arange(i_gamma + int(counts[4]) // 2, i_l + 1)
for ax, (key, b), color in zip(axes, cases.items(), palette):
    ax.plot(x, full[x][:, pair], color='#000000', lw=3.0, alpha=0.3)
    ax.plot([], [], color='#000000', lw=3.0, alpha=0.3, label='full H(R)')
    ax.plot(x, b[x][:, pair], color=color, lw=1.0)
    ax.plot([], [], color=color, lw=1.0, label=key)
    ax.set_title(f'pair split by {1000 * split[key]:.2f} meV at L', fontsize=9)
    ax.set_xticks([i_l], ['L'])
    ax.set_xlabel(r'along $\Gamma$-L')
    ax.legend(fontsize=7, loc='lower left')
axes[0].set_ylabel(r'$E - E_\mathrm{VBM}$ (eV)')
save(fig, 'degeneracy.png')

# ---- figure: convergence with bond_order
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.semilogy(orders, [1000 * shift[n] for n in orders], 'o-', color=palette[0], label='bond_order')
ax.axhline(
    1000 * shift[1e-3], color=palette[1], ls='--', lw=1.2, label=r'hopping_threshold $10^{-3}$ eV'
)
ax.axhline(
    1000 * shift[1e-4], color=palette[2], ls='--', lw=1.2, label=r'hopping_threshold $10^{-4}$ eV'
)
ax.set_xlabel('bond_order')
ax.set_ylabel('largest band shift (meV)')
ax.set_title('Si, bands within 2 eV of the VBM, against the full H(R)', fontsize=10)
ax.legend(fontsize=8)
save(fig, 'convergence.png')

bound36 = float(log_value('output_bo36', 'max eigenvalue shift from truncation (eV)'))
lowest8 = np.nanmax(np.abs(truncated[36] - full)[:, :8])
print(
    f'bond_order=36: eigenvalue bound {bound36} eV, largest shift of the lowest 8 bands {lowest8:.2f} eV'
)

# The numbers the page quotes, at the precision it quotes them.
assert '47 neighbour shells' in too_long and '36.718' in too_long, f'longest cutoff: {too_long}'
assert (
    abs(1000 * shift[36] - 93) < 1 and abs(1000 * shift[47] - 70) < 1
), f'band shift within 2 eV: bond_order=36 {1000 * shift[36]:.1f}, 47 {1000 * shift[47]:.1f} meV'
assert (
    abs(1000 * shift[1e-3] - 91) < 1 and abs(1000 * shift[1e-4] - 11) < 1
), f'band shift within 2 eV: threshold 1e-3 {1000 * shift[1e-3]:.1f}, 1e-4 {1000 * shift[1e-4]:.1f} meV'
assert (
    split['bond_order=36'] < 1e-4
), f'L pair split by bond_order=36: {split["bond_order=36"]:.2e} eV'
assert abs(1000 * split[r'hopping_threshold $10^{-3}$ eV'] - 11.9) < 0.1, 'L pair split by 1e-3 eV'
