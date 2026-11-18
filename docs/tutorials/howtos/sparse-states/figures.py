"""Figure and checks of the how-to "How to choose which states the sparse backend solves".

Run it after ``python main.py > main.log``, in the same directory: it reads the ``sparse.log``
and band files of the runs, draws ``windows.png`` and asserts the numbers the page quotes.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

palette = ['#004488', '#D55E00', '#FFC20A', '#009E73', '#8A6800']


def log_section(outputdir: str, title: str) -> str:
    """One section of ``sparse.log``: its title line up to the next blank line."""
    text = Path(outputdir, 'sparse.log').read_text()
    return re.search(rf'^{re.escape(title)}\n-+\n(.*?)(?=\n\n|\Z)', text, re.M | re.S).group(1)


def log_value(section: str, key: str) -> str:
    """The value of one ``key  value`` line of a log section."""
    return re.search(rf'^  {re.escape(key)}\s+(.*)$', section, re.M).group(1)


auto = log_section('output_auto', 'Solve window (mesh pass 1)')
projectable = log_section('output_auto', 'Well-projected range')
path = log_section('output_auto', 'Path pass (bands)')
nev_pass = log_section('output_nev', 'Solve window (mesh pass 1)')
interior = log_section('output_interior_auto', 'Solve window (mesh pass 1)')
print(auto, projectable, path, nev_pass, interior, sep='\n\n')
print(log_section('output_interior_auto', 'Skipped properties'))

# ---- figure: the states each window solves, on the bands of the path
ptop = float(log_value(projectable, 'usable top (eV)').split()[0])
shift = float(log_value(projectable, 'PAO shift (eV)'))
window_top = float(log_value(auto, 'window (eV)').split()[-1])
full = np.loadtxt('output_auto/bands_0.dat')[:, 1:]
inner = np.loadtxt('output_interior/bands_0.dat')[:, 1:]
per_k = np.isfinite(inner).sum(axis=1)
print(f'states per k-point in the interior window: {per_k.min()} to {per_k.max()}')
k = np.arange(len(full))
fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2), sharey=True)
left, right = axes
left.axhspan(full[:, 0].min() - 0.5, window_top, color=palette[2], alpha=0.25, lw=0)
left.plot(k, full, color=palette[0], lw=1.0)
left.set_title(f'from the bottom: dos + transport, top {window_top:.2f} eV', fontsize=9)
right.axhspan(-1.5, 2.5, color=palette[2], alpha=0.25, lw=0)
right.plot(k, full, color='#000000', lw=2.0, alpha=0.2)
right.plot(np.repeat(k, inner.shape[1]), inner.ravel(), '.', ms=1.5, color=palette[0])
right.set_title("interior_window {'elo': -1.5, 'ehi': 2.5}", fontsize=9)
labels = Path('output_auto/kpath_points.txt').read_text().split('\n\n')[0].split()
ticks, names = [0], []
for name, n in zip(labels[0::2], labels[1::2]):
    name = name.replace('gG', r'$\Gamma$')
    names.append(name if len(names) < len(ticks) else f'{names.pop()}|{name}')  # a jump: K|U
    if int(n) > 0:
        ticks.append(ticks[-1] + int(n))
for ax in axes:
    ax.axhline(shift, color='#000000', ls=':', lw=1.0, label='PAO shift')
    ax.axhline(ptop, color=palette[1], ls='--', lw=1.0, label='PAO shift - shift_margin')
    ax.set_xticks(ticks[: len(names)], names, fontsize=7)
    ax.set_xlim(0, len(full) - 1)
right.legend(fontsize=7, loc='lower right', framealpha=1.0)
left.set_ylabel(r'$E - E_\mathrm{VBM}$ (eV)')
left.set_ylim(-13.0, 9.0)
fig.tight_layout()
fig.savefig('windows.png', dpi=150)
plt.close(fig)

# The numbers the page quotes, at the precision it quotes them.
assert abs(shift - 7.161) < 1e-3 and abs(ptop - 6.961) < 1e-3, f'PAO shift {shift}, top {ptop}'
assert abs(window_top - 4.824) < 1e-3, f'window top of dos + transport {window_top} eV'
margin = float(log_value(auto, 'smearing margin (eV)').split()[0])
assert abs(margin - 2.624) < 1e-3, f'smearing margin {margin} eV'
assert log_value(nev_pass, 'bands per k-point').startswith('8 '), 'bands per k-point with nev=8'
assert log_value(interior, 'window (eV)').startswith('[-3.624, 4.124]'), 'automatic interior window'
