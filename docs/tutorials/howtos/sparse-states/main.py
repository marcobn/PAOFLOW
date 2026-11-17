"""Code of the how-to "How to choose which states the sparse backend solves".

The page is docs/tutorials/howtos/sparse-states.md. Run the script from the directory that holds
``tutorial01/silicon.save``, the folder of ``tutorial_assets.tar.gz`` of the release
tutorial-assets-v1. It runs the Si workflow of Tutorial 04 on the two-atom cell with the solve
window derived from the property ranges, with a fixed number of bands and with two interior
windows, prints the log lines the page quotes, draws its figure and ends with asserts.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW.PAOFLOW import PAOFLOW

savedir = 'tutorial01/silicon.save'
palette = ['#004488', '#D55E00', '#FFC20A', '#009E73', '#8A6800']


# [setup]
def si_hamiltonian(outputdir: str, sparse_config: dict) -> PAOFLOW:
    """The truncated PAO Hamiltonian of Si on the sparse backend."""
    paoflow = PAOFLOW(
        savedir=savedir,
        outputdir=outputdir,
        smearing='gauss',
        sparse=True,
        sparse_config=sparse_config,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    return paoflow


# [/setup]


def log_section(outputdir: str, title: str) -> str:
    """One section of ``sparse.log``: its title line up to the next blank line."""
    text = Path(outputdir, 'sparse.log').read_text()
    return re.search(rf'^{re.escape(title)}\n-+\n(.*?)(?=\n\n|\Z)', text, re.M | re.S).group(1)


def log_value(section: str, key: str) -> str:
    """The value of one ``key  value`` line of a log section."""
    return re.search(rf'^  {re.escape(key)}\s+(.*)$', section, re.M).group(1)


# ---- the window derived from the property ranges
paoflow = si_hamiltonian('output_auto', {'bond_order': 36})
paoflow.bands(ibrav=2, nk=500)
paoflow.interpolated_hamiltonian()
# [auto]
with paoflow.sparse.fused():
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
# [/auto]
paoflow.finish_execution()
auto = log_section('output_auto', 'Solve window (mesh pass 1)')
projectable = log_section('output_auto', 'Well-projected range')
path = log_section('output_auto', 'Path pass (bands)')
print(auto, projectable, path, sep='\n\n')

# ---- a fixed number of bands
# [nev]
sparse_config = {'bond_order': 36, 'nev': 8}
# [/nev]
paoflow = si_hamiltonian('output_nev', sparse_config)
paoflow.interpolated_hamiltonian()
with paoflow.sparse.fused():
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
paoflow.finish_execution()
nev_pass = log_section('output_nev', 'Solve window (mesh pass 1)')
print(nev_pass)

# ---- interior windows: fixed bounds for the bands, bounds around the property ranges for a pass
# [interior-fixed]
sparse_config = {'bond_order': 36, 'interior_window': {'elo': -1.5, 'ehi': 2.5}}
# [/interior-fixed]
paoflow = si_hamiltonian('output_interior', sparse_config)
paoflow.bands(ibrav=2, nk=500)
paoflow.finish_execution()

# [interior-auto]
sparse_config = {'bond_order': 36, 'interior_window': {}}
# [/interior-auto]
paoflow = si_hamiltonian('output_interior_auto', sparse_config)
paoflow.interpolated_hamiltonian()
# [interior-auto-pass]
with paoflow.sparse.fused():
    paoflow.dos(emin=-1.0, emax=1.5, ne=500)
    paoflow.transport(emin=-1.0, emax=1.5)
    paoflow.doping(emin=-12.0, emax=2.0, doping_conc=-1.0e19)  # needs every occupied state
paoflow.bands(ibrav=2, nk=500)  # after a property: the window comes from its range
# [/interior-auto-pass]
paoflow.finish_execution()
interior = log_section('output_interior_auto', 'Solve window (mesh pass 1)')
print(interior)
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
