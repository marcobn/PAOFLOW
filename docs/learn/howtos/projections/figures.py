"""Figures and checks of the how-to "How to choose the projections onto the PAO basis".

Run it in the folder where ``projections.py`` of the case ``si_bands_transport`` ran, by the path of
this script (``python <PAOFLOW>/docs/learn/howtos/projections/figures.py``). It reads
``projections.log`` (the printed output of ``projections.py``), ``output_<route>/bands_0.dat`` and
``bands.out``, the output of the DFT band run ``pw.x -in bands.in`` of the case, writes
``projectability.png``, ``bands.png`` and ``band_errors.png``, and asserts the numbers the page
quotes.
"""

import re

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import GPAO

ROUTES = ['projwfc', 'minimal', 'standard', 'extended', 'generated', 'custom']  # order of the runs
COLORS = {
    'projwfc': '#000000',
    'minimal': '#004488',
    'custom': '#56B4E9',
    'standard': '#009E73',
    'extended': '#FFC20A',
}
DFT = '#D55E00'
PTHR = 0.95

log = open('projections.log').read()
vectors = re.findall(r'Projectability vector:\s*\n\s*\[([^\]]*)\]', log)
kept = [int(n) for n in re.findall(r'# of bands with good projectability > [\d.]+ = (\d+)', log)]
shift = [float(e) for e in re.findall(r'Range of suggested shift\s+([-\d.]+)', log)]
assert len(vectors) == len(kept) == len(shift) == len(ROUTES), 'one block per route in the log'
pbar = {r: np.array(v.split(), dtype=float) for r, v in zip(ROUTES, vectors)}
kept = dict(zip(ROUTES, kept))
shift = dict(zip(ROUTES, shift))  # the PAO shift, relative to E_F like the bands files
bands = {r: np.loadtxt(f'output_{r}/bands_0.dat')[:, 1:] for r in ROUTES}
efermi = float(re.search(r'DFT Fermi energy: ([\d.]+) eV', log).group(1))

# DFT bands: k in units of 2 pi / a, as in kpath_points.txt, energies in eV relative to E_F
out = open('bands.out').read().split('End of band structure calculation')[1]
blocks = re.findall(r'k =(.*?)\(\s*\d+ PWs\)\s+bands \(ev\):\s+(.*?)\n\s*\n', out, re.S)
k_dft = np.array([re.findall(r'-?\d+\.\d+', k) for k, _ in blocks], dtype=float)
e_dft = np.array([e.split() for _, e in blocks], dtype=float) - efermi
path = np.array(
    [line.split() for line in open('output_minimal/kpath_points.txt') if len(line.split()) == 3],
    dtype=float,
)
on_path = np.array([np.linalg.norm(path - k, axis=1).argmin() for k in k_dft])
assert np.linalg.norm(path[on_path] - k_dft, axis=1).max() < 1e-3, 'bands.in follows the path'
nb = e_dft.shape[1]
error = {r: np.abs(bands[r][on_path, :nb] - e_dft) for r in ROUTES}  # eV, band by band

# Projectability of each band
fig, ax = plt.subplots(figsize=(6.4, 4.0))
n = np.arange(1, len(pbar['minimal']) + 1)
for route in ['minimal', 'custom', 'standard', 'extended']:
    ax.plot(n, pbar[route], 'o-', color=COLORS[route], ms=4, label=f'{route} ({kept[route]} kept)')
ax.plot(n, pbar['projwfc'], 'o', mfc='none', color=COLORS['projwfc'], ms=8,
        label=f'projwfc.x ({kept["projwfc"]} kept)')  # fmt: skip
ax.axhline(PTHR, color='#8A6800', ls='--', lw=1)
ax.text(1, PTHR + 0.008, 'pthr = 0.95', color='#8A6800')
ax.set_xlabel('band $n$')
ax.set_ylabel(r'projectability $\bar p_n$')
ax.set_xticks(n)
ax.set_ylim(0.3, 1.02)
ax.legend(loc='lower left')
fig.tight_layout()
fig.savefig('projectability.png', dpi=150)

# Bands of the three presets and of DFT; each basis flattens the bands it does not keep at its shift
plotter = GPAO.GPAO()
order = ['extended', 'standard', 'minimal']  # the smallest basis on top
ax = plotter.plot_bands(
    [f'output_{r}/bands_0.dat' for r in order],
    sym_points='output_minimal/kpath_points.txt',
    title='',
    label='$E - E_F$ (eV)',
    labels=order,
    y_lim=(-13, 16),
    cols=[COLORS[r] for r in order],
    show=False,
)
ax.plot(on_path, e_dft[:, 0], 'o', color=DFT, ms=3, label='DFT')
ax.plot(on_path, e_dft[:, 1:], 'o', color=DFT, ms=3)
ax.legend(loc='lower left')
ax.figure.set_size_inches(6.4, 4.4)
ax.figure.tight_layout()
ax.figure.savefig('bands.png', dpi=150)

# Error of each basis against DFT, for the DFT states below its PAO shift
fig, ax = plt.subplots(figsize=(6.4, 4.0))
for route in ['minimal', 'standard', 'extended']:
    below = e_dft < shift[route]
    ax.semilogy(e_dft[below], 1e3 * error[route][below], 'o', color=COLORS[route], ms=3,
                label=route)  # fmt: skip
    ax.axvline(shift[route], color=COLORS[route], ls='--', lw=1)
ax.set_xlabel(r'$\varepsilon_{n\mathbf{k}}^\mathrm{DFT} - E_F$ (eV)')
ax.set_ylabel(
    r'$|\varepsilon^\mathrm{PAO}_{n\mathbf{k}} - \varepsilon^\mathrm{DFT}_{n\mathbf{k}}|$ (meV)'
)
ax.set_ylim(1e-2, 2e3)
ax.legend(loc='upper left')
fig.tight_layout()
fig.savefig('band_errors.png', dpi=150)

# Numbers quoted on the page
assert [kept[r] for r in ROUTES] == [9, 9, 13, 14, 13, 10], f'kept bands {kept}'
want = {'minimal': 7.161, 'custom': 7.753, 'standard': 10.962, 'extended': 14.788}
assert all(abs(shift[r] - e) < 1e-3 for r, e in want.items()), f'PAO shifts above E_F {shift}'
dp = np.abs(pbar['projwfc'] - pbar['minimal']).max()
de = np.abs(bands['projwfc'][:, :9] - bands['minimal'][:, :9]).max()
assert dp < 2.5e-3 and de < 2.5e-3, f'projwfc.x vs minimal: {dp:.1e} in p, {de * 1e3:.1f} meV'
dg = np.abs(bands['standard'] - bands['generated']).max()
assert dg < 1e-6, f'tables vs generated basis {dg:.1e} eV'
worst = {
    (r, emax): 1e3 * error[r][e_dft < emax].max()
    for r in ['minimal', 'standard', 'extended']
    for emax in (2.0, 6.0)
}  # fmt: skip
quoted = {
    ('minimal', 2.0): 27, ('standard', 2.0): 77, ('extended', 2.0): 37,
    ('minimal', 6.0): 174, ('standard', 6.0): 77, ('extended', 6.0): 37,
}  # fmt: skip
assert all(abs(worst[key] - v) < 1 for key, v in quoted.items()), f'largest errors (meV) {worst}'
