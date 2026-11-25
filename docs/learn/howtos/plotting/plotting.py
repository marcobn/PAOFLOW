from pathlib import Path

import matplotlib.pyplot as plt
from PAOFLOW import GPAO

plotter = GPAO.GPAO()
palette = ['#004488', '#D55E00', '#FFC20A', '#009E73', '#56B4E9', '#CC79A7']
out = 'output/'  # the outputdir of main.py

# Band structure and DOS
plotter.plot_bands(
    out + 'bands_0.dat',
    sym_points=out + 'kpath_points.txt',
    label=r'$E - E_F$ (eV)',
    y_lim=(-12.0, 6.0),
    cols=palette[0],
)
plotter.plot_dos(out + 'dosdk_0.dat', x_lim=(-12.0, 2.2), col=palette[0])
plotter.plot_dos_beside_bands(
    out + 'dosdk_0.dat',
    out + 'bands_0.dat',
    sym_points=out + 'kpath_points.txt',
    title='Si, bands and DOS',
    band_label=r'$E - E_F$ (eV)',
    y_lim=(-12.0, 2.2),
    col=palette[0],
)
plt.savefig('si_bands_dos.png', dpi=150)

# Projected DOS grouped by orbital
files = sorted(Path(out).glob('Si*_pdosdk_0.dat'))
shell = [f.name.split('_')[1][:2] for f in files]  # Si1_3px_pdosdk_0.dat -> 3p
shells = sorted(set(shell), key=lambda s: (s[0], 'spdf'.index(s[1])))
plotter.plot_pdos(
    [str(f) for f in files],
    title='Si, PDOS per shell',
    x_lim=(-12.0, 2.2),
    cols=palette[: len(shells)],
    labels=[f'Si {s}' for s in shells],
    psum_inds=[[i for i, s in enumerate(shell) if s == group] for group in shells],
)
plt.savefig('si_pdos.png', dpi=150)
