import matplotlib.pyplot as plt
from PAOFLOW import GPAO

plotter = GPAO.GPAO()
palette = ['#004488', '#D55E00', '#FFC20A']  # deep blue, vermillion, amber

# Band structure and DOS
si = 'plotting/si/'
plotter.plot_bands(
    si + 'bands_0.dat',
    sym_points=si + 'kpath_points.txt',
    label=r'$E - E_F$ (eV)',
    y_lim=(-12.0, 6.0),
    cols=palette[0],
)
plotter.plot_dos(si + 'dosdk_0.dat', x_lim=(-12.0, 2.2), col=palette[0])
plotter.plot_dos_beside_bands(
    si + 'dosdk_0.dat',
    si + 'bands_0.dat',
    sym_points=si + 'kpath_points.txt',
    title='Si, bands and DOS',
    band_label=r'$E - E_F$ (eV)',
    y_lim=(-12.0, 2.2),
    col=palette[0],
)
plt.savefig('si_bands_dos.png', dpi=150)

# Projected DOS grouped by orbital
from pathlib import Path

files = sorted(Path(si).glob('Si*_pdosdk_0.dat'))
shells = ['3s', '3p', '3d']
plotter.plot_pdos(
    [str(f) for f in files],
    title='Si, PDOS per shell',
    x_lim=(-12.0, 2.2),
    cols=palette,
    labels=[f'Si {s}' for s in shells],
    psum_inds=[[i for i, f in enumerate(files) if f'_{s}' in f.name] for s in shells],
)
plt.savefig('si_pdos.png', dpi=150)

# Berry curvature under the bands of Fe
fe = 'plotting/fe/'
plotter.plot_berry_under_bands(
    fe + 'Omega_z_yz.dat',
    fe + 'bands_0.dat',
    sym_points=fe + 'kpath_points.txt',
    title='Fe, bands and Berry curvature',
    y_lim=(-4.0, 4.0),
    band_label=r'$E - E_F$ (eV)',
    berry_label=r'$\Omega_{yz}$ (bohr$^2$)',
    col=palette[0],
)
plt.savefig('fe_berry.png', dpi=150)

# Spin Hall conductivity of Pt and of the SnTe monolayer
plotter.plot_shc('plotting/pt/shcEf_z_xy.dat', title=r'Pt, $\sigma^z_{xy}$', cols=palette[0])
plt.savefig('pt_shc.png', dpi=150)
plotter.plot_shc(
    ['plotting/snte/shcEf_z_xy.dat', 'plotting/snte/shcEf_z_yx.dat'],
    title='SnTe monolayer, spin Hall conductivity',
    cols=palette[:2],
    label=r'$\sigma^z$',
)
plt.savefig('snte_shc.png', dpi=150)

# Conductivity of Al and Seebeck coefficient of AlP
plotter.plot_electrical_conductivity(
    'plotting/al/sigmagauss_0.dat', title=r'Al, $\sigma/\tau$', col=palette
)
plt.savefig('al_sigma.png', dpi=150)
plotter.plot_seebeck(
    'plotting/alp/Seebeckgauss_0.dat',
    t_ele=[],
    title='AlP, Seebeck coefficient',
    x_lim=(-0.5, 3.5),
    y_lim=(-5000.0, 5000.0),
    col=palette[0],
)
plt.savefig('alp_seebeck.png', dpi=150)
