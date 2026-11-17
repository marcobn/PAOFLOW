"""Code of the how-to "How to plot PAOFLOW output with GPAO" (docs/tutorials/howtos/plotting.md).

Run it from the directory that holds ``plotting/``, the folder of ``howto_assets.tar.gz`` of the
release tutorial-assets-v1. It writes the figures of the page to the working directory and ends
with asserts of the numbers the page quotes. The Fermi-surface figure is made by the
``fermi-plotter`` command shown on the page, which needs Mayavi.
"""

import warnings

import matplotlib

matplotlib.use('Agg')  # write the figures to files instead of opening windows
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', 'FigureCanvasAgg is non-interactive')


def save(*names: str) -> None:
    """Write the last ``len(names)`` figures, in the order they were drawn, and close all."""
    for num, name in zip(plt.get_fignums()[-len(names) :], names):
        plt.figure(num).savefig(name, dpi=150)
    plt.close('all')


# [setup]
from PAOFLOW import GPAO

plotter = GPAO.GPAO()
palette = ['#004488', '#D55E00', '#FFC20A']  # deep blue, vermillion, amber
# [/setup]

# [bands]
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
# [/bands]
save('si_bands_dos.png')

# [pdos]
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
# [/pdos]
save('si_pdos.png')

# [berry]
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
# [/berry]
save('fe_berry.png')

# [shc]
plotter.plot_shc('plotting/pt/shcEf_z_xy.dat', title=r'Pt, $\sigma^z_{xy}$', cols=palette[0])
plotter.plot_shc(
    ['plotting/snte/shcEf_z_xy.dat', 'plotting/snte/shcEf_z_yx.dat'],
    title='SnTe monolayer, spin Hall conductivity',
    cols=palette[:2],
    label=r'$\sigma^z$',
)
# [/shc]
save('pt_shc.png', 'snte_shc.png')

# [transport]
plotter.plot_electrical_conductivity(
    'plotting/al/sigmagauss_0.dat', title=r'Al, $\sigma/\tau$', col=palette
)
plotter.plot_seebeck(
    'plotting/alp/Seebeckgauss_0.dat',
    t_ele=[],
    title='AlP, Seebeck coefficient',
    x_lim=(-0.5, 3.5),
    y_lim=(-5000.0, 5000.0),
    col=palette[0],
)
# [/transport]
save('al_sigma.png', 'alp_seebeck.png')

# [checks]
import numpy as np

energy, dos = np.loadtxt('plotting/si/dosdk_0.dat', unpack=True)
pdos = sum(np.loadtxt(f)[:, 1] for f in files)
print(f'max |sum of PDOS - DOS|: {np.abs(pdos - dos).max():.1e}')

sigma = np.loadtxt('plotting/al/sigmagauss_0.dat')
at_ef = sigma[np.abs(sigma[:, 1]).argmin()]
print(f'Al at E_F: off-diagonal / xx = {np.abs(at_ef[5:8]).max() / at_ef[2]:.1e}')
# [/checks]

# The numbers the page quotes, at the precision it quotes them.
bands = np.loadtxt(si + 'bands_0.dat')[:, 1:]
cbm = bands[:, 4].min()  # the lowest conduction band: Si has four valence bands
assert abs(cbm - 0.754) < 1e-3, f'Si CBM {cbm:.4f} eV'

valence = energy <= 0
weights = [
    np.trapezoid(
        sum(np.loadtxt(f)[:, 1] for f in files if f'_{s}' in f.name)[valence], energy[valence]
    )
    for s in shells
]
assert np.allclose(weights, [0.81, 2.27, 0.92], atol=0.005), f'Si PDOS per shell {weights}'

shc = np.loadtxt('plotting/pt/shcEf_z_xy.dat')
shc_ef = shc[np.abs(shc[:, 0]).argmin(), 1]  # the grid point nearest E_F, as for Al
assert abs(shc_ef - 1961) < 1, f'Pt SHC at E_F {shc_ef:.1f} (Ohm cm)^-1'

assert abs(at_ef[2] - 3.20e21) < 0.005e21, f'Al sigma/tau at E_F {at_ef[2]:.3e}'

seebeck = np.loadtxt('plotting/alp/Seebeckgauss_0.dat')
mu, s_avg = seebeck[:, 1], seebeck[:, 2:5].mean(axis=1) * 1e6
window = (mu > -0.5) & (mu < 3.5)
extremes = s_avg[window].max(), s_avg[window].min()
assert np.allclose(extremes, [4495, -2876], atol=1), f'AlP Seebeck extremes {extremes} muV/K'
