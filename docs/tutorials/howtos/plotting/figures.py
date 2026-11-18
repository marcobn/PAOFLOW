"""Checks of the how-to "How to plot PAOFLOW output with GPAO" (docs/tutorials/howtos/plotting.md).

The figures of the page are drawn by ``main.py``, whose workflow is the plotting itself. Run this
script after it, from the same directory (the one that holds ``plotting/``): it asserts the
numbers the page quotes, read from the data files under ``plotting/``.
"""

from pathlib import Path

import numpy as np

si = 'plotting/si/'
bands = np.loadtxt(si + 'bands_0.dat')[:, 1:]
cbm = bands[:, 4].min()  # the lowest conduction band: Si has four valence bands
assert abs(cbm - 0.754) < 1e-3, f'Si CBM {cbm:.4f} eV'

energy = np.loadtxt(si + 'dosdk_0.dat')[:, 0]
files = sorted(Path(si).glob('Si*_pdosdk_0.dat'))
valence = energy <= 0
weights = [
    np.trapezoid(
        sum(np.loadtxt(f)[:, 1] for f in files if f'_{s}' in f.name)[valence], energy[valence]
    )
    for s in ['3s', '3p', '3d']
]
assert np.allclose(weights, [0.81, 2.27, 0.92], atol=0.005), f'Si PDOS per shell {weights}'

shc = np.loadtxt('plotting/pt/shcEf_z_xy.dat')
shc_ef = shc[np.abs(shc[:, 0]).argmin(), 1]  # the grid point nearest E_F, as for Al
assert abs(shc_ef - 1961) < 1, f'Pt SHC at E_F {shc_ef:.1f} (Ohm cm)^-1'

sigma = np.loadtxt('plotting/al/sigmagauss_0.dat')
at_ef = sigma[np.abs(sigma[:, 1]).argmin()]
assert abs(at_ef[2] - 3.20e21) < 0.005e21, f'Al sigma/tau at E_F {at_ef[2]:.3e}'

seebeck = np.loadtxt('plotting/alp/Seebeckgauss_0.dat')
mu, s_avg = seebeck[:, 1], seebeck[:, 2:5].mean(axis=1) * 1e6
window = (mu > -0.5) & (mu < 3.5)
extremes = s_avg[window].max(), s_avg[window].min()
assert np.allclose(extremes, [4495, -2876], atol=1), f'AlP Seebeck extremes {extremes} muV/K'
