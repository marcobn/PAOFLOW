"""Checks of the how-to "How to plot PAOFLOW output with GPAO" (docs/learn/howtos/plotting.md).

The figures of the page are drawn by ``plotting.py``, whose workflow is the plotting itself. Run
this script after it, in the folder of that run, by the path of this script (``python
<PAOFLOW>/docs/learn/howtos/plotting/figures.py``): it asserts the numbers the page quotes, read
from the files in ``output/``.
"""

from pathlib import Path

import numpy as np

out = 'output/'
bands = np.loadtxt(out + 'bands_0.dat')[:, 1:]
cbm = bands[:, 4].min()  # the lowest conduction band: Si has four valence bands
assert abs(cbm - 0.746) < 1e-3, f'Si CBM {cbm:.4f} eV'

energy = np.loadtxt(out + 'dosdk_0.dat')[:, 0]
files = sorted(Path(out).glob('Si*_pdosdk_0.dat'))
valence = energy <= 0
weights = [
    np.trapezoid(
        sum(np.loadtxt(f)[:, 1] for f in files if f'_{s}' in f.name)[valence], energy[valence]
    )
    for s in ['3s', '3p', '3d', '4s', '4p', '4d']
]
expected = [0.49, 1.51, 0.70, 0.19, 0.57, 0.54]
assert np.allclose(weights, expected, atol=0.005), f'Si PDOS per shell {weights}'
