"""Per-primitive-cell observables of one benchmark case, read from its output files.

A cell doubled ``d`` times is still bulk Si, so every observable here is a
curve ``y(x)`` that must not change with ``d``.  :func:`load` reads the curves
of one case directory and normalizes them per primitive cell (``N = 2**d``):

==================  =========================================  ======================
name                source                                     normalization
==================  =========================================  ======================
``dos``             ``dosdk_0.dat``                            / N
``pdos_s/p/d``      ``*_pdosdk_0.dat``, summed by orbital l    / N
``sigma_xx`` etc.   ``sigma/Seebeck/kappa/PF<smearing>_0.dat``  none (per volume)
``hall_trace``      ``hall_trace_<smearing>_0.dat``            none (per volume)
``cond_xx/xy``      ``cond_xx_0.dat``, ``cond_xy_0.dat``       / N (k-averaged, not per volume)
``doping_mu``       ``doping_*.dat``                           none (mu against T)
``epsi_xx`` etc.    ``epsi_xx.dat`` ...                        none (per volume)
``epsr_xx`` etc.    ``epsr/eels/refl/nref/kref/alpha/sigmar/    none (per volume)
                    sigmai_xx.dat`` ..., only with ``optics``
``inv_mass_dos``    ``effective_masses_0.dat``                 / (N nk), see below
==================  =========================================  ======================

``inv_mass_dos`` turns the per-state effective masses into a curve:
:math:`\\sum_{nk} g(E - E_{nk}) / m^{dos}_{nk}`, Gaussian ``g`` of width
``MASS_SMEARING`` eV, over all states and divided by the total number of
k-points per primitive cell.  The states of a doubled cell on its mesh are
exactly those of the primitive cell on its own mesh, so this curve is the same
for every ``d``, whereas the file itself is not comparable line by line.

``cond_xy`` vanishes by cubic symmetry, so its deviation is measured on the
scale of ``cond_xx``.  ``epsi_*`` of an energy-window run is cut at the window
top minus ``EPS_MARGIN``: the sparse Kubo-Greenwood sum runs over the window
bands only, so transitions into states above it are missing.  For the same
reason epsilon' is not compared at all, and the curves derived from it
(``optics=True``) are only read for plotting; they carry the same cut.
"""

import glob
import json
import os

import numpy as np

MASS_SMEARING = 0.05
MASS_GRID = np.linspace(-2.0, 2.0, 801)
EPS_MARGIN = 0.5
TRANSPORT = {'sigma': 'sigma', 'seebeck': 'Seebeck', 'kappa': 'kappa', 'pf': 'PF'}
TENSOR = ['xx', 'yy', 'zz', 'xy', 'xz', 'yz']
OPTICS = ['epsr', 'eels', 'refl', 'nref', 'kref', 'alpha', 'sigmar', 'sigmai']


def load_record(case_dir):
    with open(os.path.join(case_dir, 'record.json')) as f:
        return json.load(f)


def _read(path):
    return np.loadtxt(path, ndmin=2) if os.path.exists(path) else None


def _orbital_l(label):
    """``'Si1_3px'`` -> ``'p'``."""
    return label.split('_')[-1].lstrip('0123456789')[:1]


def _pdos_by_l(case_dir, rec, N):
    """PDOS per primitive cell summed by orbital l, cached in the case directory."""
    cache = os.path.join(case_dir, 'pdos_by_l.npz')
    if os.path.exists(cache):
        data = np.load(cache)
        return {key: (data['E'], data[key]) for key in data.files if key != 'E'}
    out = os.path.join(case_dir, 'output')
    labels = rec['hamiltonian']['base'].get('orbital_labels')
    nawf = rec['hamiltonian'].get('nawf', rec['hamiltonian']['base']['nawf'])
    if not labels:
        return {}
    E, sums = None, {}
    for m in range(nawf):
        base = labels[m % len(labels)]
        a = _read(os.path.join(out, '%s_pdosdk_0.dat' % base)) if N == 1 else None
        if a is None:
            a = _read(os.path.join(out, 'orb_%d_pdosdk_0.dat' % m))
        if a is None:
            return {}
        E = a[:, 0]
        l = _orbital_l(base)
        sums[l] = sums.get(l, 0.0) + a[:, 1]
    curves = {'pdos_' + l: y / N for l, y in sums.items()}
    np.savez(cache, E=E, **curves)
    return {key: (E, y) for key, y in curves.items()}


def load(case_dir, pdos=True, optics=False):
    """Every per-cell curve of one finished case.

    Parameters
    ----------
    pdos : bool
        Read the PDOS summed by orbital l (one file per orbital on a doubled cell).
    optics : bool
        Also read epsilon' and the optical constants derived from it.

    Returns
    -------
    dict
        ``{name: {'x': array, 'y': array, 'scale': float or None,
        'xmax': float or None}}``.  ``scale`` replaces the reference's RMS as
        the norm of the deviation; ``xmax`` cuts the comparison range.
    """
    rec = load_record(case_dir)
    out = os.path.join(case_dir, 'output')
    N = 2 ** int(rec['case']['d'])
    smearing = rec['case']['system']['smearing']
    curves = {}

    def add(name, x, y, scale=None, xmax=None):
        curves[name] = {'x': np.asarray(x), 'y': np.asarray(y), 'scale': scale, 'xmax': xmax}

    a = _read(os.path.join(out, 'dosdk_0.dat'))
    if a is not None:
        add('dos', a[:, 0], a[:, 1] / N)
    if pdos:
        for name, (E, y) in _pdos_by_l(case_dir, rec, N).items():
            add(name, E, y)

    for key, stem in TRANSPORT.items():
        a = _read(os.path.join(out, '%s%s_0.dat' % (stem, smearing)))
        if a is not None:
            a = a[a[:, 0] == a[0, 0]]  # first temperature
            for j, comp in enumerate(TENSOR[:3]):
                add('%s_%s' % (key, comp), a[:, 1], a[:, 2 + j])
    a = _read(os.path.join(out, 'hall_trace_%s_0.dat' % smearing))
    if a is not None:
        a = a[a[:, 0] == a[0, 0]]
        add('hall_trace', a[:, 1], a[:, 2])

    # do_conductivity divides by the number of k-points but not by the cell
    # volume, so the file is per supercell
    xx = _read(os.path.join(out, 'cond_xx_0.dat'))
    if xx is not None:
        add('cond_xx', xx[:, 0], xx[:, 1] / N)
        xy = _read(os.path.join(out, 'cond_xy_0.dat'))
        if xy is not None:
            add('cond_xy', xy[:, 0], xy[:, 1] / N, scale=float(np.abs(xx[:, 1]).max()) / N)

    for path in sorted(glob.glob(os.path.join(out, 'doping_*.dat'))):
        a = _read(path)
        add('doping_mu', a[:, 0], a[:, 1])

    ehi = (rec.get('solver') or {}).get('ehi')
    xmax = None if ehi is None else ehi - EPS_MARGIN
    for stem in ['epsi'] + (OPTICS if optics else []):
        for comp in TENSOR[:3]:
            a = _read(os.path.join(out, '%s_%s.dat' % (stem, comp)))
            if a is not None:
                add('%s_%s' % (stem, comp), a[:, 0], a[:, 1], xmax=xmax)

    path = os.path.join(out, 'effective_masses_0.dat')
    if os.path.exists(path):
        a = np.loadtxt(path, skiprows=2, ndmin=2)
        inv = np.nan_to_num(1.0 / a[:, 7], nan=0.0, posinf=0.0, neginf=0.0)
        add('inv_mass_dos', MASS_GRID, _smeared(a[:, 3], inv) / (int(rec['mesh']['nkpnts']) * N))
    return curves


def _smeared(E, weights):
    """``sum_s weights_s g(MASS_GRID - E_s)``, by binning onto the grid and
    convolving with the Gaussian (a million states fit in memory this way)."""
    step = MASS_GRID[1] - MASS_GRID[0]
    pad = int(np.ceil(5 * MASS_SMEARING / step))
    edges = MASS_GRID[0] + step * (np.arange(len(MASS_GRID) + 2 * pad + 1) - pad - 0.5)
    hist, _ = np.histogram(E, bins=edges, weights=weights)
    t = step * np.arange(-pad, pad + 1)
    kernel = np.exp(-((t / MASS_SMEARING) ** 2) / 2) / (np.sqrt(2 * np.pi) * MASS_SMEARING)
    return np.convolve(hist, kernel, mode='same')[pad:-pad]


def deviation(curve, ref):
    """Relative L2 deviation of ``curve`` from ``ref`` on their shared x range.

    The norm is ``ref['scale']`` when set, else the RMS of the reference on
    that range; the range is cut at the smaller ``xmax`` of the two.
    """
    x, y = curve['x'], curve['y']
    xr, yr = ref['x'], ref['y']
    lo = max(x.min(), xr.min())
    hi = min(x.max(), xr.max())
    for cut in (curve.get('xmax'), ref.get('xmax')):
        if cut is not None:
            hi = min(hi, cut)
    mask = (xr >= lo) & (xr <= hi)
    if mask.sum() < 2:
        return np.nan
    yi = np.interp(xr[mask], x, y)
    scale = ref.get('scale') or curve.get('scale')
    norm = scale if scale else np.sqrt(np.mean(yr[mask] ** 2))
    return float(np.sqrt(np.mean((yi - yr[mask]) ** 2)) / norm) if norm > 0 else np.nan
