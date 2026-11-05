"""Per-primitive-cell observables of one benchmark case, read from its output files.

A cell doubled ``d`` times is still the same crystal, so every observable
here is a curve ``y(x)`` that must not change with ``d``.  :func:`load` reads the curves
of one case directory and normalizes them per primitive cell (``N = 2**d``):

==================  =========================================  ======================
name                source                                     normalization
==================  =========================================  ======================
``dos``             ``dosdk_0.dat``                            / N
``pdos_s/p/d``      ``*_pdosdk_0.dat``, summed by orbital l    / N
``sigma_xx`` etc.   ``sigma/Seebeck/kappa/PF<smearing>_0.dat``  none (per volume)
``hall_trace``      ``hall_trace_<smearing>_0.dat``            none (per volume)
``hall_xyz`` etc.   ``hall_<smearing>_0.dat``                  none (per volume)
``hall_zero``       ``hall_<smearing>_0.dat``                  none (per volume)
``nernst_xyz/yxz``  ``nernst_<smearing>_0.dat``                none (per volume)
``nernst_zero``     ``nernst_<smearing>_0.dat``                none (per volume)
``cond_xx/xy``      ``cond_xx_0.dat``, ``cond_xy_0.dat``       / N (k-averaged, not per volume)
``lr_cond_xx/xy``   ``Cond_xx.dat``, ``Cond_xy.dat``           none (per volume)
``lr_ahc_xy``       ``AHC_xy.dat``                             none (per volume)
``lr_shc_*``        ``SHC_EVEN_z_xy.dat`` ...                  none (per volume)
``lr_ree_*``        ``REE_EVEN_zz.dat`` ...                    / N (k-averaged, not per volume)
``berry_shc_*``     ``shcEf_z_xy.dat`` ... (``spin_Hall``)     none (per volume)
``berry_ohc_*``     ``ohcEf_z_xy.dat`` ... (``orbital_Hall``)  none (per volume)
``berry_ahc_*``     ``ahcEf_xy.dat`` ... (``anomalous_Hall``)  none (per volume)
``scd/ocd/mcd_*``   ``SCDr/SCDi/OCDr/OCDi/MCDr/MCDi_*.dat``    none (per volume)
``ree_spin_*``      ``Ekai_zz.dat`` ... (``rashba_edelstein``) none (a ratio)
``ree_orbital_*``   ``orbital_Ekai_zz.dat`` ...                none (a ratio)
``spin_texture``    ``spin_text_band_<i>.npz``, d = 0 only     RMS |<S>| per band
``bands``           ``bands_0.dat``, d = 0 only                the bands one after another
``doping_mu``       ``doping_*.dat``                           none (mu against T)
``epsi_xx`` etc.    ``epsi_xx.dat`` ...                        none (per volume)
``epsr_xx`` etc.    ``epsr/eels/refl/nref/kref/alpha/sigmar/    none (per volume)
                    sigmai_xx.dat`` ..., only with ``optics``
``inv_mass_dos``    ``effective_masses_0.dat``                 / (N nk), see below
``inv_mass_trace``  ``effective_masses_0.dat``                 / (N nk), see below
==================  =========================================  ======================

``inv_mass_dos`` turns the per-state effective masses into a curve:
:math:`\\sum_{nk} g(E - E_{nk}) / m^{dos}_{nk}`, Gaussian ``g`` of width
``MASS_SMEARING`` eV, over all states and divided by the total number of
k-points per primitive cell.  The states of a doubled cell on its mesh are
exactly those of the primitive cell on its own mesh, so this curve is the same
for every ``d``, whereas the file itself is not comparable line by line.
``inv_mass_trace`` weights each state by the mean of its three inverse
principal masses instead, :math:`\\tfrac13 \\mathrm{tr}\\, m^{-1}_{nk} =
\\tfrac13 \\nabla^2 E_{nk} / \\hbar^2`.  The file lists the principal masses
in no fixed order, so only symmetric functions of them compare across runs;
this one stays finite where a principal curvature passes through zero, which
makes ``1/m_dos`` jump.

Components that symmetry zeroes are measured on the scale of their
symmetry-allowed partner (:func:`symmetry_zeros`): by default, in every system
of the benchmark, the off-diagonal transport components (``sigma_xy`` etc.)
on ``*_xx``, ``cond_xy`` on ``cond_xx``, and ``lr_cond_xy`` and ``lr_ahc_xy``
on ``lr_cond_xx`` (the anomalous Hall conductivity vanishes for any crystal
with time-reversal symmetry).  ``hall_zero`` and ``nernst_zero`` are the RMS
over the components with a repeated index (6 of the 9 written Hall
components, 21 of the 27 Nernst ones), on the scale of ``hall_xyz`` and
``nernst_xyz``.  A system adds its own zeros (``SYSTEM['zeros']``) and lists
the defaults that do not vanish for it (``SYSTEM['nonzero']``: the anomalous
Hall conductivity of Fe).

The PDOS by orbital type is not read when two orbitals share a file prefix,
as both spinor copies of an orbital do with spin-orbit coupling: the
undoubled cell then writes the second over the first.

``epsi_*`` of an energy-window run is cut at the window
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
# the columns of hall_<smearing>_0.dat after T and mu: R_ijk antisymmetrized
# in ij; the first three have three different indices, the others vanish in
# a cubic crystal
HALL = ['xyz', 'xzy', 'yzx', 'xyx', 'xyy', 'xzx', 'xzz', 'yzy', 'yzz']
# the first six columns of nernst_<smearing>_0.dat; the other 21 vanish in a
# cubic crystal
NERNST = ['xyz', 'yxz', 'xzy', 'zxy', 'yzx', 'zyx']
OPTICS = ['epsr', 'eels', 'refl', 'nref', 'kref', 'alpha', 'sigmar', 'sigmai']
# per-volume files, two columns (x, y): file prefix -> curve prefix.  The
# rest of the file stem, lowercased, completes the curve name, so
# 'SHC_EVEN_z_xy.dat' is 'lr_shc_even_z_xy'
PER_VOLUME = {
    # linear_response: x is the chemical potential
    'Cond_': 'lr_cond_',
    'AHC_': 'lr_ahc_',
    'SHC_': 'lr_shc_',
    # spin_Hall, orbital_Hall, anomalous_Hall (Berry curvature): x is the
    # chemical potential
    'shcEf_': 'berry_shc_',
    'ohcEf_': 'berry_ohc_',
    'ahcEf_': 'berry_ahc_',
    # and their do_ac parts: x is the frequency
    'SCDr_': 'scd_re_',
    'SCDi_': 'scd_im_',
    'OCDr_': 'ocd_re_',
    'OCDi_': 'ocd_im_',
    'MCDr_': 'mcd_re_',
    'MCDi_': 'mcd_im_',
}
# per-supercell files (k-averaged, not divided by the cell volume), divided
# by N: the Rashba-Edelstein response of linear_response
PER_SUPERCELL = {'REE_': 'lr_ree_'}
# components that vanish by symmetry in every system of the benchmark, each
# measured on the scale (largest |y|) of a symmetry-allowed partner: the
# off-diagonal Boltzmann tensors, the Hall and Nernst components with a
# repeated index, the off-diagonal conductivities, and the anomalous Hall
# conductivity, which time reversal zeroes.  A system adds its own in
# SYSTEM['zeros'] and removes the ones that do not vanish in
# SYSTEM['nonzero'] (the anomalous Hall conductivity of Fe)
DEFAULT_ZEROS = {'%s_%s' % (key, comp): '%s_xx' % key for key in TRANSPORT for comp in TENSOR[3:]}
DEFAULT_ZEROS.update(
    {
        'hall_zero': 'hall_xyz',
        'nernst_zero': 'nernst_xyz',
        'cond_xy': 'cond_xx',
        'lr_cond_xy': 'lr_cond_xx',
        'lr_ahc_xy': 'lr_cond_xx',
    }
)


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
    if not labels or len(set(labels)) < len(labels):
        # with spin-orbit coupling both spinor copies of an orbital share one
        # file prefix, so the undoubled cell's second file overwrites the
        # first and the primitive-cell PDOS is incomplete: not comparable
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
            for j, comp in enumerate(TENSOR):
                add('%s_%s' % (key, comp), a[:, 1], a[:, 2 + j])
    a = _read(os.path.join(out, 'hall_trace_%s_0.dat' % smearing))
    if a is not None:
        a = a[a[:, 0] == a[0, 0]]
        add('hall_trace', a[:, 1], a[:, 2])
    a = _read(os.path.join(out, 'hall_%s_0.dat' % smearing))
    if a is not None:
        a = a[a[:, 0] == a[0, 0]]
        for j, comp in enumerate(HALL[:3]):
            add('hall_' + comp, a[:, 1], a[:, 2 + j])
        add('hall_zero', a[:, 1], np.sqrt(np.mean(a[:, 5:] ** 2, axis=1)))
    a = _read(os.path.join(out, 'nernst_%s_0.dat' % smearing))
    if a is not None:
        a = a[a[:, 0] == a[0, 0]]
        for j, comp in enumerate(NERNST[:2]):
            add('nernst_' + comp, a[:, 1], a[:, 2 + j])
        add('nernst_zero', a[:, 1], np.sqrt(np.mean(a[:, 8:] ** 2, axis=1)))

    # do_conductivity divides by the number of k-points but not by the cell
    # volume, so the file is per supercell
    for path in sorted(glob.glob(os.path.join(out, 'cond_??_0.dat'))):
        a = _read(path)
        add('cond_' + os.path.basename(path)[5:7], a[:, 0], a[:, 1] / N)

    # linear_response and the Berry-curvature Hall routines divide by the
    # k-points and the supercell volume, so these files are per volume
    # already, except the Rashba-Edelstein response, which is per supercell
    for table, norm in ((PER_VOLUME, 1), (PER_SUPERCELL, N)):
        for prefix, name in table.items():
            for path in sorted(glob.glob(os.path.join(out, prefix + '*.dat'))):
                a = _read(path)
                stem = os.path.basename(path)[len(prefix) : -len('.dat')]
                add(name + stem.lower(), a[:, 0], a[:, 1] / norm)

    # rashba_edelstein: the spin (orbital) accumulation over the charge
    # current, a ratio, so the same per cell at every size
    for kind, prefix in (('spin', ''), ('orbital', 'orbital_')):
        for path in sorted(glob.glob(os.path.join(out, prefix + 'Ekai_??.dat'))):
            a = _read(path)
            comp = os.path.basename(path)[len(prefix) + 5 : len(prefix) + 7]
            add('ree_%s_%s' % (kind, comp), a[:, 0], a[:, 1])

    # the band path, d = 0 only (doubling folds it): the bands one after the
    # other, so the deviation compares the bands both files have (the sparse
    # file holds the well-projected ones, the dense file all nawf), and its
    # scale of 1 eV makes it the RMS band difference in eV
    a = _read(os.path.join(out, 'bands_0.dat')) if N == 1 else None
    if a is not None:
        E = a[:, 1:]
        add('bands', np.arange(E.size), E.T.ravel(), scale=1.0)

    # spin_texture on the mesh: one file per band of the window, folded by the
    # doubling, so only the undoubled cell is read.  The curve is the RMS of
    # |<S>| over the mesh against the band's place in the window
    if N == 1:
        paths = glob.glob(os.path.join(out, 'spin_text_band_*.npz'))
        if paths:
            paths.sort(key=lambda p: int(p.rsplit('_', 1)[1][: -len('.npz')]))
            rms = [
                np.sqrt(np.mean(np.sum(np.abs(np.load(p)['spinband']) ** 2, axis=-1)))
                for p in paths
            ]
            add('spin_texture', np.arange(len(rms)), np.array(rms))

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
        norm = int(rec['mesh']['nkpnts']) * N
        inv = np.nan_to_num(1.0 / a[:, 7], nan=0.0, posinf=0.0, neginf=0.0)
        add('inv_mass_dos', MASS_GRID, _smeared(a[:, 3], inv) / norm)
        inv = np.nan_to_num(1.0 / a[:, 4:7], nan=0.0, posinf=0.0, neginf=0.0).mean(axis=1)
        add('inv_mass_trace', MASS_GRID, _smeared(a[:, 3], inv) / norm)
    zeros = symmetry_zeros(rec['case'].get('system') or {})
    for name, partner in zeros.items():
        if name in curves and partner in curves:
            curves[name]['scale'] = float(np.abs(curves[partner]['y']).max())
    return curves


def symmetry_zeros(system):
    """``{curve: partner}`` of a system: :data:`DEFAULT_ZEROS` with its own
    ``zeros`` added and its ``nonzero`` curves removed.  A record made before
    the systems existed (Si) has neither, and gets the defaults."""
    zeros = dict(DEFAULT_ZEROS)
    zeros.update(system.get('zeros') or {})
    for name in system.get('nonzero') or []:
        zeros.pop(name, None)
    return zeros


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
    that range; the range is cut at the smaller ``xmax`` of the two.  Points
    that are not finite in either curve are left out (the sparse band path
    holds NaN above the usable top).
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
    yr = yr[mask]
    ok = np.isfinite(yi) & np.isfinite(yr)
    if ok.sum() < 2:
        return np.nan
    yi, yr = yi[ok], yr[ok]
    scale = ref.get('scale') or curve.get('scale')
    norm = scale if scale else np.sqrt(np.mean(yr**2))
    return float(np.sqrt(np.mean((yi - yr) ** 2)) / norm) if norm > 0 else np.nan
