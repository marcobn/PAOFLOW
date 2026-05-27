#######################################################################
# Fit UPF radial pseudowavefunctions with gaussian orbitals
# Authored: Davide Ceresoli - May 2016
# Revised:  Frank Cerasoli - June 2022
#
# Notes:
# - UPFv1 files must be embedded in <UPF version="1.0">...</UPF> element
# - contraction coefficients for d and f orbitals correspond to the
#   cubic harmonics
#######################################################################

"""upf_gaussfit — Gaussian fitting of UPF radial pseudo-wavefunctions.

This module fits the radial pseudo-wavefunctions :math:`\\chi_{nl}(r)` stored
in Quantum ESPRESSO UPF pseudopotential files (versions 1 and 2) with
contracted Gaussian-type orbital (cGTO) expansions.  The resulting basis
is used by the ACBN0 and eACBN0 workflows, where two-electron Coulomb
repulsion integrals computed in :mod:`~PAOFLOW.defs.pyints` require
analytic GTO representations of the projector functions.

Fitting procedure
-----------------
For each pseudo-wavefunction labelled :math:`n l` the radial part is
approximated as

.. math::

    \\chi_{nl}^\\text{GTO}(r) =
        \\sum_{j=1}^{N_\\zeta} c_j\\, r^l\\, e^{-\\zeta_j r^2},
    \\qquad \\zeta_j = \\alpha / \\beta^{j-1}

where :math:`(\\alpha, \\beta)` control the geometric spacing of the
exponents and :math:`c_j` are the contraction coefficients.  The two-
parameter exponential progression means only :math:`N_\\zeta + 2` free
parameters are needed regardless of the contraction length.

The fit minimises the least-squares residual
:math:`\\sum_r [\\chi_{nl}^\\text{ref}(r) - r\\, \\chi_{nl}^\\text{GTO}(r)]^2`
using either the Levenberg–Marquardt algorithm (:func:`fit` with
``least_squares=True``, default) or conjugate-gradient minimisation.
Starting from :math:`N_\\zeta = 2` the multiplicity is increased by one
until the residual norm falls below ``threshold`` or :math:`N_\\zeta = 5`.

Angular-momentum conventions
-----------------------------
Contraction coefficients output by :func:`build_basis_dict` and
:func:`write_basis_file` are expressed in the **cubic (real) harmonic**
basis with the following conventions for each :math:`l`:

* :math:`l=0` (s): single :math:`(0,0,0)` Cartesian GTO.
* :math:`l=1` (p): :math:`p_z, p_y, p_x` in standard :math:`(n_x,n_y,n_z)` form.
* :math:`l=2` (d): :math:`d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy}` as
  linear combinations of Cartesian GTOs (standard :math:`\\sqrt{3}` prefactors).
* :math:`l=3` (f): :math:`f_{z^3}, f_{xz^2}, f_{yz^2}, f_{z(x^2-y^2)},
  f_{xyz}, f_{x(x^2-3y^2)}, f_{y(3x^2-y^2)}` with appropriate prefactors.

These Cartesian tuples are consumed directly by
:func:`~PAOFLOW.defs.pyints.contr_coulomb` to evaluate four-centre
integrals.

Main entry point
----------------
:func:`gaussian_fit`
    Read a UPF file, fit every pseudo-wavefunction, and return the
    atomic number together with the full Cartesian-GTO basis dictionary.
    Automatically increments :math:`N_\\zeta` until convergence.

I/O helpers
-----------
:func:`read_upf`
    Parse the ``<PP_PSWFC>`` section of a UPF v1 or v2 XML file and
    return the radial grid, integration weights, labels, angular momenta,
    and wavefunction arrays.
:func:`read_atom_no_xml`
    Extract the element symbol and atomic number from a UPF XML header.
:func:`write_basis_file`
    Write the fitted basis to a Python-importable ``.py`` file containing
    a ``basis_data`` dict keyed by atomic number.
:func:`build_basis_dict`
    Convert per-orbital coefficient/exponent lists to the nested
    Cartesian-GTO tuple structure expected by :mod:`~PAOFLOW.defs.pyints`.

Low-level fitting helpers
--------------------------
:func:`gto`
    Evaluate the contracted radial GTO :math:`\\chi^\\text{GTO}(r)` for
    given exponent/coefficient parameters.
:func:`target` / :func:`target_squared`
    Residual vector and sum-of-squares objective for ``scipy`` optimisers.
:func:`fit`
    Fit a single pseudo-wavefunction channel with :math:`N_\\zeta`
    primitive Gaussians; returns coefficients, exponents, and an exit code.

Utilities
---------
:data:`spn_map`
    ``dict`` mapping element symbol → atomic number for Z = 1–112.
:func:`get_atom_no`
    Look up atomic number from element symbol; raises if symbol unknown.
:func:`fact2`
    Recursive double factorial :math:`n!!` used in GTO normalisation.
"""

import numpy as np

spn_map = {
    'H': 1,
    'He': 2,
    'Li': 3,
    'Be': 4,
    'B': 5,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'Ne': 10,
    'Na': 11,
    'Mg': 12,
    'Al': 13,
    'Si': 14,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Ar': 18,
    'K': 19,
    'Ca': 20,
    'Sc': 21,
    'Ti': 22,
    'V': 23,
    'Cr': 24,
    'Mn': 25,
    'Fe': 26,
    'Co': 27,
    'Ni': 28,
    'Cu': 29,
    'Zn': 30,
    'Ga': 31,
    'Ge': 32,
    'As': 33,
    'Se': 34,
    'Br': 35,
    'Kr': 36,
    'Rb': 37,
    'Sr': 38,
    'Y': 39,
    'Zr': 40,
    'Nb': 41,
    'Mo': 42,
    'Tc': 43,
    'Ru': 44,
    'Rh': 45,
    'Pd': 46,
    'Ag': 47,
    'Cd': 48,
    'In': 49,
    'Sn': 50,
    'Sb': 51,
    'Te': 52,
    'I': 53,
    'Xe': 54,
    'Cs': 55,
    'Ba': 56,
    'La': 57,
    'Ce': 58,
    'Pr': 59,
    'Nd': 60,
    'Pm': 61,
    'Sm': 62,
    'Eu': 63,
    'Gd': 64,
    'Tb': 65,
    'Dy': 66,
    'Ho': 67,
    'Er': 68,
    'Tm': 69,
    'Yb': 70,
    'Lu': 71,
    'Hf': 72,
    'Ta': 73,
    'W': 74,
    'Re': 75,
    'Os': 76,
    'Ir': 77,
    'Pt': 78,
    'Au': 79,
    'Hg': 80,
    'Tl': 81,
    'Pb': 82,
    'Bi': 83,
    'Po': 84,
    'At': 85,
    'Rn': 86,
    'Fr': 87,
    'Ra': 88,
    'Ac': 89,
    'Th': 90,
    'Pa': 91,
    'U': 92,
    'Np': 93,
    'Pu': 94,
    'Am': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109,
    'Ds': 110,
    'Rg': 111,
    'Cn': 112,
}


def get_atom_no(n):
    if n not in spn_map:
        raise Exception(f'Invalid atomic symbol: {n}')
    return spn_map[n]


# Double factorial (n!!)
def fact2(n):
    if n <= 1:
        return 1
    return n * fact2(n - 2)


# ======================================================================
# GTO orbital
# ======================================================================
def gto(r, l, params):
    alpha, beta = params[:2]
    coeffs = params[2:]

    gto = np.zeros_like(r)
    for j, c in enumerate(coeffs):
        zeta = alpha / beta**j
        i = np.where(zeta * r**2 > -12)
        gto[i] += c * r[i] ** l * np.exp(-zeta * r[i] ** 2)

    return gto


# ======================================================================
# Target function whose least square has to be minimized
# ======================================================================
def target(params, r, rab, wfc, l):
    return wfc - r * gto(r, l, params)


def target_squared(params, r, rab, wfc, l):
    return np.sum(target(params, r, rab, wfc, l) ** 2)


# ======================================================================
# Fit radial wfc with gaussians
# ======================================================================
def fit(nzeta, label, l, r, rab, wfc, threshold, least_squares=True):
    if len(wfc) != len(r):
        raise Exception('wfc and r have different dimensions.')

    wfc, r = np.array(wfc), np.array(r)

    # Initial alpha and beta
    params0 = np.array([4.0, 4.0] + [1.0] * nzeta)

    # Least squares
    if least_squares:
        from scipy.optimize import leastsq

        params, fc, info, msg, ier = leastsq(
            target,
            params0,
            full_output=1,
            args=(r, rab, wfc, l),
            maxfev=50000,
            ftol=1e-10,
            xtol=1e-10,
        )
        print(f'INFO: ier={ier} - mesg={msg}')
        print('INFO: nfev={}'.format(info['nfev']))
        print('INFO: |fvec|^2={}'.format((info['fvec'] ** 2).sum()))
        if ier not in [1, 2, 3, 4]:
            print('ERROR: Gaussion could not be fit to this state.')

    # Minimize
    else:
        from scipy.optimize import minimize

        opt = minimize(target_squared, params0, args=(r, rab, wfc, l), method='CG', tol=1e-10)
        params = opt.x
        if not opt.success:
            print('ERROR: opt.status={}'.format(opt.status))
            print('ERROR: opt.message={}'.format(opt.message))
            print('ERROR: opt.nfev={}'.format(opt.nfev))
            print('ERROR: opt.fun={}'.format(opt.fun))

    alpha, beta = params[:2]
    print(f'alpha = {alpha}, beta = {beta}')

    expon = []
    coeffs = np.sqrt(fact2(2 * l + 1) / (4 * np.pi)) * params[2:]
    for j, c in enumerate(coeffs):
        zeta = alpha / beta**j
        expon.append(zeta)
        print(f'coeff = {c}, zeta = {zeta}')

    res = target_squared(params, r, rab, wfc, l)
    print(f'Fit result: {res}')

    exit_code = 0 if np.abs(res) <= threshold else 1

    return coeffs, expon, exit_code


# ======================================================================
# Construct basis string for orbitals and write it to file
# ======================================================================
def build_basis_dict(labels, ls, coefficients, exponents):
    basis = []

    for il, _ in enumerate(labels):
        l = ls[il]
        expon = exponents[il]
        coeffs = coefficients[il]

        lbasis = []

        if l == 0:
            ibasis = []
            for i, c in enumerate(coeffs):
                ibasis.append((0, 0, 0, c, expon[i]))
            lbasis.append(ibasis)

        elif l == 1:
            for n in range(3):
                ibasis = []
                lind = [0] * 3
                lind[2 - n] = 1
                for i, c in enumerate(coeffs):
                    ibasis.append((*lind, c, expon[i]))
                lbasis.append(ibasis)

        elif l == 2:
            # 1/(2*sqrt(3))*(2*z2 - x2 - y2)
            ibasis = []
            for n in range(3):
                # ibasis = []
                lind = [0] * 3
                lind[2 - n] = 2
                fact = (1 if n == 0 else -0.5) / np.sqrt(3)
                for i, c in enumerate(coeffs):
                    ibasis.append((*lind, fact * c, expon[i]))
            lbasis.append(ibasis)

            # xz
            lbasis.append([(1, 0, 1, c, expon[i]) for i, c in enumerate(coeffs)])

            # yz
            lbasis.append([(0, 1, 1, c, expon[i]) for i, c in enumerate(coeffs)])

            # 1/2 * (x2 - y2)
            ibasis = []
            for i, c in enumerate(coeffs):
                ibasis.append((2, 0, 0, 0.5 * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 2, 0, -0.5 * c, expon[i]))
            lbasis.append(ibasis)

            # xy
            lbasis.append([(1, 1, 0, c, expon[i]) for i, c in enumerate(coeffs)])

        elif l == 3:
            # fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x3-3y2), fy(3x2-y2)

            # 1/(2*sqrt(15)) * z*(2*z2 - 3*x2 - 3*y2)
            fact = 0.5 / np.sqrt(15)
            ibasis = []
            for i, c in enumerate(coeffs):
                ibasis.append((0, 0, 3, 2 * fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((2, 0, 1, -3 * fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 2, 1, -3 * fact * c, expon[i]))
            lbasis.append(ibasis)

            # 1/(2*sqrt(10)) * x*(4*z2 - x2 - y2)
            ibasis = []
            fact = 0.5 / np.sqrt(10)
            for i, c in enumerate(coeffs):
                ibasis.append((1, 0, 2, 4 * fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 0, 3, -fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((1, 2, 0, -fact * c, expon[i]))
            lbasis.append(ibasis)

            # 1/(2*sqrt(10)) * y*(4*z2 - x2 - y2)
            ibasis = []
            fact = 0.5 / np.sqrt(10)
            for i, c in enumerate(coeffs):
                ibasis.append((0, 1, 2, 4 * fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((2, 1, 0, -fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 3, 0, -fact * c, expon[i]))
            lbasis.append(ibasis)

            # 1/2 * z*(x2 - y2)
            ibasis = []
            fact = 0.5
            for i, c in enumerate(coeffs):
                ibasis.append((2, 0, 1, fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 2, 1, -fact * c, expon[i]))
            lbasis.append(ibasis)

            # x*y*z
            for i, c in enumerate(coeffs):
                lbasis.append([(1, 1, 1, c, expon[i]) for i, c in enumerate(coeffs)])

            # 1/(2*sqrt(6)) * x*(x2 - 3*y2)
            ibasis = []
            fact = 0.5 / np.sqrt(6)
            for i, c in enumerate(coeffs):
                ibasis.append((3, 0, 0, fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((1, 2, 0, -3 * fact * c, expon[i]))
            lbasis.append(ibasis)

            # 1/(2*sqrt(6)) * y*(3*x2 - y2)
            ibasis = []
            fact = 0.5 / np.sqrt(6)
            for i, c in enumerate(coeffs):
                ibasis.append((2, 1, 0, 3 * fact * c, expon[i]))
            for i, c in enumerate(coeffs):
                ibasis.append((0, 3, 0, -fact * c, expon[i]))
            lbasis.append(ibasis)

        basis.append(lbasis)

    return basis


def write_basis_file(fname, atom_no, labels, ls, coefficients, exponents):
    rstr = f'basis_data = {{ {atom_no} : [\n'

    for il, label in enumerate(labels):
        l = ls[il]
        expon = exponents[il]
        coeffs = coefficients[il]

        rstr += f'# label= {label} l= {l}\n[[\n'

        fcon = '],  [\n'
        fbuf = '   {},\n'
        fpat = '({},{},{},{:.10f},{:.10f})'
        fline = lambda x, y, z, a, b: fbuf.format(fpat.format(x, y, z, a, b))

        if l == 0:
            for i, c in enumerate(coeffs):
                rstr += fline(0, 0, 0, c, expon[i])

        elif l == 1:
            for n in range(3):
                lind = [0] * 3
                lind[2 - n] = 1
                for i, c in enumerate(coeffs):
                    rstr += fline(*lind, c, expon[i])
                if n < 2:
                    rstr += fcon

        elif l == 2:
            # 1/(2*sqrt(3))*(2*z2 - x2 - y2)
            for n in range(3):
                lind = [0] * 3
                lind[2 - n] = 2
                fact = (1 if n == 0 else -0.5) / np.sqrt(3)
                for i, c in enumerate(coeffs):
                    rstr += fline(*lind, fact * c, expon[i])
            rstr += fcon

            # xz
            for i, c in enumerate(coeffs):
                rstr += fline(1, 0, 1, c, expon[i])
            rstr += fcon

            # yz
            for i, c in enumerate(coeffs):
                rstr += fline(0, 1, 1, c, expon[i])
            rstr += fcon

            # 1/2 * (x2 - y2)
            for i, c in enumerate(coeffs):
                rstr += fline(2, 0, 0, 0.5 * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 2, 0, -0.5 * c, expon[i])
            rstr += fcon

            # xy
            for i, c in enumerate(coeffs):
                rstr += fline(1, 1, 0, c, expon[i])

        elif l == 3:
            # fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x3-3y2), fy(3x2-y2)

            # 1/(2*sqrt(15)) * z*(2*z2 - 3*x2 - 3*y2)
            fact = 0.5 / np.sqrt(15)
            for i, c in enumerate(coeffs):
                rstr += fline(0, 0, 3, 2 * fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(2, 0, 1, -3 * fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 2, 1, -3 * fact * c, expon[i])
            rstr += fcon

            # 1/(2*sqrt(10)) * x*(4*z2 - x2 - y2)
            fact = 0.5 / np.sqrt(10)
            for i, c in enumerate(coeffs):
                rstr += fline(1, 0, 2, 4 * fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 0, 3, -fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(1, 2, 0, -fact * c, expon[i])
            rstr += fcon

            # 1/(2*sqrt(10)) * y*(4*z2 - x2 - y2)
            fact = 0.5 / np.sqrt(10)
            for i, c in enumerate(coeffs):
                rstr += fline(0, 1, 2, 4 * fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(2, 1, 0, -fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 3, 0, -fact * c, expon[i])
            rstr += fcon

            # 1/2 * z*(x2 - y2)
            fact = 0.5
            for i, c in enumerate(coeffs):
                rstr += fline(2, 0, 1, fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 2, 1, -fact * c, expon[i])
            rstr += fcon

            # x*y*z
            for i, c in enumerate(coeffs):
                rstr += fline(1, 1, 1, c, expon[i])
            rstr += fcon

            # 1/(2*sqrt(6)) * x*(x2 - 3*y2)
            fact = 0.5 / np.sqrt(6)
            for i, c in enumerate(coeffs):
                rstr += fline(3, 0, 0, fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(1, 2, 0, -3 * fact * c, expon[i])
            rstr += fcon

            # 1/(2*sqrt(6)) * y*(3*x2 - y2)
            fact = 0.5 / np.sqrt(6)
            for i, c in enumerate(coeffs):
                rstr += fline(2, 1, 0, 3 * fact * c, expon[i])
            for i, c in enumerate(coeffs):
                rstr += fline(0, 3, 0, -fact * c, expon[i])
            rstr += fcon

        rstr += ']],\n'
    rstr = rstr[:-1] + ']}\n'

    with open(fname, 'w') as f:
        f.write(rstr)

    print(f'INFO: File {fname} created.\n')


def read_atom_no_xml(upf_version, root):
    ele = None
    text = root.find('PP_HEADER')
    if upf_version == 1:
        text = text.text.split()
        ind = text.index('Element')
        ele = text[ind - 1].strip()
    elif upf_version == 2:
        ele = text.attrib['element'].strip()
    else:
        raise Exception('ERROR: Supported UPF version are v1 and v2')

    no = get_atom_no(ele)
    print(f'INFO: element={ele}, atomic number={no}')
    return ele, no


def read_upf(upf_version, root):
    ls = []
    wfcs = []
    labels = []
    text = root.find('PP_MESH/PP_R').text
    r = np.array([float(v) for v in text.split()])
    text = root.find('PP_MESH/PP_RAB').text
    rab = np.array([float(v) for v in text.split()])

    if upf_version == 1:
        from io import StringIO

        chi = root.find('PP_PSWFC')
        if chi is None:
            raise Exception('ERROR: Cannot locate PP_PSWFC tag.')

        nlines = r.shape[0] // 4
        if r.shape[0] % 4 != 0:
            nlines += 1

        text = StringIO(chi.text)
        line = text.readline()
        while line != '':
            if line == '\n':
                continue

            label, l, occ, _ = line.split()
            l, occ = int(l), float(occ)

            wfc = []
            for _ in range(nlines):
                wfc += list(map(float, text.readline().split()))

            ls.append(l)
            wfcs.append(wfc)
            labels.append(label)

            line = text.readline()

    elif upf_version == 2:
        ind = 1
        fstr = 'PP_PSWFC/PP_CHI.{}'
        chi = root.find(fstr.format(ind))
        while chi is not None:
            label = chi.attrib['label']
            l = int(chi.attrib['l'])
            wfc = [float(v) for v in chi.text.split()]
            if len(wfc) != r.shape[0]:
                msg = 'ERROR: wfc and radial grid have different dimension'
                raise Exception(msg)

            ls.append(l)
            wfcs.append(wfc)
            labels.append(label)

            ind += 1
            chi = root.find(fstr.format(ind))

    else:
        raise Exception('ERROR: Supported UPF version are v1 and v2')

    wfcs = np.array(wfcs)
    for i, w in enumerate(wfcs):
        l = ls[i]
        label = labels[i]
        norm = np.sum(rab * w**2)
        print(f'INFO: Fitting pswfc {label} l={l} norm={norm}')

    return r, rab, labels, ls, wfcs


def gaussian_fit(xml_file, threshold=0.5):
    from xml.etree import ElementTree as ET

    atno = -1
    nzeta = 2
    basis = None
    optimized = False
    while not optimized and nzeta < 6:
        root = None
        try:
            print(f'INFO: Fitting file {xml_file} with {nzeta} gaussians')
            with open(xml_file, 'r') as f:
                xml_content = f.read()
                root = ET.fromstring(xml_content)

            upf_version = int(root.attrib['version'].split('.')[0])
        except Exception as e:
            print(f'ERROR: Could not read the xml file: {xml_file}')
            raise e

        ele, atno = read_atom_no_xml(upf_version, root)
        r, rab, labels, ls, wfcs = read_upf(upf_version, root)

        failed = False
        coeffs, exponents = [], []
        for i, lab in enumerate(labels):
            coef, expon, exit_code = fit(nzeta, lab, ls[i], r, rab, wfcs[i], threshold)
            if exit_code == 0:
                coeffs.append(coef)
                exponents.append(expon)
            else:
                failed = True
                break

        if failed:
            nzeta += 1
            continue

        optimized = True
        basis = build_basis_dict(labels, ls, coeffs, exponents)

    if nzeta >= 6:
        raise Exception('ERROR: Could not optimize the wfcs')

    if atno == -1:
        raise Exception('ERROR: Could not determine atomic information')

    return atno, basis
