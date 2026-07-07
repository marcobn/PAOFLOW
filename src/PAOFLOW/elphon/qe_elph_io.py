"""Readers for Quantum ESPRESSO electron-phonon coupling output (Phase 1).

These parsers ingest the coarse-grid coupling that ``ph.x``
(``electron_phonon='interpolated'``) and the ``lambda.x`` post-processing use,
so that PAOFLOW's isotropic Eliashberg engine
(:func:`PAOFLOW.elphon.eph_kq.eliashberg_from_modes`) can be validated against
QE's ``lambda.x`` on the *same* electron-phonon matrix elements.  This isolates
the property calculation (``a2F`` / ``lambda`` / ``Tc``) from the way the
matrix elements are obtained (finite differences of the PAO Hamiltonian in the
later phase).

Three files are parsed:

* ``lambda.in`` -- the ``lambda.x`` driver input: ``emax``, ``degauss``, the
  smearing method, the irreducible q-points with their weights, the list of
  per-q ``elph.inp_lambda.N`` files and the Coulomb pseudopotential ``mu*``.
* ``elph.inp_lambda.N`` -- one per irreducible q-point: the squared phonon
  frequencies ``omega^2`` (Ry^2) and, for every Gaussian broadening, the DOS at
  ``E_F`` (states / spin / Ry) and the mode-resolved ``lambda_{q nu}`` /
  linewidth ``gamma_{q nu}`` (GHz).
* ``*.dyn`` -- a QE dynamical-matrix file: diagonalised frequencies (THz) and
  mode eigenvectors, kept for cross-checks and reuse in the later phase.

All frequencies are returned in THz for direct use by
:func:`PAOFLOW.elphon.eph_kq.eliashberg_from_modes`.
"""

import os
import re

import numpy as np

# 1 Rydberg in eV.
RY_TO_EV = 13.605693122994
# 1 THz expressed in eV (h * 1e12 Hz / e).
THZ_TO_EV = 4.135667696e-3
# 1 GHz expressed in eV (h * 1e9 Hz / e).
GHZ_TO_EV = 4.135667696e-6


def _ry2_to_thz(omega2_ry):
    """Convert squared frequencies in ``Ry^2`` to signed frequencies in THz.

    Imaginary modes (``omega^2 < 0``) are returned as negative THz frequencies
    (``-sqrt(|omega^2|)``), matching the QE ``matdyn`` / ``dynmat`` convention.
    """
    omega2_ry = np.asarray(omega2_ry, dtype=float)
    omega_ry = np.sign(omega2_ry) * np.sqrt(np.abs(omega2_ry))
    return omega_ry * RY_TO_EV / THZ_TO_EV


def read_lambda_in(path):
    """Parse a ``lambda.x`` input file.

    Parameters
    ----------
    path : str
        Path to the ``lambda.in`` file.

    Returns
    -------
    dict
        ``{'emax', 'degauss', 'smearing_method', 'qpoints', 'weights',
        'elph_files', 'mu_star'}``.  ``qpoints`` is ``(nq, 3)`` (fractional
        reciprocal coordinates), ``weights`` is ``(nq,)`` (raw, un-normalised),
        ``elph_files`` is the list of ``elph.inp_lambda.N`` paths as written in
        the file.
    """
    with open(path) as fh:
        raw = [ln.rstrip('\n') for ln in fh]

    # Strip inline comments (everything after '!') and drop blank lines.
    def _code(line):
        return line.split('!', 1)[0].strip()

    lines = [ln for ln in raw]

    # Line 0: emax, degauss, smearing method.
    tok0 = _code(lines[0]).split()
    emax = float(tok0[0])
    degauss = float(tok0[1])
    smearing_method = int(float(tok0[2]))

    # Line 1: number of q-points.
    nq = int(float(_code(lines[1]).split()[0]))

    # Next nq lines: qx qy qz weight.
    qpoints = np.zeros((nq, 3), dtype=float)
    weights = np.zeros(nq, dtype=float)
    idx = 2
    for iq in range(nq):
        tok = _code(lines[idx]).split()
        qpoints[iq] = [float(tok[0]), float(tok[1]), float(tok[2])]
        weights[iq] = float(tok[3])
        idx += 1

    # Next nq lines: the elph.inp_lambda.N file names.
    elph_files = []
    for _ in range(nq):
        name = _code(lines[idx])
        elph_files.append(name)
        idx += 1

    # Final line: mu*.
    mu_star = float(_code(lines[idx]).split()[0])

    return {
        'emax': emax,
        'degauss': degauss,
        'smearing_method': smearing_method,
        'qpoints': qpoints,
        'weights': weights,
        'elph_files': elph_files,
        'mu_star': mu_star,
    }


_RE_BROAD = re.compile(r'Gaussian Broadening:\s*([-+0-9.EeDd]+)\s*Ry')
_RE_DOS = re.compile(r'DOS\s*=\s*([-+0-9.EeDd]+)\s*states')
_RE_EF = re.compile(r'Ef\s*=\s*([-+0-9.EeDd]+)')
_RE_LAMBDA = re.compile(
    r'lambda\(\s*(\d+)\)\s*=\s*([-+0-9.EeDd]+)\s*gamma\s*=\s*([-+0-9.EeDd]+)\s*GHz'
)


def _to_float(tok):
    """Parse a Fortran-style float (accepting ``D`` exponents)."""
    return float(tok.replace('D', 'E').replace('d', 'e'))


def read_elph_inp_lambda(path):
    """Parse one ``elph.inp_lambda.N`` file (one irreducible q-point).

    Parameters
    ----------
    path : str
        Path to an ``elph.inp_lambda.N`` file.

    Returns
    -------
    dict
        ``{'q', 'nsigma', 'nmode', 'omega2_ry', 'omega_thz', 'sigma_ry',
        'dos_ry', 'ef_ev', 'lambda_qv', 'gamma_ghz'}``.  ``omega_thz`` is
        ``(nmode,)`` (signed THz); ``lambda_qv`` and ``gamma_ghz`` are
        ``(nsigma, nmode)``; ``dos_ry`` / ``ef_ev`` / ``sigma_ry`` are
        ``(nsigma,)`` (DOS in states / spin / Ry).
    """
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh]

    # Header: qx qy qz nsigma nmode.
    tok = lines[0].split()
    q = np.array([_to_float(tok[0]), _to_float(tok[1]), _to_float(tok[2])], dtype=float)
    nsigma = int(float(tok[3]))
    nmode = int(float(tok[4]))

    # Collect the omega^2 values (may wrap over several lines).
    omega2 = []
    li = 1
    while len(omega2) < nmode:
        omega2.extend(_to_float(t) for t in lines[li].split())
        li += 1
    omega2_ry = np.array(omega2[:nmode], dtype=float)

    sigma_ry = np.zeros(nsigma, dtype=float)
    dos_ry = np.zeros(nsigma, dtype=float)
    ef_ev = np.zeros(nsigma, dtype=float)
    lambda_qv = np.zeros((nsigma, nmode), dtype=float)
    gamma_ghz = np.zeros((nsigma, nmode), dtype=float)

    isig = -1
    for ln in lines[li:]:
        mb = _RE_BROAD.search(ln)
        if mb:
            isig += 1
            if isig >= nsigma:
                break
            sigma_ry[isig] = _to_float(mb.group(1))
            continue
        md = _RE_DOS.search(ln)
        if md:
            dos_ry[isig] = _to_float(md.group(1))
            me = _RE_EF.search(ln)
            if me:
                ef_ev[isig] = _to_float(me.group(1))
            continue
        ml = _RE_LAMBDA.search(ln)
        if ml:
            imode = int(ml.group(1)) - 1
            lambda_qv[isig, imode] = _to_float(ml.group(2))
            gamma_ghz[isig, imode] = _to_float(ml.group(3))

    return {
        'q': q,
        'nsigma': nsigma,
        'nmode': nmode,
        'omega2_ry': omega2_ry,
        'omega_thz': _ry2_to_thz(omega2_ry),
        'sigma_ry': sigma_ry,
        'dos_ry': dos_ry,
        'ef_ev': ef_ev,
        'lambda_qv': lambda_qv,
        'gamma_ghz': gamma_ghz,
    }


def load_qe_coupling(lambda_in_path, base_dir=None):
    """Load a complete QE coarse-grid coupling data set for the Eliashberg engine.

    Reads ``lambda.in`` and every referenced ``elph.inp_lambda.N`` file and packs
    the mode-resolved coupling into arrays indexed by ``(sigma, q, mode)``.

    Parameters
    ----------
    lambda_in_path : str
        Path to the ``lambda.x`` input file.
    base_dir : str, optional
        Directory the ``elph.inp_lambda.N`` paths are relative to (default: the
        directory of ``lambda_in_path``).

    Returns
    -------
    dict
        ``{'qpoints', 'weights', 'mu_star', 'sigma_ry', 'omega_thz',
        'lambda_qv', 'gamma_ghz', 'dos_ry', 'ef_ev', 'nq', 'nsigma', 'nmode'}``.
        ``omega_thz`` is ``(nq, nmode)``; ``lambda_qv`` / ``gamma_ghz`` are
        ``(nsigma, nq, nmode)``; ``dos_ry`` / ``ef_ev`` are ``(nsigma, nq)``;
        ``weights`` are the raw per-q weights from ``lambda.in``.
    """
    head = read_lambda_in(lambda_in_path)
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(lambda_in_path))

    per_q = []
    for name in head['elph_files']:
        p = name if os.path.isabs(name) else os.path.join(base_dir, name)
        per_q.append(read_elph_inp_lambda(p))

    nq = len(per_q)
    nsigma = per_q[0]['nsigma']
    nmode = per_q[0]['nmode']
    for d in per_q:
        if d['nsigma'] != nsigma or d['nmode'] != nmode:
            raise ValueError('inconsistent nsigma/nmode across elph.inp_lambda files.')

    omega_thz = np.stack([d['omega_thz'] for d in per_q], axis=0)  # (nq, nmode)
    lambda_qv = np.stack([d['lambda_qv'] for d in per_q], axis=1)  # (nsigma, nq, nmode)
    gamma_ghz = np.stack([d['gamma_ghz'] for d in per_q], axis=1)  # (nsigma, nq, nmode)
    dos_ry = np.stack([d['dos_ry'] for d in per_q], axis=1)  # (nsigma, nq)
    ef_ev = np.stack([d['ef_ev'] for d in per_q], axis=1)  # (nsigma, nq)

    return {
        'qpoints': head['qpoints'],
        'weights': head['weights'],
        'mu_star': head['mu_star'],
        'degauss': head['degauss'],
        'sigma_ry': per_q[0]['sigma_ry'],
        'omega_thz': omega_thz,
        'lambda_qv': lambda_qv,
        'gamma_ghz': gamma_ghz,
        'dos_ry': dos_ry,
        'ef_ev': ef_ev,
        'nq': nq,
        'nsigma': nsigma,
        'nmode': nmode,
    }


def lambda_from_gamma(gamma_ghz, dos_ry, omega_thz):
    r"""Recompute the mode coupling from the QE linewidth (units cross-check).

    QE defines the isotropic mode coupling as

    .. math::

        \lambda_{q\nu} = \frac{\gamma_{q\nu}}
                              {\pi\, N(E_F)\, \omega_{q\nu}^2},

    with the linewidth ``gamma`` and frequency ``omega`` in the same energy
    units and ``N(E_F)`` the DOS per spin.  This helper reproduces
    ``lambda_{q nu}`` from the tabulated ``gamma`` (GHz) and ``N(E_F)``
    (states / spin / Ry), so a comparison with the ``lambda(nu)`` values printed
    by QE validates the unit conventions used here.

    Parameters
    ----------
    gamma_ghz : array_like
        Linewidths in GHz (any shape).
    dos_ry : array_like
        DOS at ``E_F`` in states / spin / Ry, broadcastable to ``gamma_ghz``.
    omega_thz : array_like
        Phonon frequencies in THz, broadcastable to ``gamma_ghz``.

    Returns
    -------
    ndarray
        ``lambda_{q nu}`` (dimensionless); zero where ``omega`` is non-positive.
    """
    gamma_ev = np.asarray(gamma_ghz, dtype=float) * GHZ_TO_EV
    dos_ev = np.asarray(dos_ry, dtype=float) / RY_TO_EV
    omega_ev = np.asarray(omega_thz, dtype=float) * THZ_TO_EV
    out = np.zeros(np.broadcast(gamma_ev, dos_ev, omega_ev).shape, dtype=float)
    mask = omega_ev > 1.0e-9
    denom = np.pi * dos_ev * omega_ev**2
    np.divide(gamma_ev, denom, out=out, where=mask)
    return out


def read_qe_dyn(path):
    """Parse phonon frequencies and eigenvectors from a QE dynamical-matrix file.

    Kept for cross-checking the ``elph.inp_lambda`` frequencies and for reuse in
    the finite-difference phase (mode eigenvectors at the coarse q-points).

    Parameters
    ----------
    path : str
        Path to a QE ``*.dyn`` file (matrix + diagonalisation block).

    Returns
    -------
    dict
        ``{'q', 'freq_thz', 'eigenvectors'}`` for the (first) diagonalised
        q-point in the file.  ``freq_thz`` is ``(nmode,)``; ``eigenvectors`` is
        ``(nmode, natom, 3)`` complex (``None`` if the file has no
        diagonalisation block).
    """
    with open(path) as fh:
        lines = [ln.rstrip('\n') for ln in fh]

    q = None
    freqs = []
    vecs = []
    re_freq = re.compile(r'freq\s*\(\s*\d+\)\s*=\s*([-+0-9.EeDd]+)\s*\[THz\]')
    re_q = re.compile(r'q\s*=\s*\(\s*([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)')

    i = 0
    in_diag = False
    while i < len(lines):
        ln = lines[i]
        if 'Diagonalizing the dynamical matrix' in ln:
            in_diag = True
        if in_diag and q is None:
            mq = re_q.search(ln)
            if mq:
                q = np.array([_to_float(mq.group(k)) for k in (1, 2, 3)], dtype=float)
        mf = re_freq.search(ln)
        if in_diag and mf:
            freqs.append(_to_float(mf.group(1)))
            # The eigenvector rows follow, one '( re im re im re im )' per atom,
            # until the next 'freq' line or the closing '****' banner.
            comp = []
            j = i + 1
            while j < len(lines):
                s = lines[j].strip()
                if s.startswith('(') and s.endswith(')'):
                    nums = [_to_float(t) for t in s[1:-1].split()]
                    comp.append(nums)
                    j += 1
                else:
                    break
            vecs.append(comp)
            i = j
            continue
        i += 1

    if not freqs:
        return {'q': q, 'freq_thz': None, 'eigenvectors': None}

    freq_thz = np.array(freqs, dtype=float)
    nmode = len(freqs)
    natom = nmode // 3
    eig = np.zeros((nmode, natom, 3), dtype=complex)
    for m, comp in enumerate(vecs):
        flat = np.array(comp, dtype=float).reshape(-1)
        cplx = flat[0::2] + 1j * flat[1::2]
        eig[m] = cplx[: 3 * natom].reshape(natom, 3)

    return {'q': q, 'freq_thz': freq_thz, 'eigenvectors': eig}
