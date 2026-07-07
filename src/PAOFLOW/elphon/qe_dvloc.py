"""Bare local ionic-potential derivative ``dV_loc/du`` for the electron-phonon vertex.

The Quantum ESPRESSO ``fildvscf`` file stores only the *induced* self-consistent
potential (Hartree + xc response of the perturbed density).  The full
electron-phonon perturbation also contains the *bare* ionic contribution, which
QE applies separately in ``dvqpsi_us`` (local part) and ``dvqpsi_us_only``
(nonlocal part).  This module reconstructs the **bare local** part

.. math::

    \\Delta v_{\\mathrm{loc}}^{(a,c)}(\\mathbf{G}) =
        -i\\, v_{\\mathrm{loc}}(|\\mathbf{q}+\\mathbf{G}|)\\,
        (\\mathbf{q}+\\mathbf{G})_c\\,
        e^{-i(\\mathbf{q}+\\mathbf{G})\\cdot\\boldsymbol{\\tau}_a}

for displacing atom ``a`` along Cartesian direction ``c``, matching QE's
``compute_dvloc`` / ``vloc_of_g``.  The result is returned in real space on the
FFT grid so it can be added directly to the induced ``dvscf`` before the
matrix-element contraction.

The nonlocal (beta-projector) derivative is *not* included here.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.special import erf

# Rydberg atomic units (matching QE upflib/upf_const).
E2 = 2.0  # e^2 in Ry a.u.
FPI = 4.0 * np.pi


def read_upf_local(upf_path):
    """Read the local potential and radial mesh from a UPF (v2) pseudopotential.

    Parameters
    ----------
    upf_path : str
        Path to the ``.UPF`` file.

    Returns
    -------
    dict
        ``{'r', 'rab', 'vloc', 'zp', 'msh'}`` with ``r`` and ``rab`` the radial
        grid and integration weights (Bohr), ``vloc`` the local potential in Ry,
        ``zp`` the valence charge and ``msh`` the number of mesh points used for
        the radial integration.
    """
    root = ET.parse(upf_path).getroot()

    # UPF v2 tags may be namespaced in some files; search by local tag name.
    def _find(tag):
        for el in root.iter():
            if el.tag.split('}')[-1] == tag:
                return el
        raise KeyError('UPF tag %s not found in %s' % (tag, upf_path))

    header = _find('PP_HEADER')
    zp = float(header.attrib['z_valence'])
    r = np.array(_find('PP_R').text.split(), dtype=float)
    rab = np.array(_find('PP_RAB').text.split(), dtype=float)
    vloc = np.array(_find('PP_LOCAL').text.split(), dtype=float)  # Ry
    # QE integrates only up to r ~ 10 Bohr (rest is the analytic Coulomb tail).
    msh = int(np.searchsorted(r, 10.0)) + 1
    msh = min(msh, r.size)
    # simpson needs an odd number of intervals; QE forces msh odd.
    if msh % 2 == 0:
        msh += 1
    msh = min(msh, r.size)
    return {'r': r, 'rab': rab, 'vloc': vloc, 'zp': zp, 'msh': msh}


def _simpson(f, rab, msh):
    """QE-style Simpson integration ``\\int f dr`` on a logarithmic mesh."""
    # QE simpson: sum with weights (1,4,2,4,...,4,1)*rab/3, requires msh odd.
    n = msh if msh % 2 == 1 else msh - 1
    w = np.ones(n)
    w[1 : n - 1 : 2] = 4.0
    w[2 : n - 1 : 2] = 2.0
    return np.sum(f[:n] * w * rab[:n]) / 3.0


def vloc_of_g(vloc_data, gnorm, omega):
    """Local pseudopotential in reciprocal space, ``v_loc(G)`` (Ry).

    Reproduces QE ``vloc_of_g``: the ``erf(r)/r`` Coulomb tail is subtracted in
    real space (making the radial integrand short-ranged) and re-added
    analytically in G-space.

    Parameters
    ----------
    vloc_data : dict
        Output of :func:`read_upf_local`.
    gnorm : ndarray
        Magnitudes ``|q+G|`` in inverse Bohr (physical units).
    omega : float
        Unit-cell volume in Bohr^3.

    Returns
    -------
    ndarray
        ``v_loc(G)`` in Ry, same shape as ``gnorm``.  Entries with ``gnorm`` very
        small are handled with the analytic small-G limit of the smooth part.
    """
    r = vloc_data['r']
    rab = vloc_data['rab']
    vloc = vloc_data['vloc']
    zp = vloc_data['zp']
    msh = vloc_data['msh']

    g = np.atleast_1d(np.asarray(gnorm, dtype=float))
    out = np.zeros(g.shape, dtype=float)

    # short-ranged part: r*(r*vloc + zp*e2*erf(r))
    short = r * vloc + zp * E2 * erf(r)  # length nr, = r*vloc + tail

    small = g < 1.0e-8
    gg = g[~small]
    if gg.size:
        # integrand(ir) = (r*vloc + zp*e2*erf(r)) * sin(G r)/G  -> then *fpi/omega
        # vectorised over G: build sin(G r)/G on (nG, nr)
        n = msh if msh % 2 == 1 else msh - 1
        sr = short[:n]
        rr = r[:n]
        rabn = rab[:n]
        w = np.ones(n)
        w[1 : n - 1 : 2] = 4.0
        w[2 : n - 1 : 2] = 2.0
        wr = w * rabn / 3.0
        # sin(G r)/G
        sinint = np.sin(np.outer(gg, rr)) / gg[:, None]
        tab = (sinint * (sr * wr)[None, :]).sum(axis=1) * FPI / omega
        # re-add analytic Fourier transform of the -erf(r)/r Coulomb tail
        tab -= FPI / omega * zp * E2 * np.exp(-(gg**2) * 0.25) / gg**2
        out[~small] = tab

    # G -> 0 smooth limit (alpha Z term); rarely needed (only q=G=0).
    if np.any(small):
        n = msh if msh % 2 == 1 else msh - 1
        integ = r[:n] * (r[:n] * vloc[:n] + zp * E2)
        out[small] = _simpson_arr(integ, rab[:n]) * FPI / omega

    return out if out.size > 1 else out[0]


def _simpson_arr(f, rab):
    n = f.size if f.size % 2 == 1 else f.size - 1
    w = np.ones(n)
    w[1 : n - 1 : 2] = 4.0
    w[2 : n - 1 : 2] = 2.0
    return np.sum(f[:n] * w * rab[:n]) / 3.0


def _grid_gvectors(fft, bg, tpiba):
    """Physical G-vectors on the FFT box.

    Returns
    -------
    gcart : ndarray ``(nr1, nr2, nr3, 3)``
        Cartesian G in inverse Bohr.
    mill : tuple of ndarrays
        Integer Miller indices ``(m1, m2, m3)`` broadcast on the grid.
    """
    n1, n2, n3 = fft
    m1 = np.fft.fftfreq(n1, d=1.0 / n1).astype(int)
    m2 = np.fft.fftfreq(n2, d=1.0 / n2).astype(int)
    m3 = np.fft.fftfreq(n3, d=1.0 / n3).astype(int)
    M1, M2, M3 = np.meshgrid(m1, m2, m3, indexing='ij')
    # G_cart (tpiba units) = m . bg  ; physical = tpiba * that
    gcart = (M1[..., None] * bg[0] + M2[..., None] * bg[1] + M3[..., None] * bg[2]) * tpiba
    return gcart, (M1, M2, M3)


def bare_dvloc_cart(vloc_by_type, q_cryst, info, atom_types=None):
    """Bare local ionic derivative ``dV_loc/du`` in real space for one q.

    Parameters
    ----------
    vloc_by_type : dict or dict-output of :func:`read_upf_local`
        Either a mapping ``{atom_name: vloc_data}`` (one entry per species) or a
        single ``vloc_data`` dict when there is a single species.
    q_cryst : ndarray ``(3,)``
        q-point in crystal coordinates.
    info : dict
        Output of :func:`PAOFLOW.elphon.elph_bloch.read_nscf` (needs ``bg``,
        ``alat``, ``omega``, ``fft``, ``tau_cryst``, ``atom_names``).
    atom_types : list of str, optional
        Species name per atom; defaults to ``info['atom_names']``.

    Returns
    -------
    ndarray ``(3*nat, nr1, nr2, nr3)`` complex
        Cartesian ``dV_loc/du_c`` (Ry/Bohr) for each atom/direction, cell-periodic
        (same convention as the induced ``dvscf``), ready to add to it.
    """
    bg = info['bg']
    alat = info['alat']
    omega = info['omega']
    fft = info['fft']
    tau = info['tau_cryst']  # (nat, 3) crystal coords
    names = atom_types if atom_types is not None else info['atom_names']
    tpiba = 2.0 * np.pi / alat
    nat = tau.shape[0]

    gcart, (M1, M2, M3) = _grid_gvectors(fft, bg, tpiba)  # gcart physical, Miller ints
    # (q+G): q in crystal -> cartesian tpiba, add to G(tpiba); physical *tpiba
    qcart = (q_cryst[0] * bg[0] + q_cryst[1] * bg[1] + q_cryst[2] * bg[2]) * tpiba
    qg = gcart + qcart  # (n1,n2,n3,3) physical inverse Bohr
    qgnorm = np.linalg.norm(qg, axis=-1)

    out = np.zeros((3 * nat, fft[0], fft[1], fft[2]), dtype=complex)
    for na in range(nat):
        vd = (
            vloc_by_type[names[na]]
            if isinstance(vloc_by_type, dict) and names[na] in vloc_by_type
            else vloc_by_type
        )
        vg = vloc_of_g(vd, qgnorm.ravel(), omega).reshape(qgnorm.shape)  # v_loc(|q+G|) Ry
        # structure factor e^{-i (q+G).tau_na} = exp(-2pi i (q_cryst+Miller).s_na)
        s = tau[na]
        phase = np.exp(
            -2j
            * np.pi
            * ((M1 + q_cryst[0]) * s[0] + (M2 + q_cryst[1]) * s[1] + (M3 + q_cryst[2]) * s[2])
        )
        base = -1j * vg * phase  # common factor
        for c in range(3):
            dvg = base * qg[..., c]  # -i v_loc (q+G)_c e^{-i(q+G).tau}
            # real space cell-periodic part: sum_G dvg e^{iG.r} = N * ifftn(dvg)
            out[3 * na + c] = np.fft.ifftn(dvg) * (fft[0] * fft[1] * fft[2])
    return out


def load_vloc_for_run(info, pseudo_dir):
    """Read local potentials for every species referenced by an nscf run.

    Parameters
    ----------
    info : dict
        Output of :func:`PAOFLOW.elphon.elph_bloch.read_nscf`.
    pseudo_dir : str
        Directory containing the UPF files named in ``info['species']``.

    Returns
    -------
    dict
        ``{atom_name: vloc_data}``.
    """
    out = {}
    for name, upf in info['species'].items():
        out[name] = read_upf_local(os.path.join(pseudo_dir, upf))
    return out
