"""Readers for the Quantum ESPRESSO DFPT phonon-perturbation output (Phase 1A / EPW route).

These parse the coarse-grid ingredients written by ``ph.x`` (``fildvscf``) that
are needed to build the electron-phonon matrix elements directly, i.e. the
PAOFLOW replacement for ``pw2wannier90`` + EPW's Bloch-space vertex:

* ``read_patterns`` -- the displacement-pattern XML
  (``_ph0/<prefix>.phsave/patterns.<iq>.xml``): the unitary basis ``U`` that
  relates the DFPT perturbations to Cartesian atomic displacements.
* ``read_dvscf`` -- the induced self-consistent potential ``d_{q,pert} V_scf(r)``
  (``_ph0/<prefix>.dvscf1`` for Gamma, ``_ph0/<prefix>.q_<iq>/<prefix>.dvscf1``
  otherwise): a flat Fortran record per perturbation on the dense FFT grid.
* ``dvscf_to_cartesian`` -- rotate ``dvscf`` from the pattern basis to the
  Cartesian ``d V / d u_{kappa alpha}`` basis.
* ``read_fft_grid`` -- the dense FFT dimensions from ``data-file-schema.xml``.

``dvscf`` stores only the *local* self-consistent part of the perturbation; the
nonlocal (beta-projector) contribution is added separately when the matrix
elements are assembled.

Convention: QE stores the lattice-periodic part, ``dvscf(r) = e^{-i q.r} d_q V(r)``.
For norm-conserving pseudopotentials (e.g. Pb) the smooth and dense grids
coincide, so a single FFT grid describes both the wavefunctions and ``dvscf``.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np


def read_fft_grid(data_file_schema):
    """Return the dense FFT grid ``(nr1, nr2, nr3)`` from ``data-file-schema.xml``."""
    root = ET.parse(data_file_schema).getroot()
    fft = root.find('.//basis_set/fft_grid')
    if fft is None:
        raise ValueError('fft_grid not found in %s' % data_file_schema)
    return (int(fft.attrib['nr1']), int(fft.attrib['nr2']), int(fft.attrib['nr3']))


def read_patterns(path):
    """Parse a QE ``patterns.<iq>.xml`` displacement-pattern file.

    Parameters
    ----------
    path : str
        Path to ``_ph0/<prefix>.phsave/patterns.<iq>.xml``.

    Returns
    -------
    dict
        ``{'iq', 'group_rank', 'minus_q', 'nirr', 'npert', 'nat', 'U'}``.
        ``U`` has shape ``(3*nat, npert)`` -- column ``p`` is the Cartesian
        displacement pattern of perturbation ``p`` (ordered irrep-then-
        perturbation, matching the ``dvscf`` records); it is unitary.  ``npert``
        equals ``3*nat``.
    """
    root = ET.parse(path).getroot()
    info = root.find('IRREPS_INFO')
    iq = int(info.findtext('QPOINT_NUMBER'))
    group_rank = int(info.findtext('QPOINT_GROUP_RANK'))
    minus_q = info.findtext('MINUS_Q_SYM').strip().lower() == 'true'
    nirr = int(info.findtext('NUMBER_IRR_REP'))

    columns = []
    for irr in range(1, nirr + 1):
        rep = info.find('REPRESENTION.%d' % irr)
        npert_irr = int(rep.findtext('NUMBER_OF_PERTURBATIONS'))
        for p in range(1, npert_irr + 1):
            pert = rep.find('PERTURBATION.%d' % p)
            text = pert.findtext('DISPLACEMENT_PATTERN').split()
            vals = np.array([float(t.replace('D', 'E').replace('d', 'e')) for t in text])
            vec = vals[0::2] + 1j * vals[1::2]  # (3*nat,)
            columns.append(vec)

    U = np.array(columns, dtype=complex).T  # (3*nat, npert)
    nat = U.shape[0] // 3
    return {
        'iq': iq,
        'group_rank': group_rank,
        'minus_q': minus_q,
        'nirr': nirr,
        'npert': U.shape[1],
        'nat': nat,
        'U': U,
    }


def read_dvscf(path, nr, nmode, nspin=1):
    """Read a QE ``dvscf`` file into ``(nmode, nr1, nr2, nr3[, nspin])`` complex.

    The file is a sequence of ``nmode`` (times ``nspin``) direct-access records,
    each the perturbation potential on the dense FFT grid in Fortran order
    (fastest index ``nr1``), with no record markers.

    Parameters
    ----------
    path : str
        Path to the ``dvscf`` file.
    nr : tuple(int, int, int)
        Dense FFT dimensions ``(nr1, nr2, nr3)``.
    nmode : int
        Number of perturbations (``3*nat``).
    nspin : int, optional
        Number of spin channels (default 1).

    Returns
    -------
    ndarray
        Shape ``(nmode, nr1, nr2, nr3)`` for ``nspin == 1``, else
        ``(nmode, nspin, nr1, nr2, nr3)`` -- in the *pattern* basis.
    """
    nr1, nr2, nr3 = nr
    nnr = nr1 * nr2 * nr3
    raw = np.fromfile(path, dtype=np.complex128)
    expected = nmode * nspin * nnr
    if raw.size != expected:
        raise ValueError(
            'dvscf size mismatch: got %d, expected nmode*nspin*nr = %d (nr=%s, nmode=%d, nspin=%d)'
            % (raw.size, expected, nr, nmode, nspin)
        )
    if nspin == 1:
        out = np.empty((nmode, nr1, nr2, nr3), dtype=complex)
        blk = raw.reshape(nmode, nnr)
        for m in range(nmode):
            out[m] = blk[m].reshape((nr1, nr2, nr3), order='F')
        return out
    out = np.empty((nmode, nspin, nr1, nr2, nr3), dtype=complex)
    blk = raw.reshape(nmode, nspin, nnr)
    for m in range(nmode):
        for s in range(nspin):
            out[m, s] = blk[m, s].reshape((nr1, nr2, nr3), order='F')
    return out


def dvscf_to_cartesian(dvscf_pattern, U):
    """Rotate ``dvscf`` from the pattern basis to Cartesian ``dV/du_{kappa alpha}``.

    ``dvscf_cart[c] = sum_p conj(U[c, p]) dvscf_pattern[p]`` with ``U`` the unitary
    pattern matrix from :func:`read_patterns` (columns = patterns).

    The perturbation is linear in the displacement, so the pattern-basis response
    is ``dvscf_pattern[p] = sum_c U[c, p] dvscf_cart[c]`` (i.e. ``dv_pat = U^T
    dv_cart``).  Inverting with the unitarity of ``U`` gives ``dv_cart = conj(U)
    dv_pat``.  The conjugate is essential for q-points whose displacement
    patterns are complex (e.g. the star of a general q); for real patterns
    ``conj(U) = U`` and the result is unchanged.

    Parameters
    ----------
    dvscf_pattern : ndarray
        Shape ``(npert, ...)`` in the pattern basis.
    U : ndarray
        Shape ``(3*nat, npert)`` pattern matrix.

    Returns
    -------
    ndarray
        Shape ``(3*nat, ...)`` -- the Cartesian displacement derivatives.
    """
    dvscf_pattern = np.asarray(dvscf_pattern)
    grid_shape = dvscf_pattern.shape[1:]
    flat = dvscf_pattern.reshape(dvscf_pattern.shape[0], -1)  # (npert, ngrid)
    cart = U.conj() @ flat  # (3*nat, ngrid) ; dv_cart = conj(U) dv_pattern
    return cart.reshape((U.shape[0],) + grid_shape)


def dvscf_path(ph0_dir, prefix, iq, fildvscf='dvscf'):
    """Path to the ``dvscf`` file for irreducible q-index ``iq`` (1-based).

    ``iq == 1`` (Gamma) lives directly in ``_ph0/``; others in
    ``_ph0/<prefix>.q_<iq>/``.  The file name is ``<prefix>.<fildvscf>1``, where
    ``fildvscf`` is the ``ph.x`` input keyword (default ``'dvscf'``; e.g.
    ``'pbdv'`` gives ``lead.pbdv1``).
    """
    fname = '%s.%s1' % (prefix, fildvscf)
    if iq == 1:
        return os.path.join(ph0_dir, fname)
    return os.path.join(ph0_dir, '%s.q_%d' % (prefix, iq), fname)


def patterns_path(ph0_dir, prefix, iq):
    """Path to the ``patterns.<iq>.xml`` file."""
    return os.path.join(ph0_dir, '%s.phsave' % prefix, 'patterns.%d.xml' % iq)
