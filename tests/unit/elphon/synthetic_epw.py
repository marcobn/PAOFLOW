"""Synthetic EPW ``epbwrite`` files and ``.ukk`` bookkeeping for unit tests."""

import numpy as np


def write_record(path, payload, split=None):
    """Write one sequential unformatted record, optionally as gfortran subrecords."""
    data = np.ascontiguousarray(payload).view(np.uint8)
    cuts = [0] + list(split or []) + [data.size]
    with open(path, 'wb') as fh:
        for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
            n = b - a
            more = i < len(cuts) - 2
            np.array([-n if more else n], np.int32).tofile(fh)
            data[a:b].tofile(fh)
            np.array([n], np.int32).tofile(fh)


def epb_payload(nq, xqc, et, dynq, epmatq, zstar, epsi):
    """Byte payload of one EPW pool record, Fortran (column-major) ordering."""
    parts = [
        np.array([nq], np.int32),
        np.asarray(xqc, np.float64).ravel(),  # xqc(3, nq): q-major rows == C ravel of (nq, 3)
        np.asarray(et, np.float64).ravel(order='F'),
        np.asarray(dynq, np.complex128).ravel(order='F'),
        np.asarray(epmatq, np.complex128).ravel(order='F'),
        np.asarray(zstar, np.float64).ravel(order='F'),
        np.asarray(epsi, np.float64).ravel(order='F'),
    ]
    return np.concatenate([p.view(np.uint8) for p in parts])


def write_ukk(path, nk, nbnd, nbndskip, lwin=None):
    """A ``.ukk`` with identity rotations (``nwann = 1``) and the band bookkeeping."""
    nbndep = nbnd - nbndskip
    lwin = np.ones((nk, nbndep), bool) if lwin is None else lwin
    lines = ['%12d%12d' % (nbndep, nbndskip)] + ['%12d' % (b + 1) for b in range(nbndskip, nbnd)]
    lines += [' (1.0000000000000000E+00,0.0000000000000000E+00)'] * (nk * nbndep)  # nwann = 1
    lines += [' T' if f else ' F' for f in lwin.ravel()]
    lines += [' T' if b < nbndskip else ' F' for b in range(nbnd)]
    lines += ['  0.0E+00  0.0E+00  0.0E+00']
    path.write_text('\n'.join(lines) + '\n')


def write_epw_dir(path, prefix, q_cryst, bg, et, dynq, epmatq, nbnd, nk, nat, nbndskip=0):
    """One-pool EPW directory (``prefix.epb1`` + ``prefix.ukk``) for ``load_epw_coupling``."""
    nq = len(q_cryst)
    xqc = np.asarray(q_cryst, float) @ np.asarray(bg, float)
    payload = epb_payload(nq, xqc, et, dynq, epmatq, np.zeros((3, 3, nat)), np.zeros((3, 3)))
    write_record(path / ('%s.epb1' % prefix), payload)
    write_ukk(path / ('%s.ukk' % prefix), nk, nbnd, nbndskip)
    return path
