"""Phonons workflow: q2r + matdyn in PAOFLOW on the ``ph.x`` dynamical matrices of a system."""

from __future__ import annotations

import os
import warnings
from typing import Sequence

from ...phonon.qe_force_constants import THZ_TO_CM1, q2r_matdyn
from .system import ElphonSystem


def phonon_dispersion(
    system: ElphonSystem,
    outdir: str,
    *,
    units: str = 'cm-1',
    mesh: Sequence[int] = (24, 24, 24),
    sigma: float = 3.0,
    npoints: int = 51,
    fname: str | None = None,
    compare_plain: bool = True,
) -> dict:
    """Phonon dispersion and DOS from ``<prefix>.dyn0 ... dynN`` (q2r.x + matdyn.x in PAOFLOW).

    Parameters
    ----------
    system : ElphonSystem
        ``ph.x`` must have been run with ``ldisp`` and ``fildyn = '<prefix>.dyn'`` in the
        phonon directory.
    outdir : str
        Output directory (``<prefix>.fc``, ``<fname>_band.dat``, ``<fname>_band.labels``,
        ``<fname>_dos.dat`` and, for polar crystals, the ``_plain`` counterparts).
    units : {'cm-1', 'THz'}
    mesh : sequence of int
        q-mesh of the DOS.
    sigma : float
        Gaussian broadening of the DOS, in ``units``.
    npoints : int
        Points per line of the high-symmetry path.
    fname : str, optional
        File-name stem of the band and DOS files (default: the prefix in lower case).
    compare_plain : bool
        For polar crystals also the plain interpolation without the dipole (LO-TO) term.

    Returns
    -------
    dict
        ``result`` (``q2r_matdyn`` with the long-range model when the dynamical matrices
        carry Born charges), ``plain`` (``None`` unless polar and ``compare_plain``),
        ``polar``, ``scale`` (to ``units``), ``units`` and ``fname``.
    """
    fname = fname or system.prefix.lower()
    sigma_thz = sigma / THZ_TO_CM1 if units == 'cm-1' else sigma
    common = dict(npoints=npoints, mesh=tuple(mesh), sigma_thz=sigma_thz, units=units)
    result = q2r_matdyn(
        system.prefix,
        system.phonon_path,
        outdir=outdir,
        fname=fname,
        force_constants_file=system.prefix + '.fc',
        **common,
    )
    polar = result.get('long_range') is not None
    plain = None
    if polar and compare_plain:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # the plain interpolation warns about the LO-TO term
            plain = q2r_matdyn(
                system.prefix,
                system.phonon_path,
                outdir=outdir,
                fname=fname + '_plain',
                long_range=False,
                **common,
            )
    return {
        'result': result,
        'plain': plain,
        'polar': polar,
        'scale': THZ_TO_CM1 if units == 'cm-1' else 1.0,
        'units': units,
        'fname': fname,
        'outdir': os.path.abspath(outdir),
    }
