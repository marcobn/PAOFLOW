"""Input generation for the EPW coupling source of the PAO electron-phonon route.

``epw.x`` with ``epbwrite = .true.`` writes the coarse Bloch-basis coupling
(:func:`PAOFLOW.elphon.qe_elph_io.read_epw_epb`) computed from the nscf
wavefunctions, i.e. in the same band gauge as PAOFLOW's projections on that
save.  This module writes what such a run needs beyond the usual QE inputs:

* the explicit full k-point list of the nscf run (EPW requires the complete
  uniform grid with crystal coordinates in ``[0, 1)``);
* a placeholder ``prefix.ukk``, so that EPW can run with ``wannierize = .false.``
  when no Wannier functions are wanted (only the ``.epb`` files are used);
* a minimal ``epw.in``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# EPW caps the number of Wannier functions (readin.f90).
_EPW_MAX_NBNDSUB = 200


def uniform_kpoint_list(nk: tuple[int, int, int]) -> NDArray[np.float64]:
    """Full Gamma-centred k-grid in crystal coordinates, as EPW expects it.

    Parameters
    ----------
    nk : tuple of int
        Grid dimensions ``(nk1, nk2, nk3)``.

    Returns
    -------
    NDArray[np.float64], shape ``(nk1 * nk2 * nk3, 4)``
        Columns ``k1, k2, k3`` (crystal coordinates in ``[0, 1)``) and the
        weight ``1 / N``.  The third index runs fastest, the order of
        ``K_POINTS automatic`` that PAOFLOW's Fourier transforms assume
        (:func:`PAOFLOW.inputs.read_QE_xml.uniform_grid_from_kpoints`).
    """
    axes = [np.arange(n) / n for n in nk]
    kpts = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    weights = np.full((kpts.shape[0], 1), 1.0 / kpts.shape[0])
    return np.hstack([kpts, weights])


def kpoints_card(nk: tuple[int, int, int]) -> str:
    """``K_POINTS crystal`` card with the full grid for the nscf run.

    Parameters
    ----------
    nk : tuple of int
        Grid dimensions ``(nk1, nk2, nk3)``.

    Returns
    -------
    str
        The card, ready to append to a ``pw.x`` nscf input.
    """
    rows = uniform_kpoint_list(nk)
    lines = ['K_POINTS crystal', str(rows.shape[0])]
    lines += ['  %.10f  %.10f  %.10f  %.10e' % tuple(row) for row in rows]
    return '\n'.join(lines) + '\n'


def write_placeholder_ukk(path: str, nbnd: int, nk_total: int, nbndsub: int | None = None) -> None:
    """Write a ``prefix.ukk`` that lets EPW run without a Wannierization.

    Parameters
    ----------
    path : str
        Output file, ``prefix.ukk`` in the directory where ``epw.x`` is run.
    nbnd : int
        Number of nscf bands; all of them are kept (no ``exclude_bands``).
    nk_total : int
        Number of k-points of the nscf grid.
    nbndsub : int, optional
        Number of "Wannier functions" declared to EPW (``nbndsub`` in
        ``epw.in``).  Defaults to ``nbnd``; must not exceed 200.

    Returns
    -------
    None
        Writes ``path`` in the list-directed format of EPW's ``write_filukk``:
        identity rotations, every band inside the outer window (so EPW does not
        pack bands in the ``.epb`` files), no excluded band, zero centres.

    Raises
    ------
    ValueError
        If ``nbndsub`` exceeds ``nbnd`` or EPW's limit of 200.

    Notes
    -----
    The rotations only enter EPW's Wannier stage, which runs after the
    ``.epb`` files are written and whose output PAOFLOW does not use.  The
    format round-trips through :func:`PAOFLOW.elphon.qe_elph_io.read_epw_ukk`;
    a complete ``epw.x`` run with this placeholder has not been validated yet.
    """
    nbndsub = nbnd if nbndsub is None else int(nbndsub)
    if nbndsub > nbnd or nbndsub > _EPW_MAX_NBNDSUB:
        raise ValueError(
            'nbndsub=%d must be <= nbnd=%d and <= %d' % (nbndsub, nbnd, _EPW_MAX_NBNDSUB)
        )
    lines = ['%12d%12d' % (nbnd, 0)]
    lines += ['%12d' % (ibnd + 1) for ibnd in range(nbnd)]
    rotation = np.eye(nbnd, nbndsub)
    for _ in range(nk_total):
        lines += [' (%.16E,%.16E)' % (value, 0.0) for value in rotation.ravel()]
    lines += [' T'] * (nk_total * nbnd)
    lines += [' F'] * nbnd
    lines += ['%22.12E%22.12E%22.12E' % (0.0, 0.0, 0.0)] * nbndsub
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def epw_input(
    prefix: str,
    masses_amu: list[float],
    nk: tuple[int, int, int],
    nq: tuple[int, int, int],
    nbndsub: int,
    dvscf_dir: str = './save',
    outdir: str = './',
    wannierize: bool = False,
) -> str:
    """Minimal ``epw.in`` that writes the coarse Bloch coupling (``epbwrite``).

    Parameters
    ----------
    prefix : str
        Calculation prefix (same as the QE runs).
    masses_amu : list of float
        Atomic mass of each species (amu), in species order.
    nk, nq : tuple of int
        Coarse k-grid (the nscf grid) and coarse q-grid (the ph.x grid); ``nq``
        must divide ``nk`` in each direction.
    nbndsub : int
        Number of Wannier functions; with ``wannierize=False`` use the value
        given to :func:`write_placeholder_ukk`.
    dvscf_dir : str, optional
        Directory prepared by EPW's ``pp.py`` (dvscf, patterns, dyn files).
    outdir : str, optional
        QE ``outdir`` of the nscf run.
    wannierize : bool, optional
        Run wannier90 inside EPW (requires the usual ``proj``/window inputs,
        to be added by hand).  Default ``False`` uses the ``.ukk`` file.

    Returns
    -------
    str
        The ``&inputepw`` namelist.

    Raises
    ------
    ValueError
        If the q-grid is not commensurate with the k-grid.
    """
    if any(k % q for k, q in zip(nk, nq)):
        raise ValueError('q-grid %s must divide the k-grid %s' % (tuple(nq), tuple(nk)))
    flag = lambda value: '.true.' if value else '.false.'  # noqa: E731
    lines = [
        '&inputepw',
        "  prefix      = '%s'" % prefix,
        "  outdir      = '%s'" % outdir,
        "  dvscf_dir   = '%s'" % dvscf_dir,
    ]
    lines += ['  amass(%d)    = %.6f' % (i + 1, m) for i, m in enumerate(masses_amu)]
    lines += [
        '  elph        = .true.',
        '  epbwrite    = .true.',
        '  epbread     = .false.',
        '  epwwrite    = .true.',
        '  epwread     = .false.',
        '  wannierize  = %s' % flag(wannierize),
        '  nbndsub     = %d' % nbndsub,
        '  nk1 = %d, nk2 = %d, nk3 = %d' % tuple(nk),
        '  nq1 = %d, nq2 = %d, nq3 = %d' % tuple(nq),
        '  nkf1 = 1, nkf2 = 1, nkf3 = 1',
        '  nqf1 = 1, nqf2 = 1, nqf3 = 1',
        '/',
    ]
    return '\n'.join(lines) + '\n'
