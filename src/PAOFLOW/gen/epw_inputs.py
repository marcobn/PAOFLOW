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


def parse_exclude_bands(spec: str) -> list[int]:
    """Parse an EPW/wannier90 band list such as ``'1:5, 8'`` (or ``'1-5 8'``).

    Parameters
    ----------
    spec : str
        Comma- or space-separated 1-based band indices and ``first:last`` (or
        ``first-last``) ranges; an optional ``exclude_bands =`` prefix is ignored.

    Returns
    -------
    list of int
        Sorted, unique 1-based band indices (empty for a blank string).

    Raises
    ------
    ValueError
        If an entry is not an integer or a valid range.
    """
    body = spec.split('=', 1)[-1] if '=' in spec else spec
    bands = set()
    for token in body.replace(',', ' ').split():
        bounds = token.replace('-', ':').split(':')
        if len(bounds) == 1:
            bands.add(int(bounds[0]))
        elif len(bounds) == 2 and int(bounds[0]) <= int(bounds[1]):
            bands.update(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            raise ValueError('invalid band range %r' % token)
    return sorted(bands)


def exclude_bands_string(exclude_bands: list[int]) -> str:
    """EPW ``bands_skipped`` value for a list of 1-based band indices.

    Parameters
    ----------
    exclude_bands : list of int
        1-based band indices to exclude.

    Returns
    -------
    str
        e.g. ``'exclude_bands = 1:5, 8'``, with consecutive bands joined into
        ``first:last`` ranges as in the EPW and wannier90 documentation.
    """
    bands = sorted(set(int(b) for b in exclude_bands))
    ranges, first = [], None
    for i, band in enumerate(bands):
        if first is None:
            first = band
        if i + 1 == len(bands) or bands[i + 1] != band + 1:
            ranges.append(str(first) if first == band else '%d:%d' % (first, band))
            first = None
    return 'exclude_bands = ' + ', '.join(ranges)


def write_placeholder_ukk(
    path: str,
    nbnd: int,
    nk_total: int,
    nbndsub: int | None = None,
    exclude_bands: list[int] | None = None,
    nelec: float | None = None,
    noncolin: bool = False,
    wannier90_stubs: bool = True,
) -> None:
    """Write a ``prefix.ukk`` that lets EPW run without a Wannierization.

    Parameters
    ----------
    path : str
        Output file, ``prefix.ukk`` in the directory where ``epw.x`` is run.
    nbnd : int
        Number of nscf bands.
    nk_total : int
        Number of k-points of the nscf grid.
    nbndsub : int, optional
        Number of "Wannier functions" declared to EPW (``nbndsub`` in
        ``epw.in``).  Defaults to the number of kept bands; must not exceed it
        nor EPW's limit of 200.
    exclude_bands : list of int, optional
        1-based indices of bands excluded from the electron-phonon calculation
        (e.g. semicore states).  With ``wannierize = .false.`` EPW reads the
        exclusion from this file only; ``bands_skipped`` in ``epw.in`` is used
        solely to write wannier90's ``.win``.
    nelec : float, optional
        Number of valence electrons; required with ``exclude_bands`` to count
        the occupied excluded bands (EPW's ``nbndskip``).
    noncolin : bool, optional
        Noncollinear calculation (one electron per band in ``nbndskip``).
    wannier90_stubs : bool, optional
        Also write empty wannier90 overlap files ``prefix.bvec`` (no b-vectors)
        and ``prefix.mmn`` next to ``path`` (default ``True``).  EPW's Wannier
        stage reads them unconditionally (``vmebloch2wan``); with zero
        b-vectors it completes, with zero position matrix elements, instead of
        stopping after the ``.epb`` files are written.

    Returns
    -------
    None
        Writes ``path`` in the list-directed format of EPW's ``write_filukk``:
        kept-band list, identity rotations, every kept band inside the outer
        window (so EPW does not pack bands in the ``.epb`` files), the
        excluded-band flags and zero centres; plus the two stub files when
        ``wannier90_stubs`` is true.

    Raises
    ------
    ValueError
        If ``nbndsub`` is out of range, a band index is invalid, or ``nelec`` is
        missing while bands are excluded.

    Notes
    -----
    ``nbndskip`` follows EPW's ``setup_nnkp``: the number of excluded bands
    with index ``<= nelec / 2`` (``<= nelec`` if noncollinear).  The rotations
    only enter EPW's Wannier stage, which runs after the ``.epb`` files are
    written and whose output PAOFLOW does not use.
    """
    excluded = sorted(set(int(b) for b in (exclude_bands or [])))
    if any(b < 1 or b > nbnd for b in excluded):
        raise ValueError('excluded band indices must be in 1..%d' % nbnd)
    if excluded and nelec is None:
        raise ValueError('nelec is required to count the occupied excluded bands')
    kept = [b for b in range(1, nbnd + 1) if b not in excluded]
    nbndep = len(kept)
    nbndskip = 0
    if excluded:
        occupied = nelec if noncolin else int(nelec / 2)  # highest occupied band index
        nbndskip = sum(1 for b in excluded if b <= occupied)
    nbndsub = nbndep if nbndsub is None else int(nbndsub)
    if nbndsub < 1 or nbndsub > nbndep or nbndsub > _EPW_MAX_NBNDSUB:
        raise ValueError(
            'nbndsub=%d must be in 1..%d (kept bands) and <= %d'
            % (nbndsub, nbndep, _EPW_MAX_NBNDSUB)
        )
    lines = ['%12d%12d' % (nbndep, nbndskip)]
    lines += ['%12d' % b for b in kept]
    rotation = np.eye(nbndep, nbndsub)
    for _ in range(nk_total):
        lines += [' (%.16E,%.16E)' % (value, 0.0) for value in rotation.ravel()]
    lines += [' T'] * (nk_total * nbndep)
    lines += [' T' if b in excluded else ' F' for b in range(1, nbnd + 1)]
    lines += ['%22.12E%22.12E%22.12E' % (0.0, 0.0, 0.0)] * nbndsub
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    if wannier90_stubs:
        stem = path[: -len('.ukk')] if path.endswith('.ukk') else path
        with open(stem + '.bvec', 'w') as fh:
            fh.write('placeholder_written_by_PAOFLOW\n%d %d\n' % (nk_total, 0))
        with open(stem + '.mmn', 'w') as fh:
            fh.write('')


def epw_input(
    prefix: str,
    masses_amu: list[float],
    nk: tuple[int, int, int],
    nq: tuple[int, int, int],
    nbndsub: int,
    dvscf_dir: str = './save',
    outdir: str = './',
    wannierize: bool = False,
    exclude_bands: list[int] | None = None,
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
    exclude_bands : list of int, optional
        1-based bands to exclude, written as ``bands_skipped``.  With
        ``wannierize=False`` EPW takes the exclusion from the ``.ukk`` file
        (:func:`write_placeholder_ukk` with the same list); the line is kept so
        that the input documents, and with ``wannierize=True`` applies, the
        same choice.

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
    ]
    if exclude_bands:
        lines.append("  bands_skipped = '%s'" % exclude_bands_string(exclude_bands))
    lines += [
        '  nk1 = %d, nk2 = %d, nk3 = %d' % tuple(nk),
        '  nq1 = %d, nq2 = %d, nq3 = %d' % tuple(nq),
        '  nkf1 = 1, nkf2 = 1, nkf3 = 1',
        '  nqf1 = 1, nqf2 = 1, nqf3 = 1',
        '/',
    ]
    return '\n'.join(lines) + '\n'
