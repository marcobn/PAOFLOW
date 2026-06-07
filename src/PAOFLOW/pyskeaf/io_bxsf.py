"""BXSF (XCrysDen Band-XSF) file reader for PAOFLOW.pyskeaf.

Implements the validation rules of the Fortran subroutine ``preadbxsf`` in
``skeaf_v1p3p0_r149.F90`` (lines 2873–3067):

* Single-band files only (multi-band BXSFs must be split first).
* Reciprocal-lattice origin must be (0, 0, 0).
* Number of energies in the data block must equal ``nx * ny * nz``.
* Reject obvious "periodic-grid" BXSFs (ELK / exciting style) where the
  energy at the corner ``E[nx,ny,nz]`` equals ``E[1,1,1]``; these need to be
  converted to the General Grid before SKEAF can consume them.

Reciprocal-lattice vectors in BXSF files are in units of (atomic units)^-1
**without** the factor of 2π.  We expose two views:

* ``recip_au`` — exactly as read from the file, units of a.u.^-1.
* ``recip_ang`` — converted to Å^-1 *with* the 2π factor included, i.e.
  ``recip_ang = 2π · recip_au / CONV_AU_TO_ANG``.  This matches the
  ``plr*`` arrays in the Fortran source and is the form used by all
  downstream SKEAF computations (areas in Å^-2 → frequencies in kT).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np

from PAOFLOW.pyskeaf.constants import CONV_AU_TO_ANG, PI


class BXSFError(ValueError):
    """Raised when a BXSF file fails SKEAF's validation rules."""


@dataclass
class BXSFData:
    """Parsed contents of a BXSF (XCrysDen Band-XSF) file.

    Attributes
    ----------
    filename : str
        Path the data was read from (informational; used in output headers).
    fermi_energy : float
        Fermi energy in Rydbergs, as written in the file's BEGIN_INFO block.
    nx, ny, nz : int
        Grid dimensions (number of k-points along each reciprocal lattice
        direction).  Indexing in this package is 0-based; the Fortran code
        uses 1-based indexing.
    recip_au : np.ndarray, shape (3, 3)
        Reciprocal lattice vectors as stored in the BXSF file (a.u.^-1, **no**
        2π factor).  ``recip_au[i]`` is the i-th vector.
    recip_ang : np.ndarray, shape (3, 3)
        Reciprocal lattice vectors converted to Å^-1, *with* the 2π factor.
        Computed as ``2π · recip_au / CONV_AU_TO_ANG`` — matches the Fortran
        ``plrx*/plry*/plrz*`` variables that drive the area→frequency
        conversion.
    energies : np.ndarray, shape (nx, ny, nz)
        Band energies in Rydbergs.  Stored in C-contiguous order with the
        third index varying fastest (the BXSF "General Grid" convention,
        matching ``masterkarray(i,j,k)`` in Fortran).
    """

    filename: str
    fermi_energy: float
    nx: int
    ny: int
    nz: int
    recip_au: np.ndarray
    recip_ang: np.ndarray
    energies: np.ndarray
    origin_au: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def num_kpoints(self) -> int:
        return self.nx * self.ny * self.nz


# --- internal helpers --------------------------------------------------------

# Float pattern that tolerates Fortran-style exponents (1.0E+00, 1.0d-3, 1.0).
_FLOAT_RE = re.compile(r'[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?')
_INT_RE = re.compile(r'[-+]?\d+')
_FERMI_RE = re.compile(r'fermi\s*energy', re.IGNORECASE)
# Keyword that immediately precedes the band-count line in a BXSF bandgrid block.
_BANDGRID_RE = re.compile(r'BANDGRID_3D_BANDS', re.IGNORECASE)
_BAND_MARKER_RE = re.compile(r'(?:^|[^A-Za-z0-9_])(?:band|prod)\s*:', re.IGNORECASE)
# Matches the BXSF "END_BANDGRID_3D" / "END_BLOCK_BANDGRID_3D" / lone "END" markers.
# (cannot use \bend\b because the trailing underscore is a word character).
_END_MARKER_RE = re.compile(r'(?:^|\s)END(?:_|\s|$)', re.IGNORECASE)


def _to_float(token: str) -> float:
    """Parse a Fortran-style float token (handles ``1.5d-3`` and ``1.5D+10``)."""
    return float(token.replace('D', 'E').replace('d', 'e'))


def _read_first_int(line: str) -> int:
    m = _INT_RE.search(line)
    if m is None:
        raise BXSFError(f'Expected an integer, got: {line!r}')
    return int(m.group(0))


def _read_n_floats(line: str, n: int) -> list[float]:
    toks = _FLOAT_RE.findall(line)
    if len(toks) < n:
        raise BXSFError(f'Expected at least {n} floats, got {len(toks)} in: {line!r}')
    return [_to_float(t) for t in toks[:n]]


# --- public API --------------------------------------------------------------


def read_bxsf(path: Union[str, Path]) -> BXSFData:
    """Read a single-band BXSF file and return a :class:`BXSFData`.

    Parameters
    ----------
    path : str or Path
        Path to the BXSF file.

    Raises
    ------
    BXSFError
        If the file fails any of the validation checks copied from
        ``preadbxsf`` (multi-band, non-zero origin, count mismatch,
        periodic-grid layout, grid too small, etc.).
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    with path.open('r') as fh:
        lines = fh.readlines()

    it = iter(enumerate(lines))

    # 1. Find Fermi energy (anywhere before the data block).
    fermi_energy = None
    for _i, line in it:
        if _FERMI_RE.search(line):
            tokens = _FLOAT_RE.findall(line)
            if not tokens:
                raise BXSFError(f'Could not parse Fermi energy from line: {line!r}')
            fermi_energy = _to_float(tokens[-1])
            break
    if fermi_energy is None:
        raise BXSFError(f"BXSF file {path}: no 'Fermi Energy' line found.")

    # 2. Advance to the BANDGRID_3D_BANDS keyword that introduces the data
    #    block.  Anchoring on the keyword (rather than skipping a fixed number
    #    of lines) tolerates the optional blank line and comment that XCrysDen
    #    and other writers place between END_INFO and the bandgrid block.
    found_bandgrid = False
    for _i, line in it:
        if _BANDGRID_RE.search(line):
            found_bandgrid = True
            break
    if not found_bandgrid:
        raise BXSFError(
            f'BXSF file {path}: no BANDGRID_3D_BANDS block found after header.'
        )

    # 3. Number of bands — read from the next non-blank line; must be exactly 1.
    n_bands = None
    for _i, line in it:
        if line.strip():
            n_bands = _read_first_int(line)
            break
    if n_bands is None:
        raise BXSFError(f'BXSF file {path}: missing band-count line.')
    if n_bands != 1:
        raise BXSFError(
            f'BXSF file {path}: header indicates {n_bands} bands, but SKEAF '
            'requires single-band BXSFs (split with XCrysDen first).'
        )

    # 4. Grid dimensions.
    try:
        _, line = next(it)
    except StopIteration as exc:
        raise BXSFError(f'BXSF file {path}: missing grid-dimension line.') from exc
    dims = _read_n_floats(line, 3)
    nx, ny, nz = (int(round(v)) for v in dims)
    if nx < 2 or ny < 2 or nz < 2:
        raise BXSFError(
            f'BXSF file {path}: grid dimensions ({nx}, {ny}, {nz}) too small; '
            'need at least 2 in every direction.'
        )

    # 5. Reciprocal-lattice origin (must be 0, 0, 0).
    try:
        _, line = next(it)
    except StopIteration as exc:
        raise BXSFError(f'BXSF file {path}: missing origin line.') from exc
    origin = np.array(_read_n_floats(line, 3), dtype=float)
    if not np.allclose(origin, 0.0):
        raise BXSFError(
            f'BXSF file {path}: reciprocal-lattice origin is {tuple(origin)}, '
            'not (0, 0, 0).  SKEAF requires a Brillouin-zone-centred grid.'
        )

    # 6. Three reciprocal-lattice vectors (a.u.^-1, no 2π factor).
    recip_au = np.empty((3, 3), dtype=float)
    for j in range(3):
        try:
            _, line = next(it)
        except StopIteration as exc:
            raise BXSFError(
                f'BXSF file {path}: missing reciprocal-lattice vector {j + 1}.'
            ) from exc
        recip_au[j] = _read_n_floats(line, 3)

    # 7. "BAND: 1" line (skip).
    try:
        next(it)
    except StopIteration as exc:
        raise BXSFError(f'BXSF file {path}: missing BAND marker.') from exc

    # 8. Energy block — read floats until END marker.  The Fortran reader
    #    detects extra BAND/Prod markers as a multi-band error.
    expected = nx * ny * nz
    flat = np.empty(expected, dtype=float)
    n_read = 0
    for _i, line in it:
        if _BAND_MARKER_RE.search(line):
            raise BXSFError(
                f"BXSF file {path}: a 'BAND:' or 'Prod:' marker was found in "
                'the energy block — file contains multiple bands.  Split it '
                'with XCrysDen first.'
            )
        if _END_MARKER_RE.search(line):
            break
        for tok in _FLOAT_RE.findall(line):
            if n_read >= expected:
                # Tolerate trailing junk after the data block on the same line
                # as the END marker would have been; Fortran stops reading at
                # the first END.  Anything before END but after `expected` is
                # an error.
                raise BXSFError(
                    f'BXSF file {path}: more than {expected} energies found '
                    'before the END marker.'
                )
            flat[n_read] = _to_float(tok)
            n_read += 1
    if n_read != expected:
        raise BXSFError(
            f'BXSF file {path}: read {n_read} energies, expected ' f'{expected} ({nx}*{ny}*{nz}).'
        )

    # 9. Reshape to (nx, ny, nz).  BXSF General Grid stores energies with
    #    the third (z) index varying fastest, then y, then x — i.e. row-major
    #    (C-order) of an (nx, ny, nz) array.  This matches Fortran's
    #    ``masterkarray(i,j,k) = kreadarray(((i-1)*ny + (j-1))*nz + k)``.
    energies = flat.reshape((nx, ny, nz))

    # 10. Periodic-grid sanity check (ELK / exciting symptom).
    #
    #     The Fortran code raises a hard STOP whenever
    #     ``E[nx,ny,nz] == E[1,1,1]`` (see preadbxsf line 3019).  In practice
    #     this triggers many false positives — e.g. the bundled ``Cylinder``
    #     test file, where the outer "vacuum" region is at a constant energy
    #     so *all* boundary faces happen to match — yet the Fortran is known
    #     to produce correct results on that file.  We therefore only emit a
    #     warning, leaving the user to act on it.
    e = energies
    faces_match = (
        np.array_equal(e[0, :, :], e[-1, :, :])
        and np.array_equal(e[:, 0, :], e[:, -1, :])
        and np.array_equal(e[:, :, 0], e[:, :, -1])
    )
    if faces_match or e[-1, -1, -1] == e[0, 0, 0]:
        import warnings

        kind = 'all boundary faces match' if faces_match else 'corner energies match'
        warnings.warn(
            f'BXSF file {path}: {kind} — possible Periodic Grid layout '
            '(typical of ELK / exciting).  If the file does not look correct, '
            'run the ELK_exciting_BXSFconverter utility first.  Continuing.',
            stacklevel=2,
        )

    # 11. Convert reciprocal vectors to Å^-1 (with the 2π factor) for the
    #     SKEAF area→frequency conversion.
    recip_ang = (2.0 * PI / CONV_AU_TO_ANG) * recip_au

    return BXSFData(
        filename=str(path),
        fermi_energy=fermi_energy,
        nx=nx,
        ny=ny,
        nz=nz,
        recip_au=recip_au,
        recip_ang=recip_ang,
        energies=energies,
        origin_au=origin,
    )


def write_bxsf(
    path: Union[str, Path],
    data: BXSFData,
    *,
    fermi_energy: float | None = None,
    band_label: str = 'band_energies',
) -> None:
    """Write *data* back to a BXSF file (round-trip helper for tests).

    The output format mirrors the Fortran-generated style (six energies per
    line, ``%.6E`` formatting).  Mainly intended for round-trip tests; the
    SKEAF Python port itself never needs to *produce* BXSFs.
    """
    path = Path(path)
    fe = fermi_energy if fermi_energy is not None else data.fermi_energy

    with path.open('w') as fh:
        fh.write(' BEGIN_INFO\n')
        fh.write(f'   Fermi Energy:     {fe:.6f}\n')
        fh.write(' END_INFO\n')
        fh.write(' BEGIN_BLOCK_BANDGRID_3D\n')
        fh.write(f' {band_label}\n')
        fh.write(' BANDGRID_3D_BANDS\n')
        fh.write(' 1\n')
        fh.write(f' {data.nx:3d} {data.ny:3d} {data.nz:3d}\n')
        fh.write(
            f'      {data.origin_au[0]:.8f}      '
            f'{data.origin_au[1]:.8f}      {data.origin_au[2]:.8f}\n'
        )
        for j in range(3):
            v = data.recip_au[j]
            fh.write(f'      {v[0]:.8f}      {v[1]:.8f}      {v[2]:.8f}\n')
        fh.write(' BAND:    1\n')

        flat = data.energies.reshape(-1)
        for i in range(0, flat.size, 6):
            chunk = flat[i : i + 6]
            fh.write('  ' + '  '.join(f'{v:.6E}' for v in chunk) + '\n')

        fh.write(' END_BANDGRID_3D\n')
        fh.write(' END_BLOCK_BANDGRID_3D\n')
