"""Readers for pyskeaf / SKEAF frequency-vs-angle tables and BXSF Fermi-surface grids.

Units (as produced by ``PAOFLOW.pyskeaf``):

* ``qo_EF_<E>_freqvsangle_<band>.out``: frequency in kT, effective mass in
  electron masses, curvature ``d^2A/dk_H^2`` in kT A^2 (negative for a maximum,
  positive for a minimum), carrier type +1 electron / -1 hole, and the number of
  supercell copies in which the orbit was found.
* BXSF band energies are in Rydberg; ``fermi_energy`` passed to pyskeaf is in eV.

Both file types may be gzip-compressed (``.gz``); compressed BXSF files are
decompressed into a cache folder before being handed to ``PAOFLOW.pyskeaf``.
"""

from __future__ import annotations

import gzip
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

QO_COLUMNS = [
    'azimuthal_deg',
    'polar_deg',
    'freq_kT',
    'mstar_me',
    'curv_kTA2',
    'orbit_type',
    'num_orbit_copies',
]
RYDBERG_IN_EV = 13.605693122994
AREA_TO_KILOTESLA = 10.47576797
_QO_PATTERN = re.compile(r'qo_EF_(?P<token>.+)_freqvsangle_(?P<band>\d+)\.out(?:\.gz)?$')


def find_frequency_files(skeaf_dir: str | Path, fermi_token: str = '0') -> dict[int, Path]:
    """Map band index to its frequency-vs-angle file for one Fermi-energy token.

    Parameters
    ----------
    skeaf_dir : str or Path
        Folder holding ``qo_EF_<token>_freqvsangle_<band>.out[.gz]`` files.
    fermi_token : str, optional
        Fermi energy as written in the file names (eV), e.g. ``'0'`` or ``'-0.005'``.

    Returns
    -------
    dict[int, Path]
        Band index to file path, sorted by band.
    """
    found = {}
    for path in Path(skeaf_dir).iterdir():
        match = _QO_PATTERN.match(path.name)
        if match and match['token'] == fermi_token:
            found.setdefault(int(match['band']), path)
    return dict(sorted(found.items()))


def fermi_tokens(skeaf_dir: str | Path) -> list[str]:
    """Return every Fermi-energy token present in ``skeaf_dir``, sorted by energy."""
    tokens = {m['token'] for p in Path(skeaf_dir).iterdir() if (m := _QO_PATTERN.match(p.name))}
    return sorted(tokens, key=float)


def read_frequency_table(path: str | Path) -> pd.DataFrame:
    """Parse one frequency-vs-angle file into a table with derived columns.

    Parameters
    ----------
    path : str or Path
        ``qo_EF_*_freqvsangle_*.out`` file, optionally gzip-compressed.

    Returns
    -------
    pandas.DataFrame
        Columns of :data:`QO_COLUMNS` plus ``freq_T``, ``mstar_abs``,
        ``amp_factor`` (``|curvature|^(-1/2)``), ``carrier`` (``'e'``/``'h'``) and
        ``extremum`` (``'max'``/``'min'``). Empty or header-only files give an
        empty table, which signals a failed pyskeaf job rather than physics.
    """
    path = Path(path)
    if path.stat().st_size == 0:
        return pd.DataFrame(columns=QO_COLUMNS)
    values = np.loadtxt(path, delimiter=',', skiprows=1, ndmin=2)
    if values.size == 0:
        return pd.DataFrame(columns=QO_COLUMNS)
    table = pd.DataFrame(values, columns=QO_COLUMNS)
    table['freq_T'] = table.freq_kT * 1000.0
    table['mstar_abs'] = table.mstar_me.abs()
    table['amp_factor'] = 1.0 / np.sqrt(np.abs(table.curv_kTA2))
    table['carrier'] = np.where(table.orbit_type > 0, 'e', np.where(table.orbit_type < 0, 'h', '?'))
    table['extremum'] = np.where(table.curv_kTA2 < 0, 'max', 'min')
    return table


def read_orbits(
    skeaf_dir: str | Path,
    bands: list[int] | None = None,
    fermi_token: str = '0',
    band_groups: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Concatenate the frequency tables of several bands.

    Parameters
    ----------
    skeaf_dir : str or Path
        Folder with the pyskeaf output.
    bands : list[int] or None, optional
        Bands to read; ``None`` reads every band present.
    fermi_token : str, optional
        Fermi-energy token of the files.
    band_groups : dict[int, str] or None, optional
        Band index to degenerate-group label (e.g. ``{11: '11,12', 12: '11,12'}``).

    Returns
    -------
    pandas.DataFrame
        All orbits with ``band`` and ``group`` columns added.
    """
    files = find_frequency_files(skeaf_dir, fermi_token)
    if bands:
        files = {band: path for band, path in files.items() if band in set(bands)}
    frames = []
    for band, path in files.items():
        table = read_frequency_table(path)
        if not table.empty:
            frames.append(table.assign(band=band))
    if not frames:
        return pd.DataFrame(columns=QO_COLUMNS + ['band', 'group'])
    orbits = pd.concat(frames, ignore_index=True)
    labels = band_groups or {}
    orbits['group'] = orbits.band.map(lambda band: labels.get(band, str(band)))
    return orbits


def frequency_file_inventory(skeaf_dir: str | Path, fermi_token: str = '0') -> pd.DataFrame:
    """Rows, angular coverage and emptiness of every frequency file in a folder."""
    rows = []
    for band, path in find_frequency_files(skeaf_dir, fermi_token).items():
        table = read_frequency_table(path)
        covered = len(table) > 0
        rows.append(
            dict(
                band=band,
                file=path.name,
                rows=len(table),
                is_empty=not covered,
                azimuthal=f'{table.azimuthal_deg.min():.2f}-{table.azimuthal_deg.max():.2f}'
                if covered
                else '',
                polar=f'{table.polar_deg.min():.2f}-{table.polar_deg.max():.2f}' if covered else '',
                n_angles=int(table[['azimuthal_deg', 'polar_deg']].drop_duplicates().shape[0])
                if covered
                else 0,
            )
        )
    return pd.DataFrame(rows)


def sweep_angle_name(orbits: pd.DataFrame, requested: str = 'auto') -> str:
    """Return ``'polar'`` or ``'azimuthal'``: the angle that varies in the sweep."""
    if requested in ('polar', 'azimuthal'):
        return requested
    return 'polar' if orbits.polar_deg.nunique() >= orbits.azimuthal_deg.nunique() else 'azimuthal'


# --------------------------------------------------------------------------- BXSF grids
def find_bxsf_files(bxsf_dir: str | Path) -> dict[int, Path]:
    """Map band index to ``Fermi_surf_band_<n>.bxsf[.gz]`` in ``bxsf_dir``."""
    found = {}
    for path in Path(bxsf_dir).iterdir():
        match = re.fullmatch(r'Fermi_surf_band_(\d+)\.bxsf(?:\.gz)?', path.name)
        if match:
            found.setdefault(int(match[1]), path)
    return dict(sorted(found.items()))


def plain_bxsf_path(path: str | Path, cache_dir: str | Path) -> Path:
    """Return a plain-text BXSF path, decompressing ``.gz`` files into ``cache_dir``.

    Returns
    -------
    Path
        ``path`` itself for plain files; otherwise the decompressed copy (created
        once and reused).
    """
    path = Path(path)
    if path.suffix != '.gz':
        return path
    target = Path(cache_dir) / path.stem
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, 'rb') as source, open(target, 'wb') as destination:
            shutil.copyfileobj(source, destination)
    return target


def read_bxsf_grid(path: str | Path, cache_dir: str | Path) -> dict:
    """Read one BXSF band with ``PAOFLOW.pyskeaf`` and return the quantities used here.

    Returns
    -------
    dict
        ``energies`` (Ry, periodic end points removed), ``fermi_energy`` (Ry),
        ``recip_ang`` (reciprocal vectors in A^-1 including 2 pi), ``shape``,
        ``source_band`` (band label in the file, if any) and ``plain_path``.
    """
    from PAOFLOW.pyskeaf.io_bxsf import read_bxsf

    plain = plain_bxsf_path(path, cache_dir)
    data = read_bxsf(plain)
    with open(plain) as handle:
        head = handle.read(4000)
    label = re.search(r'BAND:\s+(\S+)', head)
    return dict(
        energies=np.asarray(data.energies)[:-1, :-1, :-1],
        fermi_energy=float(data.fermi_energy),
        recip_ang=np.asarray(data.recip_ang),
        shape=(data.nx, data.ny, data.nz),
        source_band=label[1] if label else '',
        plain_path=plain,
    )
