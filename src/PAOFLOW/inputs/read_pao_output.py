import pandas as pd
from pathlib import Path


def read_band_path_PAO(fname):
    """Read a PAOFLOW k-path file and return tick positions and labels.

    Parameters
    ----------
    fname : str
        Path to the ``kpath_points.txt`` file written by PAOFLOW.

    Returns
    -------
    findex : list of int
        Cumulative k-point indices for each high-symmetry point.
    ftags : list of str
        Labels for each high-symmetry point (``'G'`` is replaced by
        ``r'$\\Gamma$'``).
    """

    tags = []
    npnts = []
    with open(fname, 'r') as f:
        ls = f.readline().split()
        try:
            float(ls[0])
        except (ValueError, IndexError):
            pass
        else:
            raise ValueError(
                '%s holds "position label" lines (sparse backend); read it with '
                'PAOFLOW.results.loaders.read_kpath' % fname
            )
        while not len(ls) == 0:
            tags.append(ls[0])
            npnts.append(int(ls[1]))
            ls = f.readline().split()

    ftags = []
    findex = [0]
    for i in range(len(tags) - 1):
        if tags[i] == 'G' or tags[i] == 'gG':
            tags[i] = r'$\Gamma$'

        if npnts[i] == 0:
            ftags[-1] += '|' + tags[i]
        else:
            ftags.append(tags[i])
            findex.append(npnts[i] + findex[-1])
    ftags.append(tags[-1] if not tags[-1] == 'G' else r'$\Gamma$')

    return findex, ftags


def read_dos_PAO(fname):
    """Read a two-column PAOFLOW output file (DOS, PDOS, Hall and optical spectra).

    The file is parsed by :func:`PAOFLOW.results.columns.read_columns`, the reader the browser
    interface uses as well, so ``#`` header lines (``write_file_row_col_units``, ``np.savetxt``)
    are accepted.

    Parameters
    ----------
    fname : str
        Path to the file (first column energy in eV, second column the value).

    Returns
    -------
    es : np.ndarray
        Energy values (eV).
    dos : np.ndarray
        The second column.

    Raises
    ------
    ValueError
        If the file has fewer than two numeric columns.
    """
    from ..results.columns import read_columns

    data, _header = read_columns(fname)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
        raise ValueError('%s: expected two numeric columns' % fname)
    return data[:, 0].copy(), data[:, 1].copy()


def read_bands_PAO(fname):
    """Read a PAOFLOW band structure file.

    Parameters
    ----------
    fname : str
        Path to the band file (each row: k-index followed by band energies).

    Returns
    -------
    np.ndarray, shape ``(nbnd, nkpts)``
        Band energies (eV) with rows indexed by band and columns by k-point.

    Raises
    ------
    ValueError
        If the file holds no band table.
    """
    from ..results.columns import read_columns

    data, _header = read_columns(fname)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 2:
        raise ValueError('%s: expected a k index and the band energies on every row' % fname)
    return data[:, 1:].T.copy()


def read_site_projected(path: Path) -> pd.DataFrame:
    """Read a PAOFLOW site-projected band structure file into a DataFrame.

    Parameters
    ----------
    path : Path
        Path to the whitespace-separated file with columns:
        k-index, eigenvalue, and site projection weight.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``'kindex'``, ``'eigenvalue'``,
        ``'site_weight'``; rows with non-numeric entries are dropped.
    """
    df = pd.read_csv(path, sep=r'\s+', header=None)

    df = df.iloc[:, :3].copy()
    df.columns = ['kindex', 'eigenvalue', 'site_weight']

    for c in ['kindex', 'eigenvalue', 'site_weight']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    return df.dropna(subset=['kindex', 'eigenvalue', 'site_weight']).reset_index(drop=True)


def read_transport_PAO(fname):
    """Read a PAOFLOW transport tensor output file.

    Parameters
    ----------
    fname : str
        Path to the transport output file.  Each row contains temperature,
        energy, and six independent tensor components (diagonal and
        off-diagonal: xx, yy, zz, xy, xz, yz), one block of rows per temperature.

    Returns
    -------
    enes : np.ndarray, shape ``(nene,)``
        Energy grid (eV).
    temps : np.ndarray, shape ``(ntemp,)``
        Temperature grid (K).
    tensors : np.ndarray, shape ``(ntemp, nene, 3, 3)``
        Symmetric transport tensor at each temperature and energy.

    Raises
    ------
    ValueError
        If the file is not a tensor table (``hall_trace`` holds T, E and one value; ``hall`` and
        ``nernst`` hold 9 and 27 components).
    """
    import numpy as np

    from ..results.columns import read_columns

    data, _header = read_columns(fname)
    if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != 8:
        width = data.shape[1] if data.ndim == 2 and data.size else 0
        raise ValueError(
            '%s: expected T, E and six tensor components per row, found %d columns '
            '(hall_trace, hall and nernst files have other layouts)' % (fname, width)
        )
    first = data[:, 0]
    nene = int(np.argmax(first != first[0])) if np.any(first != first[0]) else len(first)
    ntemp = -(-len(first) // nene)
    enes = data[:nene, 1].copy()
    temps = first[::nene][:ntemp].copy()
    tensors = np.empty((ntemp, nene, 3, 3), dtype=float)
    for i in range(ntemp):
        block = data[i * nene : (i + 1) * nene]
        for k in range(3):
            tensors[i, : len(block), k, k] = block[:, 2 + k]
        for ik, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            tensors[i, : len(block), a, b] = tensors[i, : len(block), b, a] = block[:, 5 + ik]
    return enes, temps, tensors
