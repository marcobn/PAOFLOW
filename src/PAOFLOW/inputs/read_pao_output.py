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
    """Read a two-column PAOFLOW DOS output file.

    Parameters
    ----------
    fname : str
        Path to the DOS file (first column energy in eV, second column DOS).

    Returns
    -------
    es : np.ndarray
        Energy values (eV).
    dos : np.ndarray
        Density of states.
    """
    import numpy as np

    es = []
    dos = []
    with open(fname, 'r') as f:
        for l in f.readlines():
            ls = l.split()
            es.append(float(ls[0]))
            dos.append(float(ls[1]))

    es = np.array(es)
    dos = np.array(dos)

    return es, dos


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
    """
    import numpy as np

    bands = []
    with open(fname, 'r') as f:
        for l in f.readlines():
            bands.append([float(v) for v in l.split()[1:]])

    return np.array(bands).T


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
        off-diagonal: xx, yy, zz, xy, xz, yz).

    Returns
    -------
    enes : np.ndarray, shape ``(nene,)``
        Energy grid (eV).
    temps : np.ndarray, shape ``(ntemp,)``
        Temperature grid (K).
    tensors : np.ndarray, shape ``(ntemp, nene, 3, 3)``
        Symmetric transport tensor at each temperature and energy.
    """
    import numpy as np

    nene = 0
    ntemp = 0
    enes = temps = tensors = None

    with open(fname, 'r') as f:
        lines = f.readlines()

        nl = len(lines)
        ftemp = float(lines[ntemp].split()[0])
        ptemp = ftemp
        while ftemp == ptemp and nene < nl - 1:
            nene += 1
            ptemp = float(lines[nene].split()[0])
        if nene == nl - 1:
            nene += 1

        while ntemp * nene < nl:
            ntemp += 1

        enes = np.empty(nene, dtype=float)
        temps = np.empty(ntemp, dtype=float)
        tensors = np.empty((ntemp, nene, 3, 3), dtype=float)

        iL = 0
        for i in range(ntemp):
            temps[i] = float(lines[iL].split()[0])
            for j in range(nene):
                ls = lines[iL].split()
                if i == 0:
                    enes[j] = float(ls[1])
                for k in range(3):
                    tensors[i, j, k, k] = float(ls[2 + k])
                for ik, k in enumerate([(0, 1), (0, 2), [1, 2]]):
                    tensors[i, j, k[0], k[1]] = tensors[i, j, k[1], k[0]] = float(ls[5 + ik])
                iL += 1

        return enes, temps, tensors
