"""A scalar field on a general grid, with the atoms, in the XCrySDen format (XSF) read by XCrySDen and VESTA."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.utils.constants import BOHR_RADIUS_ANGS


def write_grid_xsf(
    path: str,
    origin: NDArray[np.float64],
    spanning_vectors: NDArray[np.float64],
    values: NDArray[np.float64],
    symbols: Sequence[str],
    positions: NDArray[np.float64],
    periodic_axes: Sequence[int] = (),
    name: str = 'field',
) -> None:
    """Write a scalar field on a grid and the atoms (molecule form, coordinates in angstrom).

    Parameters
    ----------
    path : str
    origin : ndarray, shape ``(3,)``
        Cartesian origin of the grid (bohr).
    spanning_vectors : ndarray, shape ``(3, 3)``
        The three edges of the grid box as rows (bohr).
    values : ndarray, shape ``(n_1, n_2, n_3)``
        Field values; point :math:`(i, j, k)` is at origin + (i, j, k) / (n - 1) along the edges, except along the
        periodic axes, where the grid is :math:`i/n` and the first plane is repeated at the end as XSF requires.
    symbols : sequence of str
        Element of every atom.
    positions : ndarray, shape ``(n_atoms, 3)``
        Cartesian positions (bohr).
    periodic_axes : sequence of int, optional
        Axes of ``values`` that are periodic over the box edge.
    name : str, optional
        Label of the data block.

    Returns
    -------
    None
    """
    data = np.asarray(values, dtype=float)
    for axis in periodic_axes:
        data = np.concatenate([data, np.take(data, [0], axis=axis)], axis=axis)
    origin_angstrom = np.asarray(origin, dtype=float) * BOHR_RADIUS_ANGS
    vectors_angstrom = np.asarray(spanning_vectors, dtype=float) * BOHR_RADIUS_ANGS
    positions_angstrom = np.asarray(positions, dtype=float).reshape(-1, 3) * BOHR_RADIUS_ANGS
    with open(path, 'w') as handle:
        handle.write('ATOMS\n')
        for symbol, position in zip(symbols, positions_angstrom):
            handle.write(
                f'{str(symbol):<3s} {position[0]:14.8f} {position[1]:14.8f} {position[2]:14.8f}\n'
            )
        handle.write(f'BEGIN_BLOCK_DATAGRID_3D\n {name}\n BEGIN_DATAGRID_3D_{name}\n')
        handle.write('  {:d} {:d} {:d}\n'.format(*data.shape))
        handle.write('  {:14.8f} {:14.8f} {:14.8f}\n'.format(*origin_angstrom))
        for vector in vectors_angstrom:
            handle.write('  {:14.8f} {:14.8f} {:14.8f}\n'.format(*vector))
        flat = data.transpose(2, 1, 0).ravel()  # the first index runs fastest
        for start in range(0, flat.size, 6):
            handle.write(' '.join(f'{value:14.6e}' for value in flat[start : start + 6]) + '\n')
        handle.write(' END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n')
