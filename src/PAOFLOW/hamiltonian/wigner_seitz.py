"""Wigner-Seitz Fourier interpolation of the real-space PAO Hamiltonian.

``HRs`` holds the PAO Hamiltonian on the ``N1 x N2 x N3`` grid of lattice vectors that is Fourier-conjugate to the
DFT mesh; it is periodic on the supercell ``N1 a1 x N2 a2 x N3 a3``. Off the mesh, every grid index ``m`` must be
given a lattice vector. The Wigner-Seitz choice gives it the shortest image ``n = m + N s`` (in Cartesian length) and
shares the weight equally among images of equal length, as Wannier interpolation does:

.. math::

    H(\\mathbf k) = \\sum_{\\mathbf m}\\sum_{\\mathbf n\\in\\mathrm{WS}(\\mathbf m)} w_{\\mathbf n}\\,
        e^{-2\\pi i\\,\\mathbf k\\cdot\\mathbf n}\\,\\mathtt{HRs}[\\mathbf m],
    \\qquad \\sum_{\\mathbf n\\in\\mathrm{WS}(\\mathbf m)} w_{\\mathbf n}=1,

with ``k`` in crystal coordinates (PAOFLOW's convention, ``H(k) = sum_n exp(-2 pi i k.n) HRs[n]``). It reproduces
the DFT mesh exactly, like the zero padding of :func:`PAOFLOW.utils.zero_pad.zero_pad`, and differs from it off the
mesh only through the hoppings near the edge of the supercell.

On a uniform fine grid ``k = j / M`` the phase depends on ``n`` only modulo ``M``. The Wigner-Seitz interpolant on
that grid is therefore one FFT of the weighted hoppings placed at ``n mod M`` (:func:`wigner_seitz_fine_grid_array`):
the same cost as the zero padding. When every image fits in the range of lattice vectors of the fine grid
(``-M/2 <= n < M/2``, :func:`check_fine_grid_holds_images`), the fine-grid real-space arrays that the gradient and
curvature routines rebuild by inverse FFT carry each hopping at its own Wigner-Seitz vector, so the velocities are the
exact derivatives of the interpolated bands.

The pair-resolved route (``interpolation='wigner_seitz_pairs'``, :func:`wigner_seitz_pair_images`) chooses the image
of the element ``(i, j)`` by the length of the bond it describes, :math:`-\\mathbf n + \\boldsymbol\\tau_j -
\\boldsymbol\\tau_i` (crystal coordinates, Cartesian length), instead of the cell vector :math:`\\mathbf n` alone, as
wannier90 ``use_ws_distance`` and EPW ``use_ws`` do. With several atoms per cell, the operations of the space group
that move one atom onto another (with a fractional translation, e.g. the fourfold rotations of diamond) then leave
the interpolated bands invariant; the cell-based images keep only the operations that fix every atom. Both routes
cost the same on a fine grid: one FFT per orbital pair, with the images of that pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ..DataController import DataController

SUPPORTED_INTERPOLATIONS = ('zero_padding', 'wigner_seitz', 'wigner_seitz_pairs')
SUPPORTED_BAND_INTERPOLATIONS = ('fft_grid', 'wigner_seitz', 'wigner_seitz_pairs')


@dataclass
class WignerSeitzImages:
    """Wigner-Seitz images of the grid of lattice vectors of ``HRs``.

    Attributes
    ----------
    cells : ndarray, shape ``(n_images, 3)``, int
        Lattice vectors ``n`` (crystal coordinates) of the images.
    weights : ndarray, shape ``(n_images,)``
        Degeneracy weights ``1 / deg``; they sum to ``N1 N2 N3``.
    grid_index : ndarray, shape ``(n_images, 3)``, int
        Index ``n mod N`` of each image into the ``HRs`` grid.
    grid_shape : tuple of int
        ``(N1, N2, N3)``.
    """

    cells: NDArray[np.int_]
    weights: NDArray[np.float64]
    grid_index: NDArray[np.int_]
    grid_shape: tuple[int, int, int]


def wigner_seitz_images(
    grid_shape: tuple[int, int, int] | NDArray[np.int_],
    a_vectors: NDArray[np.float64],
    span: int = 2,
    tolerance: float = 1.0e-6,
    separation_offset: NDArray[np.float64] | None = None,
) -> WignerSeitzImages:
    """Shortest images of every grid index of ``HRs`` in the supercell lattice.

    Parameters
    ----------
    grid_shape : tuple of int
        ``(N1, N2, N3)``, the shape of the lattice-vector grid (the DFT mesh).
    a_vectors : ndarray, shape ``(3, 3)``
        Direct lattice vectors as rows, any length unit (only ratios enter).
    span : int, optional
        Supercell translations ``s`` in ``[-span, span]`` searched around the wrapped index (2 covers strongly
        skewed cells).
    tolerance : float, optional
        Relative tolerance on the squared length for two images to count as degenerate.
    separation_offset : ndarray, shape ``(3,)``, optional
        Cartesian vector ``d`` (units of ``a_vectors``): the images minimise :math:`|\\mathbf n A - \\mathbf d|`
        instead of :math:`|\\mathbf n A|`. With ``d`` the separation of two orbital centres,
        :math:`(\\boldsymbol\\tau_j - \\boldsymbol\\tau_i) A`, these are the images of the pair ``(i, j)``
        (:func:`wigner_seitz_pair_images`). ``None`` (default) is the cell-based assignment.

    Returns
    -------
    WignerSeitzImages
        Images, weights and grid indices, grid index by grid index in C order.

    Notes
    -----
    The image ``n = m + N s`` of the grid index ``m`` lies in the Wigner-Seitz cell of the supercell lattice
    ``{N s}`` when it is a shortest one, :math:`|\\mathbf n| \\le |\\mathbf n - N\\mathbf s'|` for every ``s'``; the
    images of equal length share the weight. This is the criterion of QE ``wsweight`` and of
    :func:`PAOFLOW.elphon.elph_bloch._ws_lattice`, evaluated per grid index (``N1 N2 N3 (2 span + 1)^3`` candidates)
    instead of on a full candidate box, so its memory grows linearly with the grid.
    """
    grid = np.asarray(grid_shape, dtype=int).reshape(3)
    lattice = np.asarray(a_vectors, dtype=float)
    grid_points = np.stack(
        np.meshgrid(*[np.arange(n) for n in grid], indexing='ij'), axis=-1
    ).reshape(-1, 3)
    wrapped = grid_points - grid * (grid_points >= (grid + 1) // 2)
    shifts = np.stack(
        np.meshgrid(*[np.arange(-span, span + 1)] * 3, indexing='ij'), axis=-1
    ).reshape(-1, 3)
    candidates = wrapped[:, None, :] + (shifts * grid)[None, :, :]
    cartesian = candidates @ lattice
    if separation_offset is not None:
        cartesian = cartesian - np.asarray(separation_offset, dtype=float).reshape(1, 1, 3)
    squared_length = np.einsum('pic,pic->pi', cartesian, cartesian)
    shortest = squared_length.min(axis=1, keepdims=True)
    scale = max(float(np.max(np.einsum('ic,ic->i', lattice, lattice))), 1.0e-300)
    selected = squared_length <= shortest * (1.0 + tolerance) + tolerance * scale
    degeneracy = selected.sum(axis=1)
    return WignerSeitzImages(
        cells=candidates[selected].astype(int),
        weights=np.repeat(1.0 / degeneracy, degeneracy),
        grid_index=np.repeat(grid_points, degeneracy, axis=0).astype(int),
        grid_shape=(int(grid[0]), int(grid[1]), int(grid[2])),
    )


def wigner_seitz_hamiltonian_k(
    hamiltonian_r: NDArray[np.complex128],
    images: WignerSeitzImages,
    k_crystal: NDArray[np.floating] | NDArray[np.complexfloating],
    block_size: int = 256,
) -> NDArray[np.complex128]:
    """Bloch Hamiltonian at arbitrary k by the Wigner-Seitz Fourier sum.

    Parameters
    ----------
    hamiltonian_r : ndarray, shape ``(nawf, nawf, N1, N2, N3, nspin)``, complex
        PAOFLOW's ``HRs`` (eV).
    images : WignerSeitzImages
        From :func:`wigner_seitz_images` for the grid of ``hamiltonian_r``.
    k_crystal : ndarray, shape ``(nk, 3)``
        k-points in crystal coordinates (complex values are allowed, as for complex band structures).
    block_size : int, optional
        Number of k-points per phase block (bounds the memory to ``block_size x n_images``).

    Returns
    -------
    ndarray, shape ``(nawf, nawf, nk, nspin)``, complex
        :math:`H(\\mathbf k)`, Hermitised for real k; the layout of
        :func:`PAOFLOW.spectrum.do_bands.band_loop_H`.
    """
    nawf, _, n1, n2, n3, nspin = hamiltonian_r.shape
    if (n1, n2, n3) != images.grid_shape:
        raise ValueError(
            f'HRs grid {(n1, n2, n3)} does not match the images grid {images.grid_shape}'
        )
    k_points = np.atleast_2d(np.asarray(k_crystal))
    gi = images.grid_index
    weighted_hoppings = (
        hamiltonian_r[:, :, gi[:, 0], gi[:, 1], gi[:, 2], :] * images.weights[None, None, :, None]
    )  # (nawf, nawf, n_images, nspin)
    weighted_hoppings = np.moveaxis(weighted_hoppings, 2, 0).reshape(len(images.weights), -1)
    return _hermitian_hamiltonian_k(
        _phase_sum(weighted_hoppings, images.cells, k_points, block_size), nawf, nspin, k_points
    )


def _phase_sum(
    weighted_hoppings: NDArray[np.complex128],
    cells: NDArray[np.int_],
    k_points: NDArray[np.floating] | NDArray[np.complexfloating],
    block_size: int,
) -> NDArray[np.complex128]:
    """``sum_n exp(-2 pi i k.n) W[n, :]`` for blocks of k-points.

    Parameters
    ----------
    weighted_hoppings : ndarray, shape ``(n_cells, n_columns)``, complex
        Weighted hoppings ``W``, one row per lattice vector.
    cells : ndarray, shape ``(n_cells, 3)``, int
        Lattice vectors ``n`` (crystal coordinates).
    k_points : ndarray, shape ``(nk, 3)``
        Crystal coordinates.
    block_size : int
        k-points per phase block.

    Returns
    -------
    ndarray, shape ``(nk, n_columns)``, complex
    """
    phase_sum = np.empty((k_points.shape[0], weighted_hoppings.shape[1]), dtype=complex)
    for start in range(0, k_points.shape[0], block_size):
        stop = min(start + block_size, k_points.shape[0])
        phases = np.exp(-2j * np.pi * (k_points[start:stop] @ cells.T))
        phase_sum[start:stop] = phases @ weighted_hoppings
    return phase_sum


def _hermitian_hamiltonian_k(
    phase_sum: NDArray[np.complex128],
    nawf: int,
    nspin: int,
    k_points: NDArray[np.floating] | NDArray[np.complexfloating],
) -> NDArray[np.complex128]:
    """Reshape ``(nk, nawf * nawf * nspin)`` to ``(nawf, nawf, nk, nspin)``; Hermitise for real k.

    Parameters
    ----------
    phase_sum : ndarray, shape ``(nk, nawf * nawf * nspin)``, complex
    nawf, nspin : int
    k_points : ndarray, shape ``(nk, 3)``

    Returns
    -------
    ndarray, shape ``(nawf, nawf, nk, nspin)``, complex
    """
    hamiltonian_k = np.moveaxis(phase_sum.reshape(-1, nawf, nawf, nspin), 0, 2)
    if np.isrealobj(k_points) or not np.iscomplex(k_points).any():
        hamiltonian_k = 0.5 * (hamiltonian_k + np.conj(np.swapaxes(hamiltonian_k, 0, 1)))
    return np.ascontiguousarray(hamiltonian_k)


@dataclass
class PairWignerSeitzImages:
    """Wigner-Seitz images chosen pair by pair of orbital centres.

    Attributes
    ----------
    cells : ndarray, shape ``(n_cells, 3)``, int
        Union over the site pairs of the image lattice vectors ``n``.
    grid_index : ndarray, shape ``(n_cells, 3)``, int
        ``n mod N`` of each cell.
    weights : ndarray, shape ``(n_cells, n_sites, n_sites)``
        Weight of each cell for the pair of sites ``(I, J)``; zero when the cell is not one of the pair's images.
        The weights of every pair sum to ``N1 N2 N3``.
    site_of_orbital : ndarray, shape ``(nawf,)``, int
        Site (distinct orbital centre) of each orbital.
    grid_shape : tuple of int
        ``(N1, N2, N3)``.
    """

    cells: NDArray[np.int_]
    grid_index: NDArray[np.int_]
    weights: NDArray[np.float64]
    site_of_orbital: NDArray[np.int_]
    grid_shape: tuple[int, int, int]

    def images_of_pair(self, site_i: int, site_j: int) -> WignerSeitzImages:
        """The images of one pair of sites, as :class:`WignerSeitzImages`.

        Parameters
        ----------
        site_i, site_j : int
            Sites of the row and the column orbital.

        Returns
        -------
        WignerSeitzImages
        """
        kept = self.weights[:, site_i, site_j] > 0.0
        return WignerSeitzImages(
            cells=self.cells[kept],
            weights=self.weights[kept, site_i, site_j],
            grid_index=self.grid_index[kept],
            grid_shape=self.grid_shape,
        )

    def all_cells(self) -> WignerSeitzImages:
        """Every cell of every pair (weights not meaningful), for :func:`check_fine_grid_holds_images`.

        Returns
        -------
        WignerSeitzImages
        """
        return WignerSeitzImages(
            cells=self.cells,
            weights=np.ones(len(self.cells)),
            grid_index=self.grid_index,
            grid_shape=self.grid_shape,
        )


def wigner_seitz_pair_images(
    grid_shape: tuple[int, int, int] | NDArray[np.int_],
    a_vectors: NDArray[np.float64],
    orbital_positions_cryst: NDArray[np.float64],
    span: int = 2,
    tolerance: float = 1.0e-6,
) -> PairWignerSeitzImages:
    """Wigner-Seitz images of ``HRs`` chosen by the bond of every pair of orbital centres.

    Parameters
    ----------
    grid_shape : tuple of int
        ``(N1, N2, N3)``, the shape of the lattice-vector grid (the DFT mesh).
    a_vectors : ndarray, shape ``(3, 3)``
        Direct lattice vectors as rows (alat units).
    orbital_positions_cryst : ndarray, shape ``(nawf, 3)``
        Centre of every orbital, crystal coordinates (:func:`orbital_centres_for_pairs`).
    span : int, optional
        Supercell translations searched around the wrapped index (see :func:`wigner_seitz_images`).
    tolerance : float, optional
        Relative tolerance for degenerate images.

    Returns
    -------
    PairWignerSeitzImages

    Notes
    -----
    PAOFLOW's ``HRs[n]`` holds the hopping from orbital ``i`` in the home cell to orbital ``j`` in the cell
    :math:`-\\mathbf n`, a bond :math:`\\mathbf d = (-\\mathbf n + \\boldsymbol\\tau_j - \\boldsymbol\\tau_i) A`.
    For the pair of sites ``(I, J)`` the grid index ``m`` goes to the images :math:`\\mathbf n = \\mathbf m +
    N\\mathbf s` of shortest :math:`|\\mathbf d|`, i.e. :func:`wigner_seitz_images` with
    ``separation_offset`` :math:`(\\boldsymbol\\tau_J - \\boldsymbol\\tau_I) A`; images of equal length share the
    weight. This is the criterion of :func:`PAOFLOW.elphon.elph_bloch._ws_lattice_pairs` (``sign=-1``), searched
    per grid index so that the memory grows linearly with the grid. Orbitals with the same centre share a site, so
    a cell with one atom gives the cell-based images of :func:`wigner_seitz_images`.
    """
    lattice = np.asarray(a_vectors, dtype=float)
    positions = np.asarray(orbital_positions_cryst, dtype=float).reshape(-1, 3)
    sites, site_of_orbital = np.unique(np.round(positions, 8), axis=0, return_inverse=True)
    site_of_orbital = np.asarray(site_of_orbital).reshape(-1)
    n_sites = len(sites)
    cell_index: dict[tuple[int, int, int], int] = {}
    cells, grid_index, entries = [], [], []
    for site_i in range(n_sites):
        for site_j in range(n_sites):
            images = wigner_seitz_images(
                grid_shape,
                lattice,
                span=span,
                tolerance=tolerance,
                separation_offset=(sites[site_j] - sites[site_i]) @ lattice,
            )
            for cell, grid_point, weight in zip(
                images.cells.tolist(), images.grid_index.tolist(), images.weights
            ):
                key = (cell[0], cell[1], cell[2])
                if key not in cell_index:
                    cell_index[key] = len(cells)
                    cells.append(cell)
                    grid_index.append(grid_point)
                entries.append((cell_index[key], site_i, site_j, weight))
    weights = np.zeros((len(cells), n_sites, n_sites))
    for row, site_i, site_j, weight in entries:
        weights[row, site_i, site_j] = weight
    grid = np.asarray(grid_shape, dtype=int).reshape(3)
    return PairWignerSeitzImages(
        cells=np.asarray(cells, dtype=int).reshape(-1, 3),
        grid_index=np.asarray(grid_index, dtype=int).reshape(-1, 3),
        weights=weights,
        site_of_orbital=site_of_orbital.astype(int),
        grid_shape=(int(grid[0]), int(grid[1]), int(grid[2])),
    )


def wigner_seitz_pair_hamiltonian_k(
    hamiltonian_r: NDArray[np.complex128],
    pair_images: PairWignerSeitzImages,
    k_crystal: NDArray[np.floating] | NDArray[np.complexfloating],
    block_size: int = 256,
) -> NDArray[np.complex128]:
    """Bloch Hamiltonian at arbitrary k by the pair-resolved Wigner-Seitz Fourier sum.

    Parameters
    ----------
    hamiltonian_r : ndarray, shape ``(nawf, nawf, N1, N2, N3, nspin)``, complex
        PAOFLOW's ``HRs`` (eV).
    pair_images : PairWignerSeitzImages
        From :func:`wigner_seitz_pair_images` for the grid and the orbitals of ``hamiltonian_r``.
    k_crystal : ndarray, shape ``(nk, 3)``
        k-points in crystal coordinates (complex values are allowed).
    block_size : int, optional
        Number of k-points per phase block.

    Returns
    -------
    ndarray, shape ``(nawf, nawf, nk, nspin)``, complex
        :math:`H(\\mathbf k)`, Hermitised for real k; the layout of
        :func:`PAOFLOW.spectrum.do_bands.band_loop_H`.

    Raises
    ------
    ValueError
        If the grid or the number of orbitals does not match ``pair_images``.

    Notes
    -----
    .. math::

        H_{ij}(\\mathbf k) = \\sum_{\\mathbf n} w^{IJ}_{\\mathbf n}\\, e^{-2\\pi i\\,\\mathbf k\\cdot\\mathbf n}\\,
            \\mathtt{HRs}_{ij}[\\mathbf n \\bmod N],

    with ``I``, ``J`` the sites of ``i``, ``j``: one product of the phases of the union of the cells with the
    hoppings weighted pair by pair.
    """
    nawf, _, n1, n2, n3, nspin = hamiltonian_r.shape
    if (n1, n2, n3) != pair_images.grid_shape:
        raise ValueError(
            f'HRs grid {(n1, n2, n3)} does not match the images grid {pair_images.grid_shape}'
        )
    if len(pair_images.site_of_orbital) != nawf:
        raise ValueError(
            f'HRs has {nawf} orbitals, the pair images {len(pair_images.site_of_orbital)} centres'
        )
    k_points = np.atleast_2d(np.asarray(k_crystal))
    gi = pair_images.grid_index
    site = pair_images.site_of_orbital
    pair_weights = pair_images.weights[:, site[:, None], site[None, :]]  # (n_cells, nawf, nawf)
    weighted_hoppings = np.moveaxis(hamiltonian_r[:, :, gi[:, 0], gi[:, 1], gi[:, 2], :], 2, 0)
    weighted_hoppings = (weighted_hoppings * pair_weights[..., None]).reshape(len(gi), -1)
    return _hermitian_hamiltonian_k(
        _phase_sum(weighted_hoppings, pair_images.cells, k_points, block_size),
        nawf,
        nspin,
        k_points,
    )


def orbital_centres_for_pairs(data_controller: DataController) -> NDArray[np.float64]:
    """Orbital centres of a run for the pair-resolved interpolation, checked against ``nawf``.

    Parameters
    ----------
    data_controller : DataController
        A QE run after ``projections`` (``arrays['basis']``) or a tight-binding model (``arrays['tau', 'naw']``).

    Returns
    -------
    ndarray, shape ``(nawf, 3)``
        Centres in crystal coordinates (:func:`PAOFLOW.hamiltonian.bloch_derivatives.orbital_positions_from_data`).

    Raises
    ------
    ValueError
        If the run has no orbital centres, or not one per orbital: after ``doubling_Hamiltonian`` or
        ``adhoc_spin_orbit`` the stored centres describe the original orbitals only.
    """
    from .bloch_derivatives import orbital_positions_from_data

    nawf = int(data_controller.data_attributes['nawf'])
    positions = orbital_positions_from_data(data_controller)
    if positions is None or len(positions) != nawf:
        found = 'no orbital centres' if positions is None else '%d orbital centres' % len(positions)
        raise ValueError(
            "interpolation='wigner_seitz_pairs' needs one centre per orbital; the run has %s for "
            'nawf = %d (the centres are not updated by doubling_Hamiltonian or adhoc_spin_orbit). '
            "Use interpolation='wigner_seitz'." % (found, nawf)
        )
    return np.asarray(positions, dtype=float)


def fine_grid_representatives(
    cells: NDArray[np.int_], fine_shape: tuple[int, int, int]
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Fine-grid index ``n mod M`` of each image and the lattice vector the fine grid assigns to that index.

    Parameters
    ----------
    cells : ndarray, shape ``(n_images, 3)``, int
        Lattice vectors of the images.
    fine_shape : tuple of int
        ``(M1, M2, M3)``.

    Returns
    -------
    fine_index : ndarray, shape ``(n_images, 3)``, int
        ``n mod M``.
    representative : ndarray, shape ``(n_images, 3)``, int
        The vector of that index in :func:`PAOFLOW.utils.get_R_grid_fft.get_R_grid_fft`: ``i`` if ``2 i < M``,
        ``i - M`` otherwise.
    """
    fine = np.asarray(fine_shape, dtype=int)
    fine_index = np.mod(cells, fine)
    representative = np.where(2 * fine_index >= fine, fine_index - fine, fine_index)
    return fine_index, representative


def check_fine_grid_holds_images(
    images: WignerSeitzImages, fine_shape: tuple[int, int, int]
) -> None:
    """Raise if the fine grid cannot hold every Wigner-Seitz image at its own lattice vector.

    Parameters
    ----------
    images : WignerSeitzImages
    fine_shape : tuple of int
        ``(M1, M2, M3)``, the interpolation grid.

    Raises
    ------
    ValueError
        If some image ``n`` falls outside the lattice vectors of the fine grid (``-M/2 <= n < M/2``). The energies
        on the grid would still be right, but the gradients that PAOFLOW computes from the fine-grid real-space
        array would belong to a different interpolant. The message gives the smallest grid that works.
    """
    _, representative = fine_grid_representatives(images.cells, fine_shape)
    if np.array_equal(representative, images.cells):
        return
    extent = np.abs(images.cells).max(axis=0)
    needed = [max(int(m), 2 * int(e) + 1) for m, e in zip(fine_shape, extent)]
    raise ValueError(
        'Wigner-Seitz interpolation: the grid %s cannot hold the Wigner-Seitz images of the %s mesh '
        '(they reach |n| = %s); use nfft >= %s, or interpolation="zero_padding".'
        % (
            tuple(int(m) for m in fine_shape),
            images.grid_shape,
            tuple(int(e) for e in extent),
            tuple(needed),
        )
    )


@dataclass
class FineGridPlacement:
    """Where the Wigner-Seitz images go on a fine grid.

    Attributes
    ----------
    flat_index : ndarray, shape ``(n_images,)``, int
        Flat C-order index of ``n mod M`` on the fine grid.
    unique : bool
        True when no two images share a fine-grid index (direct assignment is then exact).
    fine_shape : tuple of int
        ``(M1, M2, M3)``.
    """

    flat_index: NDArray[np.int_]
    unique: bool
    fine_shape: tuple[int, int, int]


def fine_grid_placement(
    images: WignerSeitzImages, fine_shape: tuple[int, int, int]
) -> FineGridPlacement:
    """Fine-grid positions of the Wigner-Seitz images (computed once per interpolation).

    Parameters
    ----------
    images : WignerSeitzImages
    fine_shape : tuple of int
        ``(M1, M2, M3)``.

    Returns
    -------
    FineGridPlacement
    """
    shape = tuple(int(m) for m in fine_shape)
    fine_index, _ = fine_grid_representatives(images.cells, shape)
    flat_index = np.ravel_multi_index(tuple(fine_index.T), shape)
    return FineGridPlacement(
        flat_index=flat_index,
        unique=bool(np.unique(flat_index).size == flat_index.size),
        fine_shape=shape,
    )


def wigner_seitz_fine_grid_array(
    hamiltonian_r: NDArray[np.complex128],
    images: WignerSeitzImages,
    placement: FineGridPlacement,
) -> NDArray[np.complex128]:
    """Weighted Wigner-Seitz hoppings of one orbital pair placed on the fine grid, ready for a forward FFT.

    Parameters
    ----------
    hamiltonian_r : ndarray, shape ``(N1, N2, N3)``, complex
        Real-space Hamiltonian of one orbital pair and spin.
    images : WignerSeitzImages
        For the grid ``(N1, N2, N3)``.
    placement : FineGridPlacement
        From :func:`fine_grid_placement`.

    Returns
    -------
    ndarray, shape ``(M1, M2, M3)``, complex
        Array ``A`` with ``A[n mod M] += w_n HRs[n mod N]``; ``fftn`` gives :math:`H(\\mathbf k_j)` at
        ``k_j = j / M`` with the Wigner-Seitz sum.

    Notes
    -----
    Colliding images (two images with the same ``n mod M``) are added, which is still the exact Wigner-Seitz sum
    on the fine grid; :func:`check_fine_grid_holds_images` rules them out where gradients follow.
    """
    gi = images.grid_index
    values = hamiltonian_r[gi[:, 0], gi[:, 1], gi[:, 2]] * images.weights
    n_fine = int(np.prod(placement.fine_shape))
    if placement.unique:
        fine_flat = np.zeros(n_fine, dtype=complex)
        fine_flat[placement.flat_index] = values
    else:
        fine_flat = np.bincount(
            placement.flat_index, weights=values.real, minlength=n_fine
        ) + 1j * np.bincount(placement.flat_index, weights=values.imag, minlength=n_fine)
    return fine_flat.reshape(placement.fine_shape)
