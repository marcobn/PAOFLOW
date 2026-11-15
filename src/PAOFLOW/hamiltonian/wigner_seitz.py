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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SUPPORTED_INTERPOLATIONS = ('zero_padding', 'wigner_seitz')
SUPPORTED_BAND_INTERPOLATIONS = ('fft_grid', 'wigner_seitz')


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
    hamiltonian_k = np.empty((k_points.shape[0], nawf * nawf * nspin), dtype=complex)
    for start in range(0, k_points.shape[0], block_size):
        stop = min(start + block_size, k_points.shape[0])
        phases = np.exp(-2j * np.pi * (k_points[start:stop] @ images.cells.T))
        hamiltonian_k[start:stop] = phases @ weighted_hoppings
    hamiltonian_k = np.moveaxis(hamiltonian_k.reshape(-1, nawf, nawf, nspin), 0, 2)
    if np.isrealobj(k_points) or not np.iscomplex(k_points).any():
        hamiltonian_k = 0.5 * (hamiltonian_k + np.conj(np.swapaxes(hamiltonian_k, 0, 1)))
    return np.ascontiguousarray(hamiltonian_k)


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
