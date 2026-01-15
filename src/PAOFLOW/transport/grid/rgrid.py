from typing import Tuple

import numpy as np


def get_rgrid(
    mesh_dims: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a real-space R-grid in crystal coordinates and associated weights.

    Parameters
    ----------
    mesh_dims : tuple of int
        Number of divisions along each reciprocal lattice direction (``nr1, nr2, nr3``).

    Returns
    -------
    r_points : ndarray of shape (n_points, 3)
        Integer triplets defining real-space grid vectors in lattice units.
    weights : ndarray of shape (n_points,)
        Weights associated with each R-vector.

    Notes
    -----
    The function generates a regular grid of R-vectors in reciprocal space defined by the mesh dimensions
    ``nr = (nr1, nr2, nr3)``. Each grid point corresponds to a triplet of crystal coordinates:

    R_ijk = (i - (nr1 + 1)/2,
            j - (nr2 + 1)/2,
            k - (nr3 + 1)/2)

    For each R-vector, a weight ``wr`` is assigned. If the point ``-R`` is not already included in the grid, it is added explicitly and the weights of ``R`` and ``-R`` are halved to ensure time-reversal symmetry.

    The returned weights satisfy the normalization condition:

    sum_i wr[i] = nr1 * nr2 * nr3

    This ensures the sum rule is compatible with Fourier transforms over the Brillouin zone.
    """
    nr1, nr2, nr3 = mesh_dims
    if any(n <= 0 for n in (nr1, nr2, nr3)):
        raise ValueError("Mesh dimensions must be strictly positive.")

    r_grid = []
    weights = []

    for k in range(nr3):
        for j in range(nr2):
            for i in range(nr1):
                i1, j1, k1 = i + 1, j + 1, k + 1

                r = (
                    i1 - (nr1 + 1) // 2,
                    j1 - (nr2 + 1) // 2,
                    k1 - (nr3 + 1) // 2,
                )
                r_grid.append(r)
                weights.append(1.0)

    r_grid = np.array(r_grid, dtype=int)
    weights = np.array(weights, dtype=np.float64)

    # Ensure -R counterpart exists
    r_dict = {tuple(r): idx for idx, r in enumerate(r_grid)}
    extra_r = []

    for idx, r in enumerate(r_grid):
        minus_r = tuple(-r)
        if minus_r not in r_dict:
            extra_r.append((-r, idx))

    for new_r, idx in extra_r:
        r_grid = np.vstack([r_grid, new_r])
        weights = np.append(weights, 0.5 * weights[idx])
        weights[idx] *= 0.5

    expected_sum = nr1 * nr2 * nr3
    actual_sum = np.sum(weights)
    if not np.isclose(actual_sum, expected_sum, atol=1e-10):
        raise ValueError(
            f"Invalid weight sum rule: got {actual_sum}, expected {expected_sum}"
        )

    return r_grid, weights
