import numpy as np


def cartesian_to_crystal(coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from Cartesian to crystal (fractional) coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Cartesian coordinates, shape (3,) or (3, N)
    basis : np.ndarray
        Lattice basis vectors (columns), shape (3, 3)

    Returns
    -------
    np.ndarray
        Fractional crystal coordinates, same shape as input

    Notes
    -----
    The transformation is given by:

        k_crys = inv(basis) @ k_cart

    where `basis` contains the lattice vectors as columns.

    If `coords` is a single 3-vector (shape (3,)), it is automatically
    promoted to shape (3, 1), and the returned result will have shape (3, 1).
    """
    coords = np.atleast_2d(coords)
    if coords.shape[0] != 3:
        raise ValueError("Input `coords` must have shape (3,) or (3, N)")

    try:
        basis_inv = np.linalg.inv(basis)
    except np.linalg.LinAlgError:
        raise ValueError("Basis matrix is singular or not invertible")

    return basis_inv @ coords


def crystal_to_cartesian(coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Convert coordinates from crystal (fractional) to Cartesian coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Fractional crystal coordinates, shape (3,) or (3, N)
    basis : np.ndarray
        Lattice basis vectors (columns), shape (3, 3)

    Returns
    -------
    np.ndarray
        Cartesian coordinates, same shape as input

    Notes
    -----
    The transformation is given by:

        r_cart = basis @ r_crys

    where `basis` contains the lattice vectors as columns.

    If `coords` is a single 3-vector (shape (3,)), it is automatically
    promoted to shape (3, 1), and the returned result will have shape (3, 1).
    """
    coords = np.atleast_2d(coords)
    if coords.shape[0] != 3:
        raise ValueError("Input `coords` must have shape (3,) or (3, N)")

    return basis @ coords
