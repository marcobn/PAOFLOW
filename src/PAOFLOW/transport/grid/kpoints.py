import numpy as np
from typing import Tuple, Optional, Union

from PAOFLOW.transport.utils.timing import timed_function


def kpoints_mask(
    vect: Union[Tuple[int, int], np.ndarray],
    init: Union[int, float],
    transport_direction: int,
) -> np.ndarray:
    """
    Embed a 2D vector into 3D space by inserting a value along the transport direction.

    Parameters
    ----------
    `vect` : Tuple[int, int] or (2,) ndarray
        A 2D vector to embed into 3D. Can represent:
        - Integer grid descriptors like mesh sizes or shift values (e.g., `nk`, `s`, `nr`)
        - Floating-point physical vectors like k-points or R-vectors in reciprocal space
    `init` : int or float
        Value to insert along the transport direction:
        - Use `1` for mesh dimensions
        - Use `0` for shift vectors or physical coordinates
    `transport_direction` : int
        Direction of transport: 1 = x, 2 = y, 3 = z.
        This direction will receive the `init` value, and `vect` will fill the two orthogonal directions.

    Returns
    -------
    `out` : (3,) ndarray
        A 3D vector with `init` inserted along the transport direction,
        and the 2D input `vect` placed in the remaining two directions.

    Notes
    -----
    This function is a general-purpose utility for constructing 3D vectors that respect
    the geometry of transport problems, where a 2D grid or vector lies perpendicular
    to the specified transport direction.

    It applies to both grid descriptors (like number of points or shifts) and to
    physical vectors (like k-points or R-vectors in fractional coordinates).

    Mapping logic:
        If `transport_direction == 1` (x-direction transport):
            out = [init, vect[0], vect[1]]
        If `transport_direction == 2` (y-direction transport):
            out = [vect[0], init, vect[1]]
        If `transport_direction == 3` (z-direction transport):
            out = [vect[0], vect[1], init]
    """
    vect = np.asarray(vect)
    if vect.shape != (2,):
        raise ValueError("`vect` must be a 2-element tuple or 1D array of shape (2,)")

    out = np.full(3, init, dtype=np.result_type(vect.dtype, type(init)))

    if transport_direction == 1:
        out[1:] = vect
    elif transport_direction == 2:
        out[0] = vect[0]
        out[2] = vect[1]
    elif transport_direction == 3:
        out[:2] = vect
    else:
        raise ValueError(f"Invalid transport direction: {transport_direction}")

    return out


def kpoints_equivalent(v1: np.ndarray, v2: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if two 2D k-points are equivalent under time-reversal symmetry (modulo 1).

    Parameters
    ----------
    `v1`, `v2` : (2,) ndarray
        2D k-point vectors in reciprocal lattice units (fractional coordinates).
    `tol` : float
        Tolerance for equality check.

    Returns
    -------
    `is_equiv` : bool
        True if v1 ≈ -v2 (mod 1), i.e., they are time-reversal partners.

    Notes
    -----
    In a time-reversal symmetric system, a k-point `k` is equivalent to `-k` (modulo the reciprocal lattice).
    This function checks:
        (v1 + v2) % 1 ≈ 0

    For example:
        v1 = (0.25, 0.5), v2 = (-0.25, -0.5) -> equivalent
        v1 = (0.25, 0.5), v2 = (0.25, 0.5)   -> not equivalent (unless v1 == 0)

    This is essential for symmetrizing the k-point mesh and avoiding double counting.
    """
    return np.allclose((v1 + v2) % 1.0, 0.0, atol=tol)


def initialize_meshsize(
    nr_full: np.ndarray,
    transport_direction: int,
    nk_par: Optional[np.ndarray] = None,
    use_safe_kmesh: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the 2D R-vector mesh (`nr_par`) orthogonal to the transport direction
    and validate or infer the corresponding k-point mesh (`nk_par`).

    Parameters
    ----------
    `nr_full` : (3,) ndarray of int
        Full 3D R-vector mesh sizes along (x, y, z) directions.
        For example: array([2, 2, 3])
    `transport_direction` : int
        Transport direction axis (1-based): 1 = x, 2 = y, 3 = z.
    `nk_par` : Optional[(2,) ndarray of int]
        User-specified number of k-points in the 2D plane orthogonal to transport.
        If None or [0, 0], it will be inferred from `nr_par`.
    `use_safe_kmesh` : bool
        If True, enforces `nk_par[i] >= nr_par[i]` to guarantee sufficient resolution
        for Fourier-based methods.

    Returns
    -------
    `nk_par` : (2,) ndarray of int
        Validated number of k-points in the orthogonal directions.
    `nr_par` : (2,) ndarray of int
        Number of R-vectors in directions orthogonal to transport.

    Notes
    -----
    Example:
        nr_full = np.array([2, 2, 3])
        transport_direction = 3

    → nr_par = array([2, 2])  # selects x and y directions
    → nk_par = array([2, 2]) if not provided
    """
    if not isinstance(nr_full, np.ndarray):
        raise TypeError("`nr_full` must be a NumPy array")
    if nr_full.shape != (3,):
        raise ValueError("`nr_full` must have shape (3,)")
    if transport_direction not in (1, 2, 3):
        raise ValueError(f"Invalid transport direction: {transport_direction}")

    axes = [0, 1, 2]
    axes.remove(transport_direction - 1)
    nr_par = nr_full[axes]

    if nk_par is None or np.all(np.asarray(nk_par) == 0):
        nk_par = nr_par.copy()
    else:
        nk_par = np.asarray(nk_par, dtype=int)
        if nk_par.shape != (2,):
            raise ValueError("`nk_par` must have shape (2,)")
        if np.any(nk_par < 1):
            raise ValueError(f"`nk_par` must be ≥ 1 in both directions, got {nk_par}")
        if use_safe_kmesh and np.any(nk_par < nr_par):
            raise ValueError(
                f"`nk_par` must be ≥ `nr_par` when use_safe_kmesh=True. "
                f"Got nk_par={nk_par}, nr_par={nr_par}"
            )

    return nk_par, nr_par


def initialize_kpoints(
    nk_par: np.ndarray,
    s_par: np.ndarray,
    transport_direction: int,
    use_sym: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D k-points and weights on a uniform 2D mesh orthogonal to the transport direction.

    Parameters
    ----------
    `nk_par` : (2,) ndarray of int
        Number of k-points along the two non-transport directions.
    `s_par` : (2,) ndarray of int
        Shifts (in fractional units) for the mesh in the two non-transport directions.
    `transport_direction` : int
        Transport direction (1 = x, 2 = y, 3 = z).
    `use_sym` : bool
        Whether to symmetrize the mesh under time-reversal (k ≡ -k).

    Returns
    -------
    `vkpt_par3D` : (nkpnts, 3) ndarray
        Generated 3D k-points in fractional coordinates.
    `wk_par` : (nkpnts,) ndarray
        Normalized weights for each unique k-point.

    Notes
    -----
    Example:
        If nk_par = [2, 2], s_par = [0, 0], transport_direction = 3

        kx(i) = (i - 1) / 2
        ky(j) = (j - 1) / 2
        → yields 2×2 mesh in xy-plane, mapped to 3D with z = 0

    Time-reversal symmetry is enforced as:
        k1 ≡ -k2 (mod 1) ⇒ count only one of them, with doubled weight.
    """
    nk_par = np.asarray(nk_par, dtype=int)
    s_par = np.asarray(s_par, dtype=int)

    if nk_par.shape != (2,) or s_par.shape != (2,):
        raise ValueError("`nk_par` and `s_par` must both be arrays of shape (2,)")

    mesh_x, mesh_y = nk_par
    shift_x, shift_y = s_par

    vkpts_2d = []
    weights = []

    for j in range(mesh_y):
        for i in range(mesh_x):
            kx = (i - mesh_x // 2) / mesh_x + shift_x / (2 * mesh_x)
            ky = (j - mesh_y // 2) / mesh_y + shift_y / (2 * mesh_y)
            kpt = np.array([kx, ky])
            if use_sym:
                for existing in vkpts_2d:
                    if kpoints_equivalent(existing, kpt):
                        break
                else:
                    vkpts_2d.append(kpt)
                    weights.append(1.0)
            else:
                vkpts_2d.append(kpt)
                weights.append(1.0)

    vkpts_2d = np.array(vkpts_2d)
    wk_par = np.array(weights, dtype=np.float64)
    wk_par /= wk_par.sum()

    vkpt_par3D = np.array(
        [kpoints_mask(kpt, 0.0, transport_direction) for kpt in vkpts_2d]
    )
    return vkpt_par3D, wk_par


@timed_function("cft_1z")
def compute_fourier_phase_table(
    vkpts: np.ndarray,
    ivr_par: np.ndarray,
) -> np.ndarray:
    """
    Compute Fourier phase factors exp(i 2π k · R) for each pair of k-point and R-vector.

    Parameters
    ----------
    `vkpts` : (nkpnts, 2) or (nkpnts, 3) ndarray
        k-point vectors in fractional reciprocal lattice units.
    `ivr_par` : (nR, 2) or (nR, 3) ndarray
        R-vectors in fractional crystal units (integer multiples of lattice vectors).

    Returns
    -------
    `table_par` : (nR, nkpnts) ndarray of complex128
        Phase factors e^{i 2π k · R} used for Fourier transforms or interpolation.

    Notes
    -----
    This computes the plane-wave phase factors:
        table_par[ir, ik] = exp(i * 2π * (k · R))
    where:
        k : reciprocal vector (fractional units)
        R : real-space vector (fractional units)

    These factors are essential for transforming quantities between real and reciprocal space.
    For example, they are used in:
        - Evaluating Fourier series expansions
        - Computing Green's functions or self-energies in k-space
        - Constructing Hamiltonians or overlaps in different representations

    The 2π factor comes from the convention of expressing k and R in fractional units:
        k = k_cartesian / (2π)  ->  k_cartesian = 2π * k_fractional
    """
    nR = ivr_par.shape[0]
    nkpnts = vkpts.shape[0]

    table = np.empty((nR, nkpnts), dtype=np.complex128)
    for ik in range(nkpnts):
        for ir in range(nR):
            arg = 2 * np.pi * np.dot(vkpts[ik], ivr_par[ir])
            table[ir, ik] = np.exp(1j * arg)
    return table


def initialize_r_vectors(
    nr_par: Tuple[int, int],
    transport_direction: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D R-vectors and weights based on a uniform 2D integer grid orthogonal to the transport direction.

    Parameters
    ----------
    `nr_par` : Tuple[int, int]
        2D mesh sizes for R-vectors in the directions orthogonal to transport.
    `transport_direction` : int
        Direction of transport (1-based, 1 = x, 2 = y, 3 = z).

    Returns
    -------
    `ivr_par3D` : (nR, 3) ndarray of int
        3D integer R-vectors in crystal coordinates.
    `wr_par` : (nR,) ndarray of float
        Weights for each R-vector (normalized to match Fortran behavior).

    Notes
    -----
    The 2D R-vectors are generated as:
        R_i = i - (nr_x + 1) // 2
        R_j = j - (nr_y + 1) // 2
    for i in [1, nr_x], j in [1, nr_y].

    Hermitian symmetry is enforced by ensuring that for each R, -R is present.
    If a corresponding -R is not found, it is added, and the weights of both R and -R are halved.

    The 2D vectors are then expanded to 3D using:
        if transport_direction == 1: (0, R1, R2)
        if transport_direction == 2: (R1, 0, R2)
        if transport_direction == 3: (R1, R2, 0)
    """
    nx, ny = nr_par
    R_list = []
    w_list = []

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            Rx = i - (nx + 1) // 2
            Ry = j - (ny + 1) // 2
            R_list.append([Rx, Ry])
            w_list.append(1.0)

    R_array = np.array(R_list, dtype=int)
    w_array = np.array(w_list, dtype=np.float64)

    counter = len(R_array)
    i = 0
    while i < counter:
        R = R_array[i]
        found = np.any(np.all(R_array[:counter] == -R, axis=1))
        if not found:
            R_array = np.vstack([R_array, -R])
            w_array = np.append(w_array, 0.5 * w_array[i])
            w_array[i] *= 0.5
            counter += 1
        i += 1

    def kpoints_imask(ivect: np.ndarray, transport_direction: int) -> np.ndarray:
        imask = np.zeros(3, dtype=int)
        if transport_direction == 1:
            imask[1:] = ivect
        elif transport_direction == 2:
            imask[0] = ivect[0]
            imask[2] = ivect[1]
        elif transport_direction == 3:
            imask[:2] = ivect
        else:
            raise ValueError(f"Invalid transport direction: {transport_direction}")
        return imask

    ivr_par3D = np.array([kpoints_imask(R, transport_direction) for R in R_array])
    wr_par = w_array

    return ivr_par3D, wr_par


def compute_ivr_par(nr_par: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D mesh of integer real-space vectors and associated weights
    for use in constructing operator blocks in reciprocal space.

    This function produces a grid of R-vectors on a uniform mesh centered
    around the origin, spanning the two directions orthogonal to the transport axis.
    Each R-vector is a pair of integers (Rx, Ry), symmetrically distributed
    about zero to preserve hermiticity in subsequent Fourier transforms.

    For each generated R-vector, its negative counterpart is also ensured to exist
    in the final set. If a negative is not initially present, it is added, and
    both the original and its negative are assigned half the original weight.

    Parameters
    ----------
    `nr_par` : Tuple[int, int]
        Size of the uniform mesh in the two directions orthogonal to the
        transport axis. Each component specifies the number of grid points
        along one axis.

    Returns
    -------
    `ivr_par` : (2, nR) ndarray of int
        Array of 2D integer real-space vectors.
    `wr_par` : (nR,) ndarray of float
        Associated weights for each R-vector, normalized by symmetry.

    Notes
    -----
    The R-vectors are centered about the origin using the rule:
        R_i = i - (N + 1) // 2
    where N is the number of grid points in the corresponding direction.
    This centers the mesh at zero for odd values and straddles zero for even values.
    """
    nx, ny = nr_par
    R_list = []
    w_list = []

    for j in range(1, ny + 1):
        for i in range(1, nx + 1):
            R1 = i - (nx + 1) // 2
            R2 = j - (ny + 1) // 2
            R_list.append((R1, R2))
            w_list.append(1.0)

    ivr = np.array(R_list, dtype=int)
    wr = np.array(w_list, dtype=np.float64)

    counter = len(ivr)
    i = 0
    while i < counter:
        R = ivr[i]
        found = np.any(np.all(ivr[:counter] == -R, axis=1))
        if not found:
            ivr = np.vstack([ivr, -R])
            wr = np.append(wr, 0.5 * wr[i])
            wr[i] *= 0.5
            counter += 1
        i += 1

    ivr_par = ivr.T  # shape (2, nR)
    return ivr_par, wr


class KpointsData:
    """
    Encapsulates all k-point and R-point related arrays and computes memory usage.
    """

    def __init__(self):
        self.vkpt_par: Optional[np.ndarray] = None
        self.vkpt_par3D: Optional[np.ndarray] = None
        self.wk_par: Optional[np.ndarray] = None

        self.ivr_par: Optional[np.ndarray] = None
        self.ivr_par3D: Optional[np.ndarray] = None
        self.vr_par3D: Optional[np.ndarray] = None
        self.wr_par: Optional[np.ndarray] = None

        self.table_par: Optional[np.ndarray] = None

    def memory_usage(self) -> float:
        """
        Compute total memory usage in MB based on array sizes and element types.
        """
        cost = 0.0

        if self.vkpt_par is not None:
            cost += self.vkpt_par.size * 8.0 / 1_000_000.0
        if self.vkpt_par3D is not None:
            cost += self.vkpt_par3D.size * 8.0 / 1_000_000.0
        if self.wk_par is not None:
            cost += self.wk_par.size * 8.0 / 1_000_000.0

        if self.ivr_par is not None:
            cost += self.ivr_par.size * 4.0 / 1_000_000.0
        if self.ivr_par3D is not None:
            cost += self.ivr_par3D.size * 4.0 / 1_000_000.0
        if self.vr_par3D is not None:
            cost += self.vr_par3D.size * 8.0 / 1_000_000.0
        if self.wr_par is not None:
            cost += self.wr_par.size * 8.0 / 1_000_000.0

        if self.table_par is not None:
            cost += self.table_par.size * 16.0 / 1_000_000.0

        return cost
