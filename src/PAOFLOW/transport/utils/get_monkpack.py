import numpy as np
from typing import Optional, Tuple, Literal

from PAOFLOW.transport.utils.converters import cartesian_to_crystal


def get_monkhorst_pack_grid(
    kpts: np.ndarray,
    bvec: Optional[np.ndarray] = None,
    coordinate: Literal["crystal", "cartesian"] = "crystal",
    tol: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Infer Monkhorst-Pack grid dimensions and shifts from a set of k-points.

    Parameters
    ----------
    `kpts` : (3, nkpnts) ndarray
        K-points in reciprocal space. Can be in either crystal or cartesian coordinates.
    `bvec` : (3, 3) ndarray
        Reciprocal lattice vectors in columns (bohr^-1).
    `coordinate` : {"crystal", "cartesian"}
        Indicates the coordinate system of the input `kpts`.
    `tol` : float
        Tolerance for matching floating point values.

    Returns
    -------
    `nk` : (3,) ndarray
        Grid dimensions in each direction.
    `shift` : (3,) ndarray
        Shift vector: 0 for Γ-centered, 1 for shifted grid.
    `is_mp` : bool
        Whether the input matches a Monkhorst-Pack grid.

    Notes
    -----
    The 1D Monkhorst-Pack grid is defined as:

        k_i = (2n_i + s_i) / (2N_i), for n_i in 0..N_i-1 and s_i ∈ {0, 1}

    If `coordinate="cartesian"`, the k-points are converted to crystal coordinates using:

        k_crystal = bvec⁻¹ · k_cartesian

    All subsequent matching and comparison are done in crystal space.
    """
    if coordinate.lower() == "cartesian":
        kpts = cartesian_to_crystal(kpts, bvec)

    nkpnts = kpts.shape[1]
    log.log_rank0(f"nkpnts: {nkpnts}")
    nk = np.zeros(3, dtype=int)
    shift = np.ones(3, dtype=int)

    kpt_loc = np.mod(kpts.copy(), 1.0)
    kpt_loc[np.abs(kpt_loc) < tol] = 0.0

    for i in range(3):
        if np.any(np.abs(kpt_loc[i]) < tol):
            shift[i] = 0
            kpt_loc[i][np.abs(kpt_loc[i]) < tol] = 0.0

    kpt_gen = kpt_loc.copy()
    for i in range(3):
        if shift[i] == 1:
            kpt_gen[i] -= np.min(kpt_gen[i])

        with np.errstate(divide="ignore", invalid="ignore"):
            spacing_inv = np.where(kpt_gen[i] > tol, 1.0 / kpt_gen[i], 0.0)
        nk[i] = int(np.max(np.round(spacing_inv)))

    def generate_mp_grid(nk, shift):
        grid = []
        for i in range(nk[0]):
            for j in range(nk[1]):
                for k in range(nk[2]):
                    ki = (2 * i + shift[0]) / (2 * nk[0])
                    kj = (2 * j + shift[1]) / (2 * nk[1])
                    kk = (2 * k + shift[2]) / (2 * nk[2])
                    grid.append((ki % 1, kj % 1, kk % 1))
        return np.array(grid).T

    kpt_mp = generate_mp_grid(nk, shift)
    log.log_rank0(kpt_mp)
    matched = 0
    for col in kpt_loc.T:
        if np.any(np.all(np.abs(kpt_mp.T - col) < tol, axis=1)):
            matched += 1

    is_mp = matched == nkpnts
    if not is_mp:
        nk[:] = 0
        shift[:] = 0

    return nk, shift, is_mp
