from typing import Literal, Optional

import numpy as np
from PAOFLOW.transportPAO.hamiltonian.gzero_maker import compute_non_interacting_gf
from PAOFLOW.transportPAO.hamiltonian.operator_blc import OperatorBlockView
from PAOFLOW.transportPAO.utils.timing import timed_function


@timed_function("green")
def compute_surface_green_function(
    h_eff: np.ndarray,
    s_eff: np.ndarray,
    t_coupling: np.ndarray,
    transfer_matrix: np.ndarray,
    transfer_matrix_conj: np.ndarray,
    igreen: Literal[-1, 0, 1],
    delta: float = 1e-5,
) -> np.ndarray:
    """
    Construct the surface or bulk Green's function using the transfer matrices.

    Parameters
    ----------
    `h_eff` : np.ndarray
        Hamiltonian block `H_00` of shape (dim, dim).
    `s_eff` : np.ndarray
        Overlap matrix `S_00` of shape (dim, dim).
    `t_coupling` : np.ndarray
        Coupling matrix `H_01` of shape (dim, dim).
    `transfer_matrix` : np.ndarray
        Right-going transfer matrix `T` of shape (dim, dim).
    `transfer_matrix_conj` : np.ndarray
        Left-going transfer matrix `T†` of shape (dim, dim).
    `igreen` : {-1, 0, 1}
        Green’s function type:
        -1: left surface,
         0: bulk,
         1: right surface.
    `delta` : float
        Small imaginary shift to stabilize inversion.

    Returns
    -------
    `green` : np.ndarray
        The computed Green’s function matrix `G(E)` of shape (dim, dim).

    Notes
    -----
    Implements the iterative method of Lopez Sancho et al. (J. Phys. F: Met. Phys., 14, 1205, 1984)
    to compute the transfer matrices `T` and `T†` for semi-infinite leads.

    The Green’s function is stabilized using a small imaginary part `delta`:
    `G = [E⋅S - H + i⋅delta⋅S - Σ]⁻¹`
    `T` and `T†` are constructed iteratively to capture surface coupling.

    """
    z_shift = 1j * delta * s_eff
    A = h_eff + z_shift

    if igreen == 1:
        A -= t_coupling @ transfer_matrix
    elif igreen == -1:
        A -= t_coupling.conj().T @ transfer_matrix_conj
    elif igreen == 0:
        A -= t_coupling @ transfer_matrix
        A -= t_coupling.conj().T @ transfer_matrix_conj
    else:
        raise ValueError(f"Invalid value for `igreen`: {igreen}. Must be -1, 0, or 1.")

    try:
        g = np.linalg.inv(A)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            "Green's function inversion failed due to singular matrix."
        ) from e

    return g


def compute_conductor_green_function(
    blc_00C: OperatorBlockView,
    sigma_l: np.ndarray,
    sigma_r: Optional[np.ndarray] = None,
    smearing_type: str = "lorentzian",
    delta: float = 1e-5,
    delta_ratio: float = 5e-3,
    g_smear: Optional[np.ndarray] = None,
    xgrid: Optional[np.ndarray] = None,
    calc: Literal["inverse", "direct"] = "inverse",
    surface: bool = False,
) -> np.ndarray:
    """
    Construct the retarded Green's function for the conductor region.

    Parameters
    ----------
    `sigma_l` : np.ndarray
        Self-energy from the left lead.
    `sigma_r` : np.ndarray, optional
        Self-energy from the right lead. Not used if `surface=True`.
    `smearing_type` : str
        Smearing method: 'lorentzian', 'none', or 'numerical'.
    `delta` : float
        Imaginary broadening parameter.
    `delta_ratio` : float
        Delta scaling used for 'none' smearing.
    `g_smear` : np.ndarray, optional
        Precomputed smeared Green’s function values on `xgrid` (for numerical smearing).
    `xgrid` : np.ndarray, optional
        Energy grid corresponding to `g_smear`.
    `calc` : {'direct', 'inverse'}
        Whether to compute G or G⁻¹ for subtraction.
    `surface` : bool
        Whether to compute projected surface bandstructure using only left lead.

    Returns
    -------
    `g_c` : np.ndarray
        Green's function of the conductor region.

    Notes
    -----
    Computes:

        G_C = [ ω·S_C − H_C − Σ_L − Σ_R ]⁻¹

    or for surface projection:

        G_C = [ ω·S_C − H_C − Σ_L ]⁻¹
    """
    g0inv = compute_non_interacting_gf(
        blc_00C=blc_00C,
        smearing_type=smearing_type,
        delta=delta,
        delta_ratio=delta_ratio,
        g_smear=g_smear,
        xgrid=xgrid,
        calc=calc,
    )

    if surface:
        g0inv -= sigma_l
    else:
        if sigma_r is None:
            raise ValueError("Right self-energy must be provided unless surface=True.")
        g0inv -= sigma_l + sigma_r

    return np.linalg.inv(g0inv)
