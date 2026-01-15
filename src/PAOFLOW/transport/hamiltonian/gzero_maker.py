import numpy as np
from typing import Optional, Literal
from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlockView
from PAOFLOW.transport.utils.timing import timed_function


@timed_function("gzero_maker")
def compute_non_interacting_gf(
    blc_00C: OperatorBlockView,
    smearing_type: str = "lorentzian",
    delta: float = 1e-5,
    delta_ratio: float = 5e-3,
    g_smear: Optional[np.ndarray] = None,
    xgrid: Optional[np.ndarray] = None,
    calc: Literal["direct", "inverse"] = "inverse",
) -> np.ndarray:
    """
    Compute the non-interacting Green's function or its inverse using smearing.

    Parameters
    ----------
    `energy` : float
        Real energy value at which to evaluate the Green's function.
    `smearing_type` : str
        Smearing method: 'lorentzian', 'none', or 'numerical'.
    `delta` : float
        Smearing parameter for imaginary broadening.
    `delta_ratio` : float
        Used for 'none' smearing: `delta_eff = delta * delta_ratio`.
    `g_smear` : np.ndarray, optional
        Precomputed smeared Green’s function values on `xgrid` (for numerical smearing).
    `xgrid` : np.ndarray, optional
        Energy grid corresponding to `g_smear`.
    `calc` : {'direct', 'inverse'}
        Whether to return `G` or `G⁻¹`.

    Returns
    -------
    `gzero` : np.ndarray
        The resulting Green's function or its inverse.

    Notes
    -----
    The function evaluates one of the following:

    - For `'lorentzian'`:
        `G = [ (E + i·δ)·S - H ]⁻¹`
    - For `'none'`:
        `G = [ (E + i·δ·δ_ratio)·S - H ]⁻¹`
    - For `'numerical'`:
        Diagonalize `A = E·S - H`, interpolate on `xgrid`, and reconstruct `G`.

    Smearing ensures convergence and causality in retarded Green's functions.
    """
    if smearing_type in ("lorentzian", "none"):
        delta_eff = delta if smearing_type == "lorentzian" else delta * delta_ratio
        A = blc_00C.aux + 1j * delta_eff * blc_00C.S
        gzero = np.linalg.inv(A) if calc == "direct" else A

    elif smearing_type == "numerical":
        A = blc_00C.aux

        if g_smear is None or xgrid is None:
            raise ValueError("Numerical smearing requires `g_smear` and `xgrid`.")

        if not np.allclose(A, A.conj().T, atol=1e-8):
            raise ValueError(
                "Matrix A is not Hermitian; numerical smearing requires Hermitian matrix."
            )

        eigvals, eigvecs = np.linalg.eigh(A)
        scaled = eigvals / delta

        interpolated = np.empty_like(scaled, dtype=np.complex128)
        for i, x in enumerate(scaled):
            if x < xgrid[0] or x > xgrid[-1]:
                interpolated[i] = (
                    1 / (delta * (x + 1j)) if calc == "direct" else x * delta
                )
            else:
                idx = np.searchsorted(xgrid, x) - 1
                dx = xgrid[idx + 1] - xgrid[idx]
                alpha = (x - xgrid[idx]) / dx if dx != 0 else 0.0
                f1, f2 = g_smear[idx], g_smear[idx + 1]
                if calc == "direct":
                    interpolated[i] = (1 - alpha) * f1 + alpha * f2
                else:
                    f1_inv = 1.0 / f1
                    f2_inv = 1.0 / f2
                    interpolated[i] = (1 - alpha) * f1_inv + alpha * f2_inv

        gzero = eigvecs @ np.diag(interpolated) @ eigvecs.conj().T

    else:
        raise ValueError(f"Unsupported smearing_type: {smearing_type}")

    return gzero
