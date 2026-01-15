import numpy as np
from typing import Optional
from PAOFLOW.transportPAO.utils.timing import timed_function


@timed_function("hamiltonian_setup")
def hamiltonian_setup(
    ik: int,
    ie_g: int,
    egrid: np.ndarray,
    shift_L: float,
    shift_C: float,
    shift_R: float,
    blc_blocks: dict,
    shift_C_corr: float = 0.0,
    ie_buff: Optional[int] = None,
) -> None:
    """
    Compute the auxiliary matrices for Hamiltonian, overlap, and correlation blocks at given
    k-point and energy index.

    This function updates the `aux` attribute of each `OperatorBlock` with:
        aux = (E - shift) * S - H - Σ(E)

    Parameters
    ----------
    `ik` : int
        Index of the k-point.
    `ie_g` : int
        Index of the energy point.
    `egrid` : np.ndarray
        Array of energy values.
    `shift_L` : float
        Energy shift for the left lead.
    `shift_C` : float
        Energy shift for the central region.
    `shift_R` : float
        Energy shift for the right lead.
    `shift_C_corr` : float
        Additional correction shift applied to correlation blocks in the central region.
    `blc_blocks` : dict
        Dictionary of `OperatorBlock` instances keyed by block name (e.g. `'blc_00L'`).
    `ie_buff` : int, optional
        Buffered index corresponding to `ie_g` in case of correlation energy grids.

    Notes
    -----
    This function modifies the `aux` attribute in-place for each allocated block:
        `aux = (E - shift) * S[..., ik] - H[..., ik] - sgm[..., ik, ie_bl]`
    """
    omega = egrid[ie_g]
    ie_bl = ie_buff if ie_buff is not None else 1

    for name, block in blc_blocks.items():
        if not block.allocated:
            continue

        if "00L" in name or "01L" in name:
            shift = shift_L
        elif "00R" in name or "01R" in name:
            shift = shift_R
        else:
            shift = shift_C

        S_k = block.S[..., ik]
        H_k = block.H[..., ik]
        aux = (omega - shift) * S_k - H_k

        if block.sgm is not None:
            sgm_corr = block.sgm[..., ik, ie_bl]
            if "00C" in name or "LC" in name or "CR" in name:
                aux -= sgm_corr + shift_C_corr * S_k
            else:
                aux -= sgm_corr

        if "00" in name:
            block.aux[..., ik] = aux.conj()
        else:
            block.aux[..., ik] = aux

        block.update(ie=ie_g, ik=ik, ie_buff=ie_bl)
