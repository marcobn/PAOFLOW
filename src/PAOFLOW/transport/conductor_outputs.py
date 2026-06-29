from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.data import ConductorData


def initialize_conductor_outputs(
    *,
    data: ConductorData,
    dimC: int,
    dimL: int,
    dimR: int,
    ne: int,
    nkpts_par: int,
    nrtot_par: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
]:
    """Allocate transport output arrays for the conductor workflow.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and runtime flags.
    dimC : int
        Conductor block dimension.
    dimL : int
        Left lead block dimension.
    dimR : int
        Right lead block dimension.
    ne : int
        Number of energy points.
    nkpts_par : int
        Number of k-points handled by this MPI rank.
    nrtot_par : int
        Number of real-space vectors handled by this MPI rank.

    Returns
    -------
    tuple
        ``(conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out)``.
        ``conduct`` has shape ``(1 + neigchn, ne)`` and dtype ``float64``.
        ``dos`` has shape ``(ne,)`` and dtype ``float64``.
        ``conduct_k`` has shape ``(1 + neigchn, nkpts_par, ne)`` and dtype ``float64``.
        ``dos_k`` has shape ``(ne, nkpts_par)`` and dtype ``float64``.
        ``gf_out``, ``rsgmL_out``, and ``rsgmR_out`` are ``complex128`` arrays
        of shape ``(ne, nrtot_par, dimC, dimC)`` when enabled by flags, otherwise ``None``.
    """
    do_eigenchannels = data.symmetry.do_eigenchannels
    neigchnx = data.symmetry.neigchnx
    neigchn = min(dimC, dimR, dimL, neigchnx) if do_eigenchannels else 0

    conduct = np.zeros((1 + neigchn, ne), dtype=np.float64)
    conduct_k = np.zeros((1 + neigchn, nkpts_par, ne), dtype=np.float64)
    dos = np.zeros(ne, dtype=np.float64)
    dos_k = np.zeros((ne, nkpts_par), dtype=np.float64)

    gf_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128)
        if data.symmetry.write_gf
        else None
    )
    rsgmL_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128)
        if data.symmetry.write_lead_sgm
        else None
    )
    rsgmR_out = (
        np.zeros((ne, nrtot_par, dimC, dimC), dtype=np.complex128)
        if data.symmetry.write_lead_sgm
        else None
    )

    return conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out


def initialize_kpoint_operator_buffers(
    *,
    data: ConductorData,
    nkpts_par: int,
    dimC: int,
) -> tuple[
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
]:
    """Allocate per-energy k-resolved operators.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and runtime flags.
    nkpts_par : int
        Number of local k-points.
    dimC : int
        Conductor block dimension.

    Returns
    -------
    tuple
        ``(gC_k, sgmL_k, sgmR_k)`` where each entry is a ``complex128`` array
        of shape ``(nkpts_par, dimC, dimC)`` when requested by output flags,
        otherwise ``None``.
    """
    gC_k = (
        np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128) if data.symmetry.write_gf else None
    )
    sgmL_k = (
        np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128)
        if data.symmetry.write_lead_sgm
        else None
    )
    sgmR_k = (
        np.zeros((nkpts_par, dimC, dimC), dtype=np.complex128)
        if data.symmetry.write_lead_sgm
        else None
    )
    return gC_k, sgmL_k, sgmR_k
