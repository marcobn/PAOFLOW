from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

from PAOFLOW.transport.conductor_energy_loop import (
    process_energy_point,
    reduce_conductor_results,
)
from PAOFLOW.transport.conductor_outputs import initialize_conductor_outputs
from PAOFLOW.transport.conductor_writers import (
    write_conductor_operators,
    write_conductor_output,
)
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.utils.divide_et_impera import divide_work


def run_conductor(
    data: ConductorData,
    blc_blocks: Mapping[str, Any],
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
    NDArray[np.complex128] | None,
    NDArray[np.float64],
]:
    """Execute the procedural conductor transport workflow.

    Parameters
    ----------
    data : ConductorData
        Validated transport input, runtime metadata, and output flags.
    blc_blocks : Mapping[str, Any]
        Block-operator dictionary used to build self-energies and
        conductor Green's functions at each ``(E, k)`` point.
    comm : MPI.Comm, optional
        Communicator used for work distribution and reductions.
        Default is ``MPI.COMM_WORLD``.

    Returns
    -------
    tuple
        ``(conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid)``.
        ``conduct`` has shape ``(1 + neigchn, ne)`` and ``dos`` has shape ``(ne,)``.
        ``conduct_k`` has shape ``(1 + neigchn, nkpts_par, ne)`` and
        ``dos_k`` has shape ``(ne, nkpts_par)``.
        ``gf_out``, ``rsgmL_out``, and ``rsgmR_out`` are optional
        ``complex128`` arrays with shape ``(ne, nrtot_par, dimC, dimC)``.
        ``egrid`` is the energy grid with shape ``(ne,)``.

    Notes
    -----
    This routine preserves the legacy conductor sequence:

    1. Distribute energies across MPI ranks.
    2. For each local energy, loop over local k-points.
    3. Compute :math:`G_C^r`, :math:`\\Sigma_L`, and :math:`\\Sigma_R`.
    4. Accumulate DOS and transmission observables.
    5. Optionally transform k-resolved operators into real-space output.
    6. Reduce all arrays across MPI and write final outputs.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    runtime = data.get_runtime_data()

    dimC = data.dimC
    dimR = data.dimR
    dimL = data.dimL
    ne = data.energy.ne
    delta = data.energy.delta
    nkpts_par = int(runtime.nkpts_par)
    wk_par = data._runtime.wk_par
    ivr_par3D = runtime.ivr_par3D
    vr_par3D = 2 * np.pi * ivr_par3D.astype(np.float64)
    nrtot_par = int(runtime.nrtot_par)
    vkpt_par3D = data._runtime.vkpt_par3D
    egrid = initialize_energy_grid(
        emin=data.energy.emin,
        emax=data.energy.emax,
        ne=ne,
        carriers=data.carriers,
    )

    conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out = initialize_conductor_outputs(
        data=data,
        dimC=dimC,
        dimL=dimL,
        dimR=dimR,
        ne=ne,
        nkpts_par=nkpts_par,
        nrtot_par=nrtot_par,
    )
    ie_start, ie_end = divide_work(0, ne - 1, rank, size, 'energies')

    for ie_g in range(ie_start, ie_end + 1):
        process_energy_point(
            data=data,
            blc_blocks=blc_blocks,
            egrid=egrid,
            delta=delta,
            ie_g=ie_g,
            ie_start=ie_start,
            ie_end=ie_end,
            rank=rank,
            nkpts_par=nkpts_par,
            dimC=dimC,
            nrtot_par=nrtot_par,
            vkpt_par3D=vkpt_par3D,
            vr_par3D=vr_par3D,
            wk_par=wk_par,
            conduct=conduct,
            dos=dos,
            conduct_k=conduct_k,
            dos_k=dos_k,
            gf_out=gf_out,
            rsgmL_out=rsgmL_out,
            rsgmR_out=rsgmR_out,
        )

    reduce_conductor_results(
        comm=comm,
        data=data,
        conduct=conduct,
        dos=dos,
        conduct_k=conduct_k,
        dos_k=dos_k,
        gf_out=gf_out,
        rsgmL_out=rsgmL_out,
        rsgmR_out=rsgmR_out,
    )
    write_conductor_operators(
        rank=rank,
        data=data,
        gf_out=gf_out,
        rsgmL_out=rsgmL_out,
        rsgmR_out=rsgmR_out,
        ivr_par3D=ivr_par3D,
        egrid=egrid,
        dimC=dimC,
    )
    write_conductor_output(
        rank=rank,
        data=data,
        conduct=conduct,
        dos=dos,
        conduct_k=conduct_k,
        dos_k=dos_k,
        egrid=egrid,
    )

    return conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid
