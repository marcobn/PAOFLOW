from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from mpi4py import MPI

from PAOFLOW.transport.conductor_energy_loop import (
    ConductorAccumulators,
    ParallelGridGeometry,
    process_energy_point,
    reduce_conductor_results,
)
from PAOFLOW.transport.conductor_writers import (
    write_conductor_operators,
    write_conductor_output,
)
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.io.write_data import write_data
from PAOFLOW.transport.io.write_data import write_operator_xml
from PAOFLOW.transport.results import TransportResults
from PAOFLOW.utils.constants import AMCONV, RYDCM1
from PAOFLOW.transport.utils.divide_et_impera import divide_work


def initialize_conductor_outputs(
    *,
    data: ConductorData,
    dimC: int,
    dimL: int,
    dimR: int,
    ne: int,
    nkpts_par: int,
    nrtot_par: int,
) -> ConductorAccumulators:
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
    ConductorAccumulators
        Bundle of zero-initialized accumulators. ``conduct`` has shape
        ``(1 + neigchn, ne)`` and dtype ``float64``; ``dos`` has shape ``(ne,)``;
        ``conduct_k`` has shape ``(1 + neigchn, nkpts_par, ne)``; ``dos_k`` has
        shape ``(ne, nkpts_par)``. ``gf_out``, ``rsgmL_out``, and ``rsgmR_out``
        are ``complex128`` arrays of shape ``(ne, nrtot_par, dimC, dimC)`` when
        enabled by flags, otherwise ``None``.
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

    return ConductorAccumulators(
        conduct=conduct,
        dos=dos,
        conduct_k=conduct_k,
        dos_k=dos_k,
        gf_out=gf_out,
        rsgmL_out=rsgmL_out,
        rsgmR_out=rsgmR_out,
    )


def compute_conductor_results(
    data: ConductorData,
    blc_blocks: Mapping[str, Any],
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> TransportResults:
    """Compute transport observables across the full energy and k-point grid.

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
    TransportResults
        Container with ``transmission``, ``dos``, ``transmission_k``,
        ``dos_k``, optional ``green_functions``, optional ``self_energy_L``,
        optional ``self_energy_R``, and ``energy_grid``.

    Notes
    -----
    This function:

    1. Distributes energies across MPI ranks.
    2. For each local energy, loops over local k-points.
    3. Computes :math:`G_C^r`, :math:`\\Sigma_L`, and :math:`\\Sigma_R`.
    4. Accumulates DOS and transmission observables.
    5. Optionally transforms k-resolved operators into real-space output.
    6. Reduces all arrays across MPI and returns ``TransportResults``.
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

    geom = ParallelGridGeometry(
        nkpts_par=nkpts_par,
        nrtot_par=nrtot_par,
        dimC=dimC,
        vkpt_par3D=vkpt_par3D,
        vr_par3D=vr_par3D,
        wk_par=wk_par,
    )
    acc = initialize_conductor_outputs(
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
            geom=geom,
            acc=acc,
        )

    reduce_conductor_results(comm=comm, data=data, acc=acc)

    return TransportResults(
        transmission=acc.conduct,
        dos=acc.dos,
        transmission_k=acc.conduct_k,
        dos_k=acc.dos_k,
        green_functions=acc.gf_out,
        self_energy_L=acc.rsgmL_out,
        self_energy_R=acc.rsgmR_out,
        energy_grid=egrid,
    )


def write_conductor_results(
    data: ConductorData,
    results: TransportResults,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Write transport observables and operators to files.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and output configuration.
    results : TransportResults
        Container with transport observables from ``compute_conductor_results``.
    comm : MPI.Comm, optional
        Communicator for rank-aware file writing.
        Default is ``MPI.COMM_WORLD``.

    Notes
    -----
    Writes operator arrays (Green's functions, lead self-energies) in real space
    if requested, and writes scalar/k-resolved transport observables.
    """
    rank = comm.Get_rank()
    runtime = data.get_runtime_data()
    ivr_par3D = runtime.ivr_par3D

    write_conductor_operators(
        rank=rank,
        data=data,
        gf_out=results.green_functions,
        rsgmL_out=results.self_energy_L,
        rsgmR_out=results.self_energy_R,
        ivr_par3D=ivr_par3D,
        egrid=results.energy_grid,
        dimC=data.dimC,
    )
    write_conductor_output(
        rank=rank,
        data=data,
        conduct=results.transmission,
        dos=results.dos,
        conduct_k=results.transmission_k,
        dos_k=results.dos_k,
        egrid=results.energy_grid,
    )


def write_self_energy_results(
    data: ConductorData,
    results: TransportResults,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Write lead self-energies to XML output files.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and output configuration.
    results : TransportResults
        Container with transport observables from ``compute_conductor_results``.
    comm : MPI.Comm, optional
        Communicator for rank-aware file writing.
        Default is ``MPI.COMM_WORLD``.

    Notes
    -----
    Writes ``lead_L_sgm.xml`` and ``lead_R_sgm.xml`` under the configured
    output directory. Non-root ranks perform no writes.
    """
    rank = comm.Get_rank()
    if rank != 0 or results.self_energy_L is None or results.self_energy_R is None:
        return
    ivr_par3D = data.get_runtime_data().ivr_par3D
    output_dir = Path(data.file_names.output_dir)
    write_operator_xml(
        output_dir=output_dir,
        filename='lead_L_sgm.xml',
        operator_matrix=results.self_energy_L,
        ivr=ivr_par3D,
        grid=results.energy_grid,
        dimwann=data.dimC,
        dynamical=True,
        eunits='eV',
        analyticity='retarded',
    )
    write_operator_xml(
        output_dir=output_dir,
        filename='lead_R_sgm.xml',
        operator_matrix=results.self_energy_R,
        ivr=ivr_par3D,
        grid=results.energy_grid,
        dimwann=data.dimC,
        dynamical=True,
        eunits='eV',
        analyticity='retarded',
    )


def write_greens_function_results(
    data: ConductorData,
    results: TransportResults,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Write conductor Green's functions to XML output files.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and output configuration.
    results : TransportResults
        Container with transport observables from ``compute_conductor_results``.
    comm : MPI.Comm, optional
        Communicator for rank-aware file writing.
        Default is ``MPI.COMM_WORLD``.

    Notes
    -----
    Writes ``greenf.xml`` under the configured output directory.
    Non-root ranks perform no writes.
    """
    rank = comm.Get_rank()
    if rank != 0 or results.green_functions is None:
        return
    ivr_par3D = data.get_runtime_data().ivr_par3D
    output_dir = Path(data.file_names.output_dir)
    write_operator_xml(
        output_dir=output_dir,
        filename='greenf.xml',
        operator_matrix=results.green_functions,
        ivr=ivr_par3D,
        grid=results.energy_grid,
        dimwann=data.dimC,
        dynamical=True,
        eunits='eV',
        analyticity='retarded',
    )


def write_transmission_results(
    data: ConductorData,
    results: TransportResults,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Write transmission and k-resolved transmission data files.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and output configuration.
    results : TransportResults
        Container with transport observables from ``compute_conductor_results``.
    comm : MPI.Comm, optional
        Communicator for rank-aware file writing.
        Default is ``MPI.COMM_WORLD``.

    Notes
    -----
    Writes ``conductance*.dat`` under the configured output directory.
    When ``write_kdata`` is enabled, also writes per-k-point conductance files.
    Non-root ranks perform no writes.
    """
    rank = comm.Get_rank()
    if rank != 0:
        return
    output_dir = Path(data.file_names.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = data.file_names.postfix
    if data.carriers == 'phonons':
        egrid_out = np.sqrt(results.energy_grid * RYDCM1**2 / AMCONV)
    else:
        egrid_out = results.energy_grid
    write_data(egrid_out, results.transmission, 'conductance', output_dir, postfix=postfix)
    if data.symmetry.write_kdata:
        nkpts_par = data.get_runtime_data().nkpts_par
        for ik in range(nkpts_par):
            ik_str = f'{ik + 1:04d}'
            filename_cond = f'cond{postfix}-{ik_str}.dat'
            with (output_dir / filename_cond).open('w') as f:
                for ie in range(results.energy_grid.shape[0]):
                    values = ' '.join(
                        f'{results.transmission_k[ch, ik, ie]:15.9f}'
                        for ch in range(results.transmission_k.shape[0])
                    )
                    f.write(f'{results.energy_grid[ie]:15.9f} {values}\n')


def write_dos_results(
    data: ConductorData,
    results: TransportResults,
    *,
    comm: MPI.Comm = MPI.COMM_WORLD,
) -> None:
    """Write DOS and k-resolved DOS data files.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and output configuration.
    results : TransportResults
        Container with transport observables from ``compute_conductor_results``.
    comm : MPI.Comm, optional
        Communicator for rank-aware file writing.
        Default is ``MPI.COMM_WORLD``.

    Notes
    -----
    Writes ``doscond*.dat`` under the configured output directory.
    When ``write_kdata`` is enabled, also writes per-k-point DOS files.
    Non-root ranks perform no writes.
    """
    rank = comm.Get_rank()
    if rank != 0:
        return
    output_dir = Path(data.file_names.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = data.file_names.postfix
    if data.carriers == 'phonons':
        egrid_out = np.sqrt(results.energy_grid * RYDCM1**2 / AMCONV)
    else:
        egrid_out = results.energy_grid
    write_data(egrid_out, results.dos, 'doscond', output_dir, postfix=postfix)
    if data.symmetry.write_kdata:
        nkpts_par = data.get_runtime_data().nkpts_par
        for ik in range(nkpts_par):
            ik_str = f'{ik + 1:04d}'
            filename_dos = f'doscond{postfix}-{ik_str}.dat'
            with (output_dir / filename_dos).open('w') as f:
                for ie in range(results.energy_grid.shape[0]):
                    f.write(f'{results.energy_grid[ie]:15.9f} {results.dos_k[ie, ik]:15.9f}\n')
