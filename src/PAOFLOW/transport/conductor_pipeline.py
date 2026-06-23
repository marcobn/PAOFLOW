from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.transport.conductor_kpoint import compute_kpoint_conductor_quantities
from PAOFLOW.transport.conductor_observables import (
    accumulate_dos,
    accumulate_transmission,
)
from PAOFLOW.transport.grid.egrid import initialize_energy_grid
from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham
from PAOFLOW.transport.io.input_parameters import ConductorData
from PAOFLOW.transport.io.write_data import (
    write_data,
    write_operator_xml,
)
from PAOFLOW.transport.io.write_header import headered_function
from PAOFLOW.transport.utils.constants import amconv, rydcm1
from PAOFLOW.transport.utils.divide_et_impera import divide_work
from PAOFLOW.transport.utils.timing import global_timing


def _initialize_outputs(
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


def _initialize_k_dependent_operators(
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


def _transform_k_to_r_at_energy(
    *,
    data: ConductorData,
    ie_g: int,
    gC_k: NDArray[np.complex128] | None,
    sgmL_k: NDArray[np.complex128] | None,
    sgmR_k: NDArray[np.complex128] | None,
    gf_out: NDArray[np.complex128] | None,
    rsgmL_out: NDArray[np.complex128] | None,
    rsgmR_out: NDArray[np.complex128] | None,
    nrtot_par: int,
    vr_par3D: NDArray[np.float64],
    vkpt_par3D: NDArray[np.float64],
    wk_par: NDArray[np.float64],
) -> None:
    """Transform local k-space operators to real-space operators at one energy.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and runtime flags.
    ie_g : int
        Global energy index.
    gC_k : NDArray[np.complex128] or None
        k-resolved conductor Green's function, shape ``(nkpts_par, dimC, dimC)``.
    sgmL_k : NDArray[np.complex128] or None
        k-resolved left self-energy, shape ``(nkpts_par, dimC, dimC)``.
    sgmR_k : NDArray[np.complex128] or None
        k-resolved right self-energy, shape ``(nkpts_par, dimC, dimC)``.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function accumulator, shape ``(ne, nrtot_par, dimC, dimC)``.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy accumulator, shape ``(ne, nrtot_par, dimC, dimC)``.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy accumulator, shape ``(ne, nrtot_par, dimC, dimC)``.
    nrtot_par : int
        Number of local real-space vectors.
    vr_par3D : NDArray[np.float64]
        Local real-space vectors, shape ``(nrtot_par, 3)``.
    vkpt_par3D : NDArray[np.float64]
        Local k-point coordinates, shape ``(3, nkpts_par)``.
    wk_par : NDArray[np.float64]
        Local k-point weights, shape ``(nkpts_par,)``.

    Returns
    -------
    None
        Updates ``gf_out``, ``rsgmL_out``, and ``rsgmR_out`` in place for ``ie_g``
        when the corresponding output flags are enabled.

    Notes
    -----
    The k-to-R transform is computed with ``compute_rham`` using the same
    k-point grid and weights as the original class-based implementation.
    """
    if data.symmetry.write_gf and gC_k is not None and gf_out is not None:
        for ir in range(nrtot_par):
            gf_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], gC_k, vkpt_par3D.T, wk_par)
    if data.symmetry.write_lead_sgm and sgmL_k is not None and sgmR_k is not None:
        if rsgmL_out is None or rsgmR_out is None:
            return
        for ir in range(nrtot_par):
            rsgmL_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], sgmL_k, vkpt_par3D.T, wk_par)
            rsgmR_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], sgmR_k, vkpt_par3D.T, wk_par)


def _reduce_results(
    *,
    comm: MPI.Comm,
    data: ConductorData,
    conduct: NDArray[np.float64],
    dos: NDArray[np.float64],
    conduct_k: NDArray[np.float64],
    dos_k: NDArray[np.float64],
    gf_out: NDArray[np.complex128] | None,
    rsgmL_out: NDArray[np.complex128] | None,
    rsgmR_out: NDArray[np.complex128] | None,
) -> None:
    """Reduce distributed transport results across MPI ranks.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator used for collective reductions.
    data : ConductorData
        Validated transport input and runtime flags.
    conduct : NDArray[np.float64]
        Conductance accumulator, shape ``(1 + neigchn, ne)``.
    dos : NDArray[np.float64]
        DOS accumulator, shape ``(ne,)``.
    conduct_k : NDArray[np.float64]
        k-resolved conductance accumulator, shape ``(1 + neigchn, nkpts_par, ne)``.
    dos_k : NDArray[np.float64]
        k-resolved DOS accumulator, shape ``(ne, nkpts_par)``.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function accumulator.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy accumulator.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy accumulator.

    Returns
    -------
    None
        Applies in-place ``MPI.Allreduce`` with ``MPI.SUM`` to all enabled arrays.
    """
    comm.Allreduce(MPI.IN_PLACE, conduct, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, conduct_k, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, dos_k, op=MPI.SUM)

    if data.symmetry.write_gf and gf_out is not None:
        comm.Allreduce(MPI.IN_PLACE, gf_out, op=MPI.SUM)
    if data.symmetry.write_lead_sgm and rsgmL_out is not None and rsgmR_out is not None:
        comm.Allreduce(MPI.IN_PLACE, rsgmL_out, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, rsgmR_out, op=MPI.SUM)


def _write_operators(
    *,
    rank: int,
    data: ConductorData,
    gf_out: NDArray[np.complex128] | None,
    rsgmL_out: NDArray[np.complex128] | None,
    rsgmR_out: NDArray[np.complex128] | None,
    ivr_par3D: NDArray[np.int64],
    egrid: NDArray[np.float64],
    dimC: int,
) -> None:
    """Write real-space operators to XML files on rank 0.

    Parameters
    ----------
    rank : int
        MPI rank.
    data : ConductorData
        Validated transport input and runtime flags.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function output array.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy output array.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy output array.
    ivr_par3D : NDArray[np.int64]
        Integer real-space vectors associated with operator rows.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    dimC : int
        Conductor block dimension.

    Returns
    -------
    None
        Writes ``greenf.xml``, ``lead_L_sgm.xml``, and ``lead_R_sgm.xml``
        depending on enabled flags. Non-root ranks perform no writes.
    """
    if rank != 0:
        return

    if data.symmetry.write_gf and gf_out is not None:
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='greenf.xml',
            operator_matrix=gf_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )
    if data.symmetry.write_lead_sgm and rsgmL_out is not None and rsgmR_out is not None:
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='lead_L_sgm.xml',
            operator_matrix=rsgmL_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )
        write_operator_xml(
            output_dir=Path(data.file_names.output_dir),
            filename='lead_R_sgm.xml',
            operator_matrix=rsgmR_out,
            ivr=ivr_par3D,
            grid=egrid,
            dimwann=dimC,
            dynamical=True,
            eunits='eV',
            analyticity='retarded',
        )


@headered_function('Writing data')
def _write_output(
    *,
    rank: int,
    data: ConductorData,
    conduct: NDArray[np.float64],
    dos: NDArray[np.float64],
    conduct_k: NDArray[np.float64],
    dos_k: NDArray[np.float64],
    egrid: NDArray[np.float64],
) -> None:
    """Write conductance and DOS data products for the conductor workflow.

    Parameters
    ----------
    rank : int
        MPI rank.
    data : ConductorData
        Validated transport input and runtime flags.
    conduct : NDArray[np.float64]
        Total conductance and optional eigenchannels, shape ``(1 + neigchn, ne)``.
    dos : NDArray[np.float64]
        Total DOS, shape ``(ne,)``.
    conduct_k : NDArray[np.float64]
        k-resolved conductance, shape ``(1 + neigchn, nkpts_par, ne)``.
    dos_k : NDArray[np.float64]
        k-resolved DOS, shape ``(ne, nkpts_par)``.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.

    Returns
    -------
    None
        Writes ``conductance*.dat`` and ``doscond*.dat`` under
        ``data.file_names.output_dir``. When ``write_kdata`` is enabled,
        also writes per-kpoint files. Non-root ranks perform no writes.
    """
    if rank != 0:
        return

    output_dir = Path(data.file_names.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    postfix = data.file_names.postfix

    if data.carriers == 'phonons':
        egrid_out = np.sqrt(egrid * rydcm1**2 / amconv)
    else:
        egrid_out = egrid

    write_data(egrid_out, conduct, 'conductance', output_dir, postfix=postfix)
    write_data(egrid_out, dos, 'doscond', output_dir, postfix=postfix)

    if data.symmetry.write_kdata:
        nkpts_par = data.get_runtime_data().nkpts_par
        prefix = os.path.basename(data.file_names.datafile_C)

        for ik in range(nkpts_par):
            ik_str = f'{ik + 1:04d}'
            filename_cond = f'{prefix}_cond-{ik_str}.dat'
            filename_dos = f'{prefix}_doscond-{ik_str}.dat'

            with (output_dir / filename_cond).open('w') as f:
                for ie in range(egrid.shape[0]):
                    values = ' '.join(
                        f'{conduct_k[ch, ik, ie]:15.9f}' for ch in range(conduct_k.shape[0])
                    )
                    f.write(f'{egrid[ie]:15.9f} {values}\n')

            with (output_dir / filename_dos).open('w') as f:
                for ie in range(egrid.shape[0]):
                    f.write(f'{egrid[ie]:15.9f} {dos_k[ie, ik]:15.9f}\n')


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

    conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out = _initialize_outputs(
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
        nprint = data.iteration.nprint
        if (ie_g % nprint == 0 or ie_g == 0 or ie_g == ne - 1) and rank == 0:
            if data.carriers == 'phonons':
                omega_val = np.sqrt(egrid[ie_g] * rydcm1**2 / amconv)
                log.log_rank0(f'  Computing omega({ie_g:6d}) = {omega_val:12.5f} cm-1')
            else:
                log.log_rank0(f'  Computing E({ie_g:6d}) = {egrid[ie_g]:12.5f} eV')

        gC_k, sgmL_k, sgmR_k = _initialize_k_dependent_operators(
            data=data, nkpts_par=nkpts_par, dimC=dimC
        )
        avg_iter = 0.0

        for ik in range(nkpts_par):
            gC, sigma_L, sigma_R, niter_sum = compute_kpoint_conductor_quantities(
                data=data,
                blc_blocks=blc_blocks,
                egrid=egrid,
                delta=delta,
                ie_g=ie_g,
                ik=ik,
            )
            avg_iter += niter_sum

            accumulate_dos(dos, dos_k, gC, wk_par, ie_g, ik)
            accumulate_transmission(
                conduct,
                conduct_k,
                gC,
                sigma_L,
                sigma_R,
                wk_par,
                ie_g,
                ik,
                data=data,
                delta=delta,
                rank=rank,
                vkpt=vkpt_par3D,
            )

            if data.symmetry.write_gf and gC_k is not None:
                gC_k[ik] = gC
            if data.symmetry.write_lead_sgm and sgmL_k is not None and sgmR_k is not None:
                sgmL_k[ik], sgmR_k[ik] = sigma_L, sigma_R

        _transform_k_to_r_at_energy(
            data=data,
            ie_g=ie_g,
            gC_k=gC_k,
            sgmL_k=sgmL_k,
            sgmR_k=sgmR_k,
            gf_out=gf_out,
            rsgmL_out=rsgmL_out,
            rsgmR_out=rsgmR_out,
            nrtot_par=nrtot_par,
            vr_par3D=vr_par3D,
            vkpt_par3D=vkpt_par3D,
            wk_par=wk_par,
        )

        if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
            avg_iter /= 2 * nkpts_par
            log.log_rank0(f'  T matrix converged after avg. # of iterations {avg_iter:10.3f}\n')
            global_timing.timing_upto_now('do_conductor', label='Total time spent up to now')

    _reduce_results(
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
    _write_operators(
        rank=rank,
        data=data,
        gf_out=gf_out,
        rsgmL_out=rsgmL_out,
        rsgmR_out=rsgmR_out,
        ivr_par3D=ivr_par3D,
        egrid=egrid,
        dimC=dimC,
    )
    _write_output(
        rank=rank,
        data=data,
        conduct=conduct,
        dos=dos,
        conduct_k=conduct_k,
        dos_k=dos_k,
        egrid=egrid,
    )

    return conduct, dos, conduct_k, dos_k, gf_out, rsgmL_out, rsgmR_out, egrid
