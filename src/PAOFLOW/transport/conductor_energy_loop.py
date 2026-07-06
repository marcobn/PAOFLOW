from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.transport.calculators.accumulation import accumulate_dos, accumulate_transmission
from PAOFLOW.transport.conductor_kpoint import (
    compute_kpoint_green,
    compute_kpoint_self_energies,
)
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham
from PAOFLOW.transport.utils.timing import global_timing
from PAOFLOW.utils.constants import AMCONV, RYDCM1


@dataclass
class ParallelGridGeometry:
    """Local (per-MPI-rank) k-point and real-space grid geometry.

    Attributes
    ----------
    nkpts_par : int
        Number of local k-points.
    nrtot_par : int
        Number of local real-space vectors.
    dimC : int
        Conductor block dimension.
    vkpt_par3D : NDArray[np.float64]
        Local k-point coordinates, shape ``(3, nkpts_par)``.
    vr_par3D : NDArray[np.float64]
        Local real-space vectors, shape ``(nrtot_par, 3)``.
    wk_par : NDArray[np.float64]
        Local k-point weights, shape ``(nkpts_par,)``.
    """

    nkpts_par: int
    nrtot_par: int
    dimC: int
    vkpt_par3D: NDArray[np.float64]
    vr_par3D: NDArray[np.float64]
    wk_par: NDArray[np.float64]


@dataclass
class ConductorAccumulators:
    """Transport observable accumulators updated across the ``(E, k)`` grid.

    Attributes
    ----------
    conduct : NDArray[np.float64]
        Total conductance and optional eigenchannels, shape ``(1 + neigchn, ne)``.
    dos : NDArray[np.float64]
        Total DOS, shape ``(ne,)``.
    conduct_k : NDArray[np.float64]
        k-resolved conductance, shape ``(1 + neigchn, nkpts_par, ne)``.
    dos_k : NDArray[np.float64]
        k-resolved DOS, shape ``(ne, nkpts_par)``.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function accumulator, shape
        ``(ne, nrtot_par, dimC, dimC)``, or ``None`` when not requested.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy accumulator, same shape, or ``None``.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy accumulator, same shape, or ``None``.
    """

    conduct: NDArray[np.float64]
    dos: NDArray[np.float64]
    conduct_k: NDArray[np.float64]
    dos_k: NDArray[np.float64]
    gf_out: NDArray[np.complex128] | None = None
    rsgmL_out: NDArray[np.complex128] | None = None
    rsgmR_out: NDArray[np.complex128] | None = None


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
    """Allocate per-energy k-resolved operator buffers.

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


def compute_kpoint_conductor_quantities(
    *,
    data: ConductorData,
    blc_blocks: Mapping[str, Any],
    egrid: NDArray[np.float64],
    delta: float,
    ie_g: int,
    ik: int,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int]:
    """Compute conductor operators at one energy and one k-point.

    Parameters
    ----------
    data : ConductorData
        Validated transport input data and runtime flags.
    blc_blocks : Mapping[str, Any]
        Dictionary of block operators. Required keys include ``blc_00C``,
        ``blc_00L``, ``blc_01L``, ``blc_00R``, ``blc_01R``, ``blc_LC``,
        and ``blc_CR``.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    delta : float
        Positive infinitesimal broadening added to the retarded Green's
        function denominator.
    ie_g : int
        Global energy index into ``egrid``.
    ik : int
        Local k-point index.

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], int]
        ``(gC, sigma_L, sigma_R, niter_sum)``, where ``gC`` is the conductor
        retarded Green's function, ``sigma_L`` and ``sigma_R`` are lead
        self-energies, and ``niter_sum`` is the total Sancho-Rubio iteration
        count used to converge the transfer matrices.

    Notes
    -----
    The sequence is:

    1. Update k-resolved Hamiltonian blocks at the selected energy.
    2. Compute lead self-energies from block couplings.
    3. Build the conductor Green's function

       .. math::

           G_C^r = \\left[\\left(E + i\\delta\\right)I - H_C - \\Sigma_L - \\Sigma_R\\right]^{-1}.
    """
    sigma_L, sigma_R, niter_sum = compute_kpoint_self_energies(
        blc_blocks=blc_blocks,
        ik=ik,
        ie_g=ie_g,
        egrid=egrid,
        delta=delta,
        shift_L=data.shift_L,
        shift_C=data.shift_C,
        shift_R=data.shift_R,
        shift_corr=getattr(data, 'shift_corr', 0.0),
        leads_are_identical=data.advanced.leads_are_identical,
        niterx=data.iteration.niterx,
        transfer_thr=data.iteration.transfer_thr,
        nfailx=data.iteration.nfailx,
        surface=data.advanced.surface,
    )

    gC = compute_kpoint_green(
        blc_00C=blc_blocks['blc_00C'].at_k(ik),
        sigma_L=sigma_L,
        sigma_R=sigma_R,
        delta=delta,
        surface=data.advanced.surface,
    )

    return gC, sigma_L, sigma_R, niter_sum


def transform_k_to_r_at_energy(
    *,
    data: ConductorData,
    ie_g: int,
    gC_k: NDArray[np.complex128] | None,
    sgmL_k: NDArray[np.complex128] | None,
    sgmR_k: NDArray[np.complex128] | None,
    acc: ConductorAccumulators,
    geom: ParallelGridGeometry,
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
    acc : ConductorAccumulators
        Observable accumulators; the real-space ``gf_out``, ``rsgmL_out``, and
        ``rsgmR_out`` arrays are updated in place for ``ie_g``.
    geom : ParallelGridGeometry
        Local k-point and real-space grid geometry.

    Returns
    -------
    None
        Updates ``acc.gf_out``, ``acc.rsgmL_out``, and ``acc.rsgmR_out`` in place
        for ``ie_g`` when the corresponding output flags are enabled.
    """
    nrtot_par = geom.nrtot_par
    vr_par3D = geom.vr_par3D
    vkpt_par3D = geom.vkpt_par3D
    wk_par = geom.wk_par
    if data.symmetry.write_gf and gC_k is not None and acc.gf_out is not None:
        for ir in range(nrtot_par):
            acc.gf_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], gC_k, vkpt_par3D.T, wk_par)
    if data.symmetry.write_lead_sgm and sgmL_k is not None and sgmR_k is not None:
        if acc.rsgmL_out is None or acc.rsgmR_out is None:
            return
        for ir in range(nrtot_par):
            acc.rsgmL_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], sgmL_k, vkpt_par3D.T, wk_par)
            acc.rsgmR_out[ie_g, ir] = compute_rham(vr_par3D[ir, :], sgmR_k, vkpt_par3D.T, wk_par)


def process_energy_point(
    *,
    data: ConductorData,
    blc_blocks: Mapping[str, Any],
    egrid: NDArray[np.float64],
    delta: float,
    ie_g: int,
    ie_start: int,
    ie_end: int,
    rank: int,
    geom: ParallelGridGeometry,
    acc: ConductorAccumulators,
) -> None:
    """Process all k-points for one local energy index.

    Parameters
    ----------
    data : ConductorData
        Validated transport input and runtime flags.
    blc_blocks : Mapping[str, Any]
        Block-operator dictionary for Hamiltonian and coupling terms.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    delta : float
        Broadening parameter used in retarded quantities.
    ie_g : int
        Global energy index being processed.
    ie_start : int
        First energy index owned by this rank.
    ie_end : int
        Last energy index owned by this rank.
    rank : int
        MPI rank.
    geom : ParallelGridGeometry
        Local k-point and real-space grid geometry.
    acc : ConductorAccumulators
        Observable accumulators updated in place for ``ie_g``.

    Returns
    -------
    None
        Updates ``acc.conduct``, ``acc.dos``, ``acc.conduct_k``, ``acc.dos_k``
        and optional real-space operator arrays in place for ``ie_g``.
    """
    nkpts_par = geom.nkpts_par
    wk_par = geom.wk_par
    vkpt_par3D = geom.vkpt_par3D

    nprint = data.iteration.nprint
    if (ie_g % nprint == 0 or ie_g == 0 or ie_g == egrid.shape[0] - 1) and rank == 0:
        if data.carriers == 'phonons':
            omega_val = np.sqrt(egrid[ie_g] * RYDCM1**2 / AMCONV)
            log.log_rank0(f'  Computing omega({ie_g:6d}) = {omega_val:12.5f} cm-1')
        else:
            log.log_rank0(f'  Computing E({ie_g:6d}) = {egrid[ie_g]:12.5f} eV')

    gC_k, sgmL_k, sgmR_k = initialize_kpoint_operator_buffers(
        data=data,
        nkpts_par=nkpts_par,
        dimC=geom.dimC,
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

        accumulate_dos(acc.dos, acc.dos_k, gC, wk_par, ie_g, ik)
        accumulate_transmission(
            acc.conduct,
            acc.conduct_k,
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

    transform_k_to_r_at_energy(
        data=data,
        ie_g=ie_g,
        gC_k=gC_k,
        sgmL_k=sgmL_k,
        sgmR_k=sgmR_k,
        acc=acc,
        geom=geom,
    )

    if (ie_g % nprint == 0 or ie_g == ie_start or ie_g == ie_end) and rank == 0:
        avg_iter /= 2 * nkpts_par
        log.log_rank0(f'  T matrix converged after avg. # of iterations {avg_iter:10.3f}\n')
        global_timing.timing_upto_now('do_conductor', label='Total time spent up to now')


def reduce_conductor_results(
    *,
    comm: MPI.Comm,
    data: ConductorData,
    acc: ConductorAccumulators,
) -> None:
    """Reduce distributed transport results across MPI ranks.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator used for collective reductions.
    data : ConductorData
        Validated transport input and runtime flags.
    acc : ConductorAccumulators
        Observable accumulators reduced in place across ranks.

    Returns
    -------
    None
        Applies in-place ``MPI.Allreduce`` with ``MPI.SUM`` to all enabled arrays.
    """
    comm.Allreduce(MPI.IN_PLACE, acc.conduct, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, acc.conduct_k, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, acc.dos, op=MPI.SUM)
    comm.Allreduce(MPI.IN_PLACE, acc.dos_k, op=MPI.SUM)

    if data.symmetry.write_gf and acc.gf_out is not None:
        comm.Allreduce(MPI.IN_PLACE, acc.gf_out, op=MPI.SUM)
    if data.symmetry.write_lead_sgm and acc.rsgmL_out is not None and acc.rsgmR_out is not None:
        comm.Allreduce(MPI.IN_PLACE, acc.rsgmL_out, op=MPI.SUM)
        comm.Allreduce(MPI.IN_PLACE, acc.rsgmR_out, op=MPI.SUM)
