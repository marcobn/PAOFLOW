from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from mpi4py import MPI
from numpy.typing import NDArray

import PAOFLOW.transport.io.log_module as log
from PAOFLOW.transport.calculators.green import compute_conductor_green_function
from PAOFLOW.transport.calculators.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW.transport.hamiltonian.compute_rham import compute_rham
from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW.transport.data import ConductorData
from PAOFLOW.transport.observables.accumulation import accumulate_dos, accumulate_transmission
from PAOFLOW.utils.constants import AMCONV, RYDCM1
from PAOFLOW.transport.utils.timing import global_timing


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
    hamiltonian_setup(
        ik=ik,
        ie_g=ie_g,
        egrid=egrid,
        shift_L=data.shift_L,
        shift_C=data.shift_C,
        shift_R=data.shift_R,
        shift_C_corr=getattr(data, 'shift_corr', 0.0),
        blc_blocks=blc_blocks,
        ie_buff=1,
    )

    sigma_R, sigma_L, niter_R, niter_L = build_self_energies_from_blocks(
        blc_00R=blc_blocks['blc_00R'].at_k(ik),
        blc_01R=blc_blocks['blc_01R'].at_k(ik),
        blc_00L=blc_blocks['blc_00L'].at_k(ik),
        blc_01L=blc_blocks['blc_01L'].at_k(ik),
        blc_CR=blc_blocks['blc_CR'].at_k(ik),
        blc_LC=blc_blocks['blc_LC'].at_k(ik),
        leads_are_identical=data.advanced.leads_are_identical,
        delta=delta,
        niterx=data.iteration.niterx,
        transfer_thr=data.iteration.transfer_thr,
        fail_counter=None,
        fail_limit=data.iteration.nfailx,
        verbose=False,
    )

    gC = compute_conductor_green_function(
        blc_00C=blc_blocks['blc_00C'].at_k(ik),
        sigma_l=sigma_L,
        sigma_r=sigma_R if not data.advanced.surface else None,
        delta=delta,
        surface=data.advanced.surface,
    )

    niter_sum = niter_R + (niter_L if not data.advanced.leads_are_identical else 0)
    return gC, sigma_L, sigma_R, niter_sum


def transform_k_to_r_at_energy(
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
    nkpts_par: int,
    dimC: int,
    nrtot_par: int,
    vkpt_par3D: NDArray[np.float64],
    vr_par3D: NDArray[np.float64],
    wk_par: NDArray[np.float64],
    conduct: NDArray[np.float64],
    dos: NDArray[np.float64],
    conduct_k: NDArray[np.float64],
    dos_k: NDArray[np.float64],
    gf_out: NDArray[np.complex128] | None,
    rsgmL_out: NDArray[np.complex128] | None,
    rsgmR_out: NDArray[np.complex128] | None,
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
    nkpts_par : int
        Number of local k-points.
    dimC : int
        Conductor block dimension.
    nrtot_par : int
        Number of local real-space vectors.
    vkpt_par3D : NDArray[np.float64]
        Local k-point coordinates, shape ``(3, nkpts_par)``.
    vr_par3D : NDArray[np.float64]
        Local real-space vectors, shape ``(nrtot_par, 3)``.
    wk_par : NDArray[np.float64]
        Local k-point weights, shape ``(nkpts_par,)``.
    conduct : NDArray[np.float64]
        Total conductance accumulator.
    dos : NDArray[np.float64]
        Total DOS accumulator.
    conduct_k : NDArray[np.float64]
        k-resolved conductance accumulator.
    dos_k : NDArray[np.float64]
        k-resolved DOS accumulator.
    gf_out : NDArray[np.complex128] or None
        Real-space Green's function output accumulator.
    rsgmL_out : NDArray[np.complex128] or None
        Real-space left self-energy output accumulator.
    rsgmR_out : NDArray[np.complex128] or None
        Real-space right self-energy output accumulator.

    Returns
    -------
    None
        Updates ``conduct``, ``dos``, ``conduct_k``, ``dos_k`` and optional
        real-space operator arrays in place for ``ie_g``.
    """
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
        dimC=dimC,
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

    transform_k_to_r_at_energy(
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


def reduce_conductor_results(
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
