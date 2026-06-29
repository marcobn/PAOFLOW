from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.calculators.green import compute_conductor_green_function
from PAOFLOW.transport.calculators.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup
from PAOFLOW.transport.data import ConductorData


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
