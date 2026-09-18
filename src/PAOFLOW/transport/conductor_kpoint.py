from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from PAOFLOW.transport.calculators.green import compute_conductor_green_function
from PAOFLOW.transport.calculators.leads_self_energy import build_self_energies_from_blocks
from PAOFLOW.transport.hamiltonian.hamiltonian_setup import hamiltonian_setup


def compute_kpoint_self_energies(
    *,
    blc_blocks: Mapping[str, Any],
    ik: int,
    ie_g: int,
    egrid: NDArray[np.float64],
    delta: float,
    shift_L: float,
    shift_C: float,
    shift_R: float,
    shift_corr: float,
    leads_are_identical: bool,
    niterx: int,
    transfer_thr: float,
    nfailx: int,
    surface: bool,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], int]:
    r"""Compute lead self-energies at one ``(E, k)`` point.

    This is the single per-point self-energy kernel shared by the full-grid
    energy loop and the staged conductor API.

    Parameters
    ----------
    blc_blocks : Mapping[str, Any]
        Block-operator dictionary. Required keys include ``blc_00R``,
        ``blc_01R``, ``blc_00L``, ``blc_01L``, ``blc_CR``, and ``blc_LC``.
    ik : int
        Local k-point index.
    ie_g : int
        Global energy index into ``egrid``.
    egrid : NDArray[np.float64]
        Energy grid in eV, shape ``(ne,)``.
    delta : float
        Positive infinitesimal broadening.
    shift_L, shift_C, shift_R : float
        Rigid on-site energy shifts (eV) for the left-lead, conductor, and
        right-lead blocks.
    shift_corr : float
        On-site energy shift (eV) applied to the correlation self-energy.
    leads_are_identical : bool
        If ``True``, the left self-energy reuses the right transfer matrices and
        its iteration count is excluded from the returned sum.
    niterx : int
        Maximum number of transfer-matrix iterations.
    transfer_thr : float
        Convergence threshold for the transfer-matrix iteration.
    nfailx : int
        Maximum number of allowed convergence failures.
    surface : bool
        Surface-mode flag (unused here; retained for signature symmetry with
        :func:`compute_kpoint_green`).

    Returns
    -------
    tuple[NDArray[np.complex128], NDArray[np.complex128], int]
        ``(sigma_L, sigma_R, niter_sum)``, where self-energies have shape
        ``(dimC, dimC)`` and ``niter_sum`` is the total Sancho-Rubio iteration
        count used to converge the transfer matrices.

    Notes
    -----
    The sequence is:

    1. Update k-resolved Hamiltonian blocks at the selected energy via
       :func:`hamiltonian_setup`.
    2. Build lead self-energies from block couplings.
    """
    hamiltonian_setup(
        ik=ik,
        ie_g=ie_g,
        egrid=egrid,
        shift_L=shift_L,
        shift_C=shift_C,
        shift_R=shift_R,
        shift_C_corr=shift_corr,
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
        leads_are_identical=leads_are_identical,
        delta=delta,
        niterx=niterx,
        transfer_thr=transfer_thr,
        fail_counter=None,
        fail_limit=nfailx,
        verbose=False,
    )

    niter_sum = niter_R + (niter_L if not leads_are_identical else 0)
    return sigma_L, sigma_R, niter_sum


def compute_kpoint_green(
    *,
    blc_00C: Any,
    sigma_L: NDArray[np.complex128],
    sigma_R: NDArray[np.complex128] | None,
    delta: float,
    surface: bool,
) -> NDArray[np.complex128]:
    r"""Compute the conductor retarded Green's function at one k-point.

    This is the single per-point Green's-function kernel shared by the full-grid
    energy loop and the staged conductor API.

    Parameters
    ----------
    blc_00C : Any
        Conductor on-site block operator evaluated at the target k-point
        (``blc_blocks['blc_00C'].at_k(ik)``).
    sigma_L : NDArray[np.complex128]
        Left lead self-energy, shape ``(dimC, dimC)``.
    sigma_R : NDArray[np.complex128] or None
        Right lead self-energy, shape ``(dimC, dimC)``. Ignored when
        ``surface`` is ``True``.
    delta : float
        Positive infinitesimal broadening added to the retarded denominator.
    surface : bool
        If ``True``, compute the surface Green's function (right self-energy
        excluded).

    Returns
    -------
    NDArray[np.complex128]
        Retarded conductor Green's function, shape ``(dimC, dimC)``.

    Notes
    -----
    .. math::

        G_C^r = \left[\left(E + i\delta\right)I - H_C
            - \Sigma_L - \Sigma_R\right]^{-1}.
    """
    return compute_conductor_green_function(
        blc_00C=blc_00C,
        sigma_l=sigma_L,
        sigma_r=sigma_R if not surface else None,
        delta=delta,
        surface=surface,
    )
