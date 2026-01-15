import numpy as np
from PAOFLOW.transport.calculators.green import compute_surface_green_function
from PAOFLOW.transport.calculators.transfer import compute_surface_transfer_matrices


from PAOFLOW.transport.hamiltonian.operator_blc import OperatorBlockView


def build_self_energies_from_blocks(
    blc_00R: OperatorBlockView,
    blc_01R: OperatorBlockView,
    blc_00L: OperatorBlockView,
    blc_01L: OperatorBlockView,
    blc_CR: OperatorBlockView,
    blc_LC: OperatorBlockView,
    leads_are_identical: bool,
    delta: float = 1e-5,
    niterx: int = 200,
    transfer_thr: float = 1e-12,
    fail_counter: dict | None = None,
    fail_limit: int = 10,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Construct lead self-energies Σ_R and Σ_L using Green's function recursion.

    Parameters
    ----------
    `blc_00R` : OperatorBlockView
        On-site Hamiltonian of the right lead.
    `blc_01R` : OperatorBlockView
        Hopping between right lead cells.
    `blc_00L` : OperatorBlockView
        On-site Hamiltonian of the left lead.
    `blc_01L` : OperatorBlockView
        Hopping between left lead cells.
    `blc_CR` : OperatorBlockView
        Coupling block between conductor and right lead.
    `blc_LC` : OperatorBlockView
        Coupling block between conductor and left lead.
    `leads_are_identical` : bool
        If True, reuse transfer matrices for both leads.
    `delta` : float
        Broadening parameter.
    `niterx` : int
        Max iterations for Sancho-Rubio method.
    `transfer_thr` : float
        Threshold for transfer matrix convergence.
    `fail_counter` : dict or None
        Shared dict for tracking failures across calls.
    `fail_limit` : int
        Maximum allowed failures.
    `verbose` : bool
        Enable logging information.

    Returns
    -------
    `sigma_R` : (nC, nC) complex ndarray
        Self-energy from the right lead.
    `sigma_L` : (nC, nC) complex ndarray
        Self-energy from the left lead.
    `niter_R` : int
        Iteration count for right lead.
    `niter_L` : int
        Iteration count for left lead.

    Notes
    -----
    The conductor self-energies are assembled as:

        Σ_R = H_CR · G_surf(R) · H_CR†
        Σ_L = H_LC† · G_surf(L) · H_LC

    If the leads are identical, the same transfer matrices are reused,
    but the surface Green's function must still be computed with `igreen=-1` for Σ_L.
    """
    tot, tott, niter_R = compute_surface_transfer_matrices(
        h_eff=blc_00R.aux,
        s_eff=blc_00R.S,
        t_coupling=blc_01R.aux,
        delta=delta,
        niterx=niterx,
        transfer_thr=transfer_thr,
        fail_counter=fail_counter,
        fail_limit=fail_limit,
        verbose=verbose,
    )

    gR = compute_surface_green_function(
        h_eff=blc_00R.aux,
        s_eff=blc_00R.S,
        t_coupling=blc_01R.aux,
        transfer_matrix=tot,
        transfer_matrix_conj=tott,
        igreen=1,
        delta=delta,
    )

    if leads_are_identical:
        gL = compute_surface_green_function(
            h_eff=blc_00R.aux,
            s_eff=blc_00R.S,
            t_coupling=blc_01R.aux,
            transfer_matrix=tot,
            transfer_matrix_conj=tott,
            igreen=-1,
            delta=delta,
        )
        niter_L = niter_R
    else:
        totL, tottL, niter_L = compute_surface_transfer_matrices(
            h_eff=blc_00L.aux,
            s_eff=blc_00L.S,
            t_coupling=blc_01L.aux,
            delta=delta,
            niterx=niterx,
            transfer_thr=transfer_thr,
            fail_counter=fail_counter,
            fail_limit=fail_limit,
            verbose=verbose,
        )
        gL = compute_surface_green_function(
            h_eff=blc_00L.aux,
            s_eff=blc_00L.S,
            t_coupling=blc_01L.aux,
            transfer_matrix=totL,
            transfer_matrix_conj=tottL,
            igreen=-1,
            delta=delta,
        )

    sigma_R = blc_CR.aux @ gR @ blc_CR.aux.conj().T
    sigma_L = blc_LC.aux.conj().T @ gL @ blc_LC.aux

    return sigma_R, sigma_L, niter_R, niter_L


def compute_lead_surface_green_function(
    h_eff: np.ndarray,
    s_eff: np.ndarray,
    t_coupling: np.ndarray,
    delta: float = 1e-5,
    direction: str = "right",
    niterx: int = 200,
    transfer_thr: float = 1e-12,
    fail_counter: dict = None,
    fail_limit: int = 10,
    verbose: bool = False,
) -> tuple[np.ndarray, int]:
    """
    Compute the **surface Green's function** for a semi-infinite lead

    Parameters
    ----------
    `h_eff` : np.ndarray
        On-site effective block (E·S - H) for the lead principal layer.
    `s_eff` : np.ndarray
        Overlap matrix S_00 of the lead.
    `t_coupling` : np.ndarray
        Inter-cell coupling block (effective form consistent with `h_eff`).
    `delta` : float
        Imaginary broadening parameter.
    `direction` : {'right', 'left'}
        Indicates which semi-infinite lead is being modeled.
    `niterx` : int
        Maximum number of iterations for Sancho-Rubio method.
    `transfer_thr` : float
        Convergence threshold for transfer matrices.
    `fail_counter` : dict
        Optional mutable dict to track convergence failures.
    `fail_limit` : int
        Maximum number of allowed convergence failures.
    `verbose` : bool
        Whether to log progress and warnings.

    Returns
    -------
    `g_surf` : np.ndarray
        Surface Green's function of the lead principal layer.
    `niter` : int
        Number of iterations used in the Sancho-Rubio recursion.

    Notes
    -----
    This routine now returns the surface Green’s function ``G_surf(E)``.

        Right surface: G_surf = [ (E+iδ)·S - H - H_01·T ]⁻¹
        Left  surface: G_surf = [ (E+iδ)·S - H - H_01†·T† ]⁻¹

    where T and T† are transfer matrices from the Sancho–Rubio iteration.
    """
    tot, tott, niter = compute_surface_transfer_matrices(
        h_eff,
        s_eff,
        t_coupling,
        delta=delta,
        niterx=niterx,
        transfer_thr=transfer_thr,
        fail_counter=fail_counter,
        fail_limit=fail_limit,
        verbose=verbose,
    )

    igreen = 1 if direction == "right" else -1

    g_surf = compute_surface_green_function(
        h_eff, s_eff, t_coupling, tot, tott, igreen=igreen, delta=delta
    )

    return g_surf, niter
