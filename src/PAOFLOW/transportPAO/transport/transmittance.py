import numpy as np
from numpy.linalg import eigh
from scipy.linalg import solve

from transportPAO.io.input_parameters import ConductFormula
from transportPAO.utils.timing import timed_function


@timed_function("transmittance")
def evaluate_transmittance(
    gamma_L: np.ndarray,
    gamma_R: np.ndarray,
    G_ret: np.ndarray,
    formula: ConductFormula,
    do_eigenchannels: bool,
    do_eigplot: bool,
    sgm_corr: np.ndarray | None = None,
    eta: float = 1e-5,
    S_overlap: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    r"""
    Evaluate the quantum transmittance from lead coupling matrices and the retarded Green's function using the LANDAUER formula in the MEAN FIELD case otherwise the generalized expression derived in A. Ferretti et al, PRL 94, 116802 (2005).

    Parameters
    ----------
    `gamma_L` : (n, n) complex ndarray
        Coupling matrix for the left lead.
    `gamma_R` : (n, n) complex ndarray
        Coupling matrix for the right lead.
    `G_ret` : (n, n) complex ndarray
        Retarded Green's function of the conductor.
    `formula` : {"landauer", "generalized"}
        Choice of transmittance formula.
    `do_eigenchannels` : bool
        Whether to compute eigenchannel decomposition.
    `do_eigplot` : bool
        Whether to compute orbital-resolved eigenchannel information.
    `sgm_corr` : (n, n) complex ndarray, optional
        Correlation self-energy for generalized formula (required if formula == "generalized").
    `eta` : float
        Small imaginary part added for regularization in the generalized formula.
    `S_overlap` : (n, n) complex ndarray, optional
        Overlap matrix used in eigenvalue problem (if non-orthogonal basis).

    Returns
    -------
    `conduct` : (n,) real ndarray
        Channel-resolved transmission probabilities.
    `z_eigplot` : (n, n) complex ndarray or None
        Eigenvectors for plotting eigenchannels, if `do_eigplot` is True.

    Notes
    -----
    For the landauer formula:
        T(E) = Tr[Gamma_L * G^r * Gamma_R * G^a]

    For the generalized formula:
        T(E) = Tr[Gamma_L * G^r * Gamma_R * G^a * Lambda]
    where:
        Lambda = I + (Gamma_L + Gamma_R + 2*eta)^-1 * (i (Sigma - Sigma^\dagger))

    When do_eigenchannels is False:
        Transmission is computed as the diagonal of T = G^a * Gamma_L * G^r * Gamma_R (and optionally * Lambda)

    When do_eigenchannels is True:
        Transmission eigenchannels are computed from the eigenvalues of the Hermitian matrix:
            T = sqrt(A_L) * Gamma_R * sqrt(A_L)
        where:
            A_L = G^a * Gamma_L * G^r

        Eigenvalues w < 0 are clipped to 0 for numerical stability, and an error is raised
        if any eigenvalue is found to be significantly negative (< -1e-6).
    """
    assert gamma_L.shape == gamma_R.shape == G_ret.shape
    dim = gamma_L.shape[0]

    G_adv = G_ret.conj().T
    work = gamma_L @ G_ret

    if do_eigenchannels:
        A_L = G_adv @ work
    else:
        work = G_adv @ work

    if formula == "generalized":
        assert sgm_corr is not None, (
            "Correlation self-energy required for generalized formula."
        )
        lambda_corr = 1j * (sgm_corr - sgm_corr.conj().T)
        regularized = gamma_L + gamma_R + 2 * eta * np.eye(dim)
        lambda_mat = solve(regularized, lambda_corr)
        lambda_mat += np.eye(dim)
    else:
        lambda_mat = np.eye(dim)

    if not do_eigenchannels:
        Tmat = work @ gamma_R
        if formula == "generalized":
            Tmat = Tmat @ lambda_mat
        conduct = np.real(np.diag(Tmat))
        return conduct, None

    # -----------------------------------------------------
    # Eigenchannels, no eigplot: √Γ_R (G^a Γ_L G^r) √Γ_R
    # -----------------------------------------------------
    elif do_eigenchannels and not do_eigplot:
        evals_R, vecs_R = eigh(gamma_R)
        evals_R = np.clip(evals_R, 0.0, None)
        sqrt_gamma_R = vecs_R @ np.diag(np.sqrt(evals_R)) @ vecs_R.conj().T

        Tmat = sqrt_gamma_R @ A_L @ sqrt_gamma_R
        if formula == "generalized":
            Tmat = Tmat @ lambda_mat

        evals, _ = eigh(-Tmat) if S_overlap is None else eigh(-Tmat, S_overlap)
        conduct = -np.real(evals)
        return conduct, None
    # ----------------------------------------------------------------------
    # Eigenchannels with eigplot: √(G^a Γ_L G^r) Γ_R √(G^a Γ_L G^r)
    # ----------------------------------------------------------------------
    elif do_eigenchannels and do_eigplot:
        evals, vecs = eigh(A_L, S_overlap) if S_overlap is not None else eigh(A_L)
        if np.any(evals < -1e-6):
            raise ValueError("A_L not positive semi-definite.")
        evals = np.clip(evals, 0.0, None)

        sqrt_A_L = vecs @ np.diag(np.sqrt(evals)) @ vecs.conj().T
        Tmat = sqrt_A_L @ gamma_R @ sqrt_A_L

        evals, vecs = eigh(-Tmat, S_overlap) if S_overlap is not None else eigh(-Tmat)
        conduct = -np.real(evals)
        return conduct, vecs

    else:
        raise ValueError("Unexpected combination of eigenchannel flags.")
