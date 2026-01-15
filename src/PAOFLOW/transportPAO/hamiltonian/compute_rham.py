import numpy as np
from PAOFLOW.transportPAO.utils.timing import timed_function


@timed_function("compute_rham")
def compute_rham(
    rvec: np.ndarray,
    Hk: np.ndarray,
    kpts: np.ndarray,
    wk: np.ndarray,
) -> np.ndarray:
    """Compute the real-space Hamiltonian.

    Perform an inverse Fourier transform to compute the real-space Hamiltonian
    corresponding to a single lattice vector.

    Parameters
    ----------
    `rvec` : (3,) ndarray
        Real-space lattice vector in Cartesian coordinates.
    `Hk` : (nkpts, n, n) complex ndarray
        Hamiltonian matrix at each k-point in reciprocal space.
    `kpts` : (3, nkpts) ndarray
        Reciprocal space k-points in Cartesian coordinates.
    `wk` : (nkpts,) ndarray
        Integration weights associated with each k-point.

    Returns
    -------
    `Hr` : (n, n) complex ndarray
        Hamiltonian matrix in real space corresponding to the given `rvec`.

    Notes
    -----
    This function computes the Hamiltonian in real space by summing over
    all k-points in reciprocal space:

        H(R) = ∑_k w_k · exp(-i k · R) · H(k)

    where:
        - `R` is the real-space lattice vector,
        - `k` is the reciprocal space vector (in Cartesian coordinates),
        - `H(k)` is the Hamiltonian at k-point `k`,
        - `w_k` is the integration weight for `k`.

    The phase factor exp(-i k · R) ensures the transformation respects periodic
    boundary conditions and correctly maps between reciprocal and real-space
    representations of the Hamiltonian.
    """
    nkpts, n, _ = Hk.shape
    Hr = np.zeros((n, n), dtype=np.complex128)

    for ik in range(nkpts):
        arg = np.dot(kpts[:, ik], rvec)
        phase = np.cos(arg) - 1j * np.sin(arg)  # exp(-i * arg)
        weight = phase * wk[ik]
        for j in range(n):
            for i in range(n):
                Hr[i, j] += weight * Hk[ik, i, j]
    return Hr
