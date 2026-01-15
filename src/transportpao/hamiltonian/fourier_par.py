import numpy as np
from PAOFLOW_QTpy.utils.timing import timed_function


@timed_function("fourier_par")
def fourier_transform_real_to_kspace(
    rh: np.ndarray,
    wr: np.ndarray,
    table: np.ndarray,
) -> np.ndarray:
    """Perform a 2D Fourier transform from real space to reciprocal space.

    Parameters
    ----------
    `rh` : ndarray of shape (dim1, dim2, nR)
        Real-space operator matrices for each lattice vector R.
    `wr` : ndarray of shape (nR,)
        Integration weights for each real-space vector R.
    `table` : ndarray of shape (nR, nk)
        Phase factors, typically exp(i * k · R), for each R and k-point.

    Returns
    -------
    `kh` : ndarray of shape (dim1, dim2, nk)
        k-space operator matrices at each k-point.

    Notes
    -----
    Computes:

        kh[i, j, k] = sum_R wr[R] * table[R, k] * rh[i, j, R]

    for each orbital pair `(i, j)` and k-point index `k`.
    """
    dim1, dim2, nR = rh.shape
    _, nk = table.shape
    kh = np.zeros((dim1, dim2, nk), dtype=np.complex128)

    for k in range(nk):
        for R in range(nR):
            kh[:, :, k] += wr[R] * table[R, k] * rh[:, :, R]

    return kh
