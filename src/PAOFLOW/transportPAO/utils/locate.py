import numpy as np


def locate(xx: np.ndarray, x: float) -> int:
    """
    Locate the index `j` in `xx` such that `xx[j] <= x < xx[j+1]`.

    Parameters
    ----------
    `xx` : ndarray
        Monotonic array (ascending or descending) of values.
    `x` : float
        Value to locate within the array.

    Returns
    -------
    `j` : int
        Index satisfying `xx[j] <= x < xx[j+1]`.

    Raises
    ------
    ValueError
        If the value is outside the bounds of the array.
    """
    n = len(xx)
    jl = 0
    ju = n
    is_ascending = xx[-1] > xx[0]

    while ju - jl > 1:
        jm = (ju + jl) // 2
        if (x > xx[jm]) == is_ascending:
            jl = jm
        else:
            ju = jm

    if jl == 0 or jl == n - 1:
        raise ValueError("Located index out of valid bounds.")

    return jl
