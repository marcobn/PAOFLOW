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

    if (x < xx[0] and is_ascending) or (x > xx[0] and not is_ascending):
        raise ValueError('Value is outside the bounds of the array.')
    if (x >= xx[-1] and is_ascending) or (x <= xx[-1] and not is_ascending):
        raise ValueError('Value is outside the bounds of the array.')

    while ju - jl > 1:
        jm = (ju + jl) // 2
        if (x > xx[jm]) == is_ascending:
            jl = jm
        else:
            ju = jm

    if jl == n - 1:
        raise ValueError('Located index out of valid bounds.')

    return jl
