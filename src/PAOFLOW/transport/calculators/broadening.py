"""Lead broadening observable helpers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_broadening_matrix(
    self_energy: NDArray[np.complex128],
) -> NDArray[np.complex128]:
    r"""Compute the lead broadening matrix from a retarded self-energy.

    Parameters
    ----------
    self_energy : NDArray[np.complex128]
        Retarded lead self-energy :math:`\Sigma` with shape ``(dimC, dimC)``.

    Returns
    -------
    NDArray[np.complex128]
        Broadening matrix :math:`\Gamma = i(\Sigma - \Sigma^\dagger)` with
        shape ``(dimC, dimC)``.

    Notes
    -----
    This preserves the exact broadening definition already used in the
    conductor workflow.
    """
    return 1j * (self_energy - self_energy.conj().T)
