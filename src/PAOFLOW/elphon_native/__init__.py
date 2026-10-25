"""Optional Rust kernels of the electron-phonon pipeline (``paoflow_rs``).

The pure-NumPy implementations stay the reference; the native kernels are used when the
extension is importable and ``PAOFLOW_ELPHON_DISABLE`` is not set.  Parity is tested at
``1e-10`` relative (the summation order differs).
"""

from __future__ import annotations

import os

import numpy as np

try:
    import paoflow_rs as _native
except ImportError:  # pragma: no cover - depends on the build
    _native = None


def _disabled() -> bool:
    return os.environ.get('PAOFLOW_ELPHON_DISABLE', '').strip().lower() in (
        '1',
        'true',
        'yes',
        'on',
    )


def available() -> bool:
    """Whether the native electron-phonon kernels can be used."""
    return (
        _native is not None and hasattr(_native, 'elphon_indirect_absorption') and not _disabled()
    )


def indirect_absorption(
    energy_k,
    energy_kq,
    velocity_k,
    velocity_kq,
    coupling,
    omega_ph,
    occ_k,
    occ_kq,
    bose,
    photon,
    etas,
    degauss,
    cutoff,
    pairs_k,
    pairs_i,
    pairs_j,
):
    """Native ``indirect_absorption_kernel`` accumulation (see the Rust kernel for the layout).

    Returns
    -------
    gaussian, lorentzian : ndarray ``(ntemp, neta, 3, nomega)``
        Without the ``weight * 8 pi^2 g_s / omega^2`` prefactor.
    """
    c = np.ascontiguousarray
    ntemp, neta, nomega = bose.shape[0], len(etas), len(photon)
    g, lo = _native.elphon_indirect_absorption(
        c(energy_k, dtype=np.float64),
        c(energy_kq, dtype=np.float64),
        c(velocity_k, dtype=np.complex128),
        c(velocity_kq, dtype=np.complex128),
        c(coupling, dtype=np.complex128),
        c(omega_ph, dtype=np.float64),
        c(occ_k, dtype=np.float64),
        c(occ_kq, dtype=np.float64),
        c(bose, dtype=np.float64),
        c(photon, dtype=np.float64),
        c(np.asarray(etas), dtype=np.float64),
        float(degauss),
        float(cutoff),
        c(pairs_k, dtype=np.int64),
        c(pairs_i, dtype=np.int64),
        c(pairs_j, dtype=np.int64),
    )
    shape = (ntemp, neta, 3, nomega)
    return np.asarray(g).reshape(shape), np.asarray(lo).reshape(shape)
