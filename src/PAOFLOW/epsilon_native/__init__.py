"""Optional Rust backend bridge for the dielectric / JDOS response loops.

This package opportunistically imports the compiled ``paoflow_rs`` extension
and exposes a small, NumPy-friendly API mirroring the vectorised
:func:`PAOFLOW.response.do_epsilon.eps_loop` and ``jdos_loop`` inner loops in
*batched* form (whole rank-local k-slice in one call, ``rayon``-parallel, GIL
released).

If the extension is not installed, :func:`available` returns ``False`` and the
callers fall back to the vectorised NumPy path, so the backend is fully optional.

Threads: set ``PAOFLOW_RS_THREADS`` to cap intra-rank threads. Force the
NumPy fallback (even when installed) with a truthy ``PAOFLOW_EPSILON_DISABLE``.
"""

import os

import numpy as np

try:
    import paoflow_rs as _native
except ImportError:  # pragma: no cover - exercised only when extension absent
    _native = None


__all__ = ['available', 'eps_loop', 'jdos_loop']


def _disabled():
    """Return ``True`` when the backend is force-disabled via the environment."""
    return os.environ.get('PAOFLOW_EPSILON_DISABLE', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def available():
    """Return ``True`` when the compiled Rust backend can be used."""
    return _native is not None and not _disabled()


def eps_loop(ek, fn_occ, pksp2, ene, intersmear, th0, th1, spin_factor, deltakp2, eta_floor, fnf):
    """Interband dielectric inner loop over the local k-slice.

    Parameters mirror the vectorised reference; ``pksp2`` is the precomputed
    ``Re(P.T * Q)`` of shape ``(nk, nbnd, nbnd)``. ``deltakp2``/``fnf`` may be
    ``None``. Returns ``(epsi, epsr, drude_weight)``.
    """
    if _native is None:
        raise RuntimeError('paoflow_rs extension is not available')
    ek = np.ascontiguousarray(ek, dtype=np.float64)
    fn_occ = np.ascontiguousarray(fn_occ, dtype=np.float64)
    pksp2 = np.ascontiguousarray(pksp2, dtype=np.float64)
    ene = np.ascontiguousarray(ene, dtype=np.float64)
    dk = None if deltakp2 is None else np.ascontiguousarray(deltakp2, dtype=np.float64)
    fnf_c = None if fnf is None else np.ascontiguousarray(fnf, dtype=np.float64)
    return _native.epsilon_eps_loop(
        ek,
        fn_occ,
        pksp2,
        ene,
        float(intersmear),
        float(th0),
        float(th1),
        float(spin_factor),
        dk,
        float(eta_floor),
        fnf_c,
    )


def jdos_loop(ek, fn_occ, kweights, ene, intersmear, smeartype):
    """JDOS inner loop over the local k-slice.

    ``smeartype`` is ``'gauss'`` or ``'lorentz'``. Returns ``(jdos, count)``
    partials for MPI reduction by the caller.
    """
    if _native is None:
        raise RuntimeError('paoflow_rs extension is not available')
    code = 0 if smeartype == 'gauss' else 1
    ek = np.ascontiguousarray(ek, dtype=np.float64)
    fn_occ = np.ascontiguousarray(fn_occ, dtype=np.float64)
    kweights = np.ascontiguousarray(kweights, dtype=np.float64)
    ene = np.ascontiguousarray(ene, dtype=np.float64)
    return _native.epsilon_jdos_loop(ek, fn_occ, kweights, ene, float(intersmear), code)
