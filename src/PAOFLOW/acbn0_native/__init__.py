"""Optional Rust backend bridge for the ACBN0 / eACBN0 Coulomb integrals.

This package opportunistically imports the compiled ``paoflow_rs``
extension (built from ``rust/paoflow_rs``) and exposes a small, NumPy
friendly API that mirrors the per-call :meth:`PAOFLOW.ACBN0._HartreeKernel.coulomb`
semantics in *batched* form.

If the extension is not installed, :func:`available` returns ``False`` and the
callers fall back to the pure-Python :mod:`PAOFLOW.utils.pyints` kernel, so the
backend is a fully optional dependency.

The basis crosses the FFI boundary as flat CSR-style arrays:

- ``origins``       — ``(nbasis, 3)`` float64 centre coordinates.
- ``prim_offsets``  — ``(nbasis + 1,)`` int64 CSR row pointers.
- ``exps``/``coefs``/``norms`` — ``(nprim_total,)`` float64 per-primitive data.
- ``powers``        — ``(nprim_total, 3)`` int64 Cartesian angular momenta.

and the integral request as ``keys`` — ``(nkeys, 4)`` int64 index tuples.

Threads: the native kernels parallelise over keys with ``rayon`` and release the
GIL. Set ``PAOFLOW_RS_THREADS`` (legacy alias ``PAOFLOW_ACBN0_THREADS``) to a
positive integer to cap intra-rank threads (useful to avoid oversubscription
with many MPI ranks per node); otherwise the global ``rayon`` pool is used
(honours ``RAYON_NUM_THREADS``).
"""

import os

import numpy as np

try:
    import paoflow_rs as _native
except ImportError:  # pragma: no cover - exercised only when extension absent
    try:
        import paoflow_acbn0_rs as _native  # legacy module name
    except ImportError:
        _native = None


__all__ = ['available', 'flatten_basis', 'eri_batch', 'eri_batch_2c']


def _disabled():
    """Return ``True`` when the backend is force-disabled via the environment.

    Setting ``PAOFLOW_ACBN0_DISABLE`` to a truthy value (``1``/``true``/``yes``/
    ``on``) forces the pure-Python ``pyints`` fallback even when the compiled
    extension is installed. Useful for A/B timing a real ACBN0 run without
    uninstalling the wheel.
    """
    return os.environ.get('PAOFLOW_ACBN0_DISABLE', '').strip().lower() in {
        '1',
        'true',
        'yes',
        'on',
    }


def available():
    """Return ``True`` when the compiled Rust backend can be used."""
    return _native is not None and not _disabled()


def flatten_basis(basis):
    """Flatten a list of contracted Gaussians into CSR-style arrays.

    Parameters
    ----------
    basis : sequence of :class:`PAOFLOW.utils.pyints.CGBF`
        Contracted Gaussian basis functions. Each must expose ``origin``
        (length-3), and the per-primitive lists ``pexps``, ``pcoefs``,
        ``pnorms`` and ``powers`` (each entry a length-3 tuple).

    Returns
    -------
    tuple of numpy.ndarray
        ``(origins, prim_offsets, exps, coefs, norms, powers)`` with dtypes
        ``float64`` (origins/exps/coefs/norms), ``int64`` (prim_offsets) and
        ``int64`` (powers).
    """
    nbasis = len(basis)
    origins = np.empty((nbasis, 3), dtype=np.float64)
    prim_offsets = np.empty(nbasis + 1, dtype=np.int64)
    exps, coefs, norms, powers = [], [], [], []

    prim_offsets[0] = 0
    for b, bf in enumerate(basis):
        origins[b] = bf.origin
        nprim = len(bf.pexps)
        exps.extend(bf.pexps)
        coefs.extend(bf.pcoefs)
        norms.extend(bf.pnorms)
        powers.extend([tuple(p) for p in bf.powers])
        prim_offsets[b + 1] = prim_offsets[b] + nprim

    return (
        origins,
        prim_offsets,
        np.asarray(exps, dtype=np.float64),
        np.asarray(coefs, dtype=np.float64),
        np.asarray(norms, dtype=np.float64),
        np.asarray(powers, dtype=np.int64).reshape(-1, 3),
    )


def eri_batch(basis, keys):
    """Batched on-site ``(ab|cd)`` integrals over a single basis.

    Parameters
    ----------
    basis : sequence of CGBF
        The (active) contracted Gaussian basis.
    keys : array_like, shape ``(nkeys, 4)``
        Integer index tuples ``(a, b, c, d)`` into ``basis``.

    Returns
    -------
    numpy.ndarray, shape ``(nkeys,)``
        One contracted Coulomb integral per key.
    """
    if _native is None:
        raise RuntimeError('paoflow_rs extension is not available')
    origins, prim_offsets, exps, coefs, norms, powers = flatten_basis(basis)
    keys = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 4)
    return _native.acbn0_eri_batch(origins, prim_offsets, exps, coefs, norms, powers, keys)


def eri_batch_2c(basis_i, basis_j, keys):
    """Batched intersite ``(ik|jl)`` integrals over two bases (eACBN0).

    Parameters
    ----------
    basis_i, basis_j : sequence of CGBF
        Contracted Gaussian bases on atoms I and J respectively.
    keys : array_like, shape ``(nkeys, 4)``
        Integer index tuples ``(i, k, j, l)`` where ``i, k`` index
        ``basis_i`` and ``j, l`` index ``basis_j``.

    Returns
    -------
    numpy.ndarray, shape ``(nkeys,)``
        One contracted Coulomb integral per key.
    """
    if _native is None:
        raise RuntimeError('paoflow_rs extension is not available')
    oi, po_i, ei, ci, ni, pw_i = flatten_basis(basis_i)
    oj, po_j, ej, cj, nj, pw_j = flatten_basis(basis_j)
    keys = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 4)
    return _native.acbn0_eri_batch_2c(
        oi,
        po_i,
        ei,
        ci,
        ni,
        pw_i,
        oj,
        po_j,
        ej,
        cj,
        nj,
        pw_j,
        keys,
    )
