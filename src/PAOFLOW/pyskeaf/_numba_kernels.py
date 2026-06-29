"""Optional Numba-accelerated kernels.

This module is imported unconditionally, but Numba itself is treated as an
**optional dependency**: if ``import numba`` fails, ``HAS_NUMBA`` is set to
``False`` and :func:`lagrange4_eval` falls back to a pure-NumPy implementation
that is byte-for-byte equivalent to the previous einsum path.

Profiling on the cylinder test (5 angles, ``numint=30``) showed that
``slice_ops._interpolate_per_point`` accounted for ~67 % of total wall-clock
time, dominated by the 4×4×4 fancy-index gather and trailing einsum.  A
hand-written ``@njit`` kernel that fuses the gather and the contraction is
~25× faster on a per-slice basis (0.32 ms vs 8 ms for ``M ≈ 1.4·10⁴``,
``nx=ny=nz=30``), turning a ~1 s/angle interpolation into ~40 ms/angle.

The Numba kernel is **serial** — at typical SKEAF problem sizes the per-slice
workload is too small for ``prange`` parallelism to pay back the threading
overhead.  Coarser-grained parallelism (over angles, via :mod:`joblib`)
multiplies on top of the JIT speedup.

To install Numba support:: ``pip install pyskeaf[fast]`` (or ``pip install numba``).
"""

from __future__ import annotations

import numpy as np

try:
    if 'long' not in np.__dict__:
        raise ImportError('installed Numba is incompatible with this NumPy')
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover — exercised only when Numba absent/broken
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        """No-op fallback for ``@numba.njit`` when Numba cannot be used."""
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator


# ---------------------------------------------------------------------------
# Numba kernel — fused gather + 4×4×4 Lagrange contraction.
# ---------------------------------------------------------------------------


@njit(cache=True)
def _lagrange4_eval_numba(energies, ix, iy, iz, wx, wy, wz):
    """Evaluate ``Σ_{abc} wx·wy·wz · E[ix,iy,iz]`` for each of the M points.

    Uses a **nested factored** summation order — sum over c first, then b, then a:
    ``s = Σ_a wxa · (Σ_b wyb · (Σ_c wzc · E[ixa,iyb,izc]))``.
    This matches the per-axis contraction order of the previous einsum-based
    implementation, so on a constant field the result is exactly E₀ (the
    weight sums collapse cleanly), avoiding spurious iso-contours from
    accumulated rounding noise on a flat band.

    Parameters
    ----------
    energies : ndarray, shape (nx, ny, nz)
        Source field.
    ix, iy, iz : ndarray, shape (M, 4), intp
        4-point stencil indices along each axis (already periodic-wrapped).
    wx, wy, wz : ndarray, shape (M, 4), float64
        Lagrange weights along each axis (rows sum to 1).

    Returns
    -------
    out : ndarray, shape (M,), float64
    """
    M = ix.shape[0]
    out = np.empty(M, dtype=np.float64)
    for m in range(M):
        s = 0.0
        for a in range(4):
            ixa = ix[m, a]
            sb = 0.0
            for b in range(4):
                iyb = iy[m, b]
                sc = 0.0
                for c in range(4):
                    sc += wz[m, c] * energies[ixa, iyb, iz[m, c]]
                sb += wy[m, b] * sc
            s += wx[m, a] * sb
        out[m] = s
    return out


# ---------------------------------------------------------------------------
# Pure-NumPy fallback — semantically identical, used when Numba absent.
# ---------------------------------------------------------------------------


def _lagrange4_eval_numpy(energies, ix, iy, iz, wx, wy, wz):
    """Pure-NumPy reference implementation; same signature/semantics as the JIT version."""
    sub = energies[
        ix[:, :, None, None],
        iy[:, None, :, None],
        iz[:, None, None, :],
    ]
    return np.einsum('ma,mb,mc,mabc->m', wx, wy, wz, sub, optimize=True)


def lagrange4_eval(energies, ix, iy, iz, wx, wy, wz):
    """Public dispatcher — uses the JIT kernel when Numba is installed.

    Inputs and outputs match :func:`_lagrange4_eval_numba` /
    :func:`_lagrange4_eval_numpy` exactly.  Arrays are coerced to the dtypes
    Numba expects (``intp`` indices, ``float64`` weights/energies); the cost
    of the coerce is amortised by the kernel speedup.
    """
    if HAS_NUMBA:
        # Numba dispatch is sensitive to dtype/contiguity; normalise once.
        energies = np.ascontiguousarray(energies, dtype=np.float64)
        ix = np.ascontiguousarray(ix, dtype=np.intp)
        iy = np.ascontiguousarray(iy, dtype=np.intp)
        iz = np.ascontiguousarray(iz, dtype=np.intp)
        wx = np.ascontiguousarray(wx, dtype=np.float64)
        wy = np.ascontiguousarray(wy, dtype=np.float64)
        wz = np.ascontiguousarray(wz, dtype=np.float64)
        return _lagrange4_eval_numba(energies, ix, iy, iz, wx, wy, wz)
    return _lagrange4_eval_numpy(energies, ix, iy, iz, wx, wy, wz)
