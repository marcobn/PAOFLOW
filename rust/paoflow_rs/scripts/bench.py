#!/usr/bin/env python
"""Benchmark the Rust ERI backend against the pure-Python pyints kernel.

Builds a realistic on-site Hubbard d-shell (5 contracted Gaussians, several
primitives each), enumerates the unique ``(ab|cd)`` keys under the 8-fold
permutation symmetry used by ``ACBN0_Hartree.hartree_energy``, and times the
batched native ``eri_batch`` against per-key ``pyints.contr_coulomb``.

Run:
    python rust/paoflow_rs/scripts/bench.py [--nprim N] [--repeat R]
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'src'))

from PAOFLOW import acbn0_native  # noqa: E402
from PAOFLOW.utils import pyints  # noqa: E402

_D_POWERS = [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1)]


def make_d_shell(nprim):
    rng = np.random.default_rng(0)
    exps = list(rng.uniform(0.2, 6.0, size=nprim))
    coefs = list(rng.uniform(0.1, 0.8, size=nprim))
    norms = list(rng.uniform(0.4, 1.1, size=nprim))
    basis = []
    for p in _D_POWERS:
        b = pyints.CGBF((0.05, -0.1, 0.0))
        b.powers = [p] * nprim
        b.pexps, b.pcoefs, b.pnorms = exps, coefs, norms
        basis.append(b)
    return basis


def unique_keys(n):
    keys = []
    for a in range(n):
        for b in range(a, n):
            for c in range(n):
                for d in range(c, n):
                    if (a, b) <= (c, d):
                        keys.append((a, b, c, d))
    return np.array(keys, dtype=np.int64)


def ref_value(basis, key):
    a, b, c, d = (basis[i] for i in key)
    return pyints.contr_coulomb(
        a.pexps,
        a.pcoefs,
        a.pnorms,
        a.origin,
        a.powers,
        b.pexps,
        b.pcoefs,
        b.pnorms,
        b.origin,
        b.powers,
        c.pexps,
        c.pcoefs,
        c.pnorms,
        c.origin,
        c.powers,
        d.pexps,
        d.pcoefs,
        d.pnorms,
        d.origin,
        d.powers,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nprim', type=int, default=8, help='primitives per CGBF')
    ap.add_argument('--repeat', type=int, default=3, help='timing repeats')
    args = ap.parse_args()

    if not acbn0_native.available():
        print('paoflow_rs not installed; cannot benchmark the native path.')
        return

    import paoflow_rs as rs

    basis = make_d_shell(args.nprim)
    keys = unique_keys(len(basis))
    print(f'd-shell: {len(basis)} CGBFs x {args.nprim} primitives, {len(keys)} unique keys')
    print(f'native threads: {rs.thread_count()}')

    # Correctness first.
    vals = acbn0_native.eri_batch(basis, keys)
    max_rel = 0.0
    for k, v in zip(keys, vals):
        r = ref_value(basis, k)
        denom = max(abs(r), 1e-300)
        max_rel = max(max_rel, abs(v - r) / denom if abs(r) > 1e-300 else abs(v))
    print(f'max relative error vs pyints: {max_rel:.3e}')

    def time_native():
        t = time.perf_counter()
        acbn0_native.eri_batch(basis, keys)
        return time.perf_counter() - t

    def time_python():
        t = time.perf_counter()
        for k in keys:
            ref_value(basis, k)
        return time.perf_counter() - t

    t_native = min(time_native() for _ in range(args.repeat))
    t_python = min(time_python() for _ in range(args.repeat))
    print(f'pyints : {t_python * 1e3:8.2f} ms')
    print(f'native : {t_native * 1e3:8.2f} ms')
    print(f'speedup: {t_python / t_native:6.1f}x')


if __name__ == '__main__':
    main()
