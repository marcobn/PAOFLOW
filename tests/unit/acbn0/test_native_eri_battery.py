"""Extended randomized parity battery for the Rust ERI backend.

Compares :func:`PAOFLOW.acbn0_native.eri_batch` against the pure-Python
:func:`PAOFLOW.utils.pyints.contr_coulomb` over a mix of s/p/d/f contracted
Gaussians with several primitives each. Skipped when the extension is absent.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from PAOFLOW import acbn0_native
from PAOFLOW.utils.pyints import CGBF, contr_coulomb

pytestmark = pytest.mark.skipif(
    not acbn0_native.available(),
    reason='paoflow_acbn0_rs extension not installed',
)

# A representative Cartesian power tuple for each shell type.
_SHELL_POWERS = {
    's': (0, 0, 0),
    'p': (1, 0, 0),
    'd': (1, 1, 0),
    'f': (1, 1, 1),
}


def _random_cgbf(rng, power, nprim):
    b = CGBF(tuple(rng.uniform(-1.0, 1.0, size=3)))
    b.powers = [tuple(power)] * nprim
    b.pexps = list(rng.uniform(0.2, 4.0, size=nprim))
    b.pcoefs = list(rng.uniform(-0.6, 0.8, size=nprim))
    b.pnorms = list(rng.uniform(0.4, 1.2, size=nprim))
    return b


def _ref(a, b, c, d):
    return contr_coulomb(
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


def test_mixed_shell_battery_matches_pyints():
    rng = np.random.default_rng(20240628)
    # One basis function per shell type, with a varying number of primitives.
    basis = [
        _random_cgbf(rng, _SHELL_POWERS['s'], 3),
        _random_cgbf(rng, _SHELL_POWERS['p'], 2),
        _random_cgbf(rng, _SHELL_POWERS['d'], 4),
        _random_cgbf(rng, _SHELL_POWERS['f'], 2),
    ]
    keys = np.array(list(itertools.product(range(len(basis)), repeat=4)), dtype=np.int64)

    vals = acbn0_native.eri_batch(basis, keys)
    assert vals.shape == (len(keys),)

    max_rel = 0.0
    for key, got in zip(keys, vals):
        ref = _ref(*(basis[i] for i in key))
        denom = max(abs(ref), 1e-300)
        rel = abs(got - ref) / denom if abs(ref) > 1e-300 else abs(got)
        max_rel = max(max_rel, rel)
    assert max_rel < 1e-12, f'max relative error {max_rel:.3e}'


def test_thread_count_is_positive():
    import paoflow_acbn0_rs as rs

    assert rs.thread_count() >= 1


def test_empty_keys_returns_empty():
    rng = np.random.default_rng(1)
    basis = [_random_cgbf(rng, _SHELL_POWERS['d'], 2)]
    vals = acbn0_native.eri_batch(basis, np.empty((0, 4), dtype=np.int64))
    assert vals.shape == (0,)
