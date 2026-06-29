#!/usr/bin/env python
"""Generate golden reference values from the pure-Python pyints kernel.

These values are consumed by the Rust crate's unit/integration tests to
guarantee bit-for-bit (to <1e-12 relative) parity between the Rust port and
the reference implementation in ``src/PAOFLOW/utils/pyints.py``.

Run:
    python rust/paoflow_rs/scripts/gen_golden.py

Writes ``rust/paoflow_rs/tests/golden.json``.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
SRC = os.path.join(REPO_ROOT, 'src')
sys.path.insert(0, SRC)

from PAOFLOW.utils import pyints  # noqa: E402


def gen_gammln():
    xs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 5.5, 10.0, 0.25, 7.3]
    return [{'x': x, 'val': float(pyints.gammln(x))} for x in xs]


def gen_fgamma():
    out = []
    for m in range(0, 9):
        for x in [1e-8, 1e-4, 0.1, 0.5, 1.0, 2.5, 5.0, 12.0, 30.0, 75.0]:
            out.append({'m': m, 'x': x, 'val': float(pyints.Fgamma(m, x))})
    return out


def gen_binomial_prefactor():
    out = []
    for s in range(0, 5):
        for ia in range(0, 3):
            for ib in range(0, 3):
                for xpa, xpb in [(0.1, -0.2), (0.7, 0.3), (-0.5, 0.9)]:
                    out.append(
                        {
                            's': s,
                            'ia': ia,
                            'ib': ib,
                            'xpa': xpa,
                            'xpb': xpb,
                            'val': float(pyints.binomial_prefactor(s, ia, ib, xpa, xpb)),
                        }
                    )
    return out


def gen_b_array():
    out = []
    # (l1,l2,l3,l4, p,a,b, q,c,d, g1,g2,delta)
    cases = [
        (0, 0, 0, 0, 0.1, 0.0, 0.2, 0.4, 0.3, 0.5, 1.2, 1.4, 0.3),
        (1, 0, 0, 0, 0.1, 0.0, 0.2, 0.4, 0.3, 0.5, 1.2, 1.4, 0.3),
        (1, 1, 1, 1, 0.15, -0.1, 0.25, 0.45, 0.35, 0.55, 1.3, 1.5, 0.28),
        (2, 0, 1, 1, 0.2, 0.1, 0.3, 0.5, 0.2, 0.6, 1.6, 1.1, 0.33),
        (2, 2, 0, 0, 0.05, -0.2, 0.1, 0.3, 0.4, 0.5, 2.0, 1.8, 0.22),
    ]
    for c in cases:
        val = pyints.B_array(*c)
        out.append({'args': list(c), 'val': [float(v) for v in val]})
    return out


def gen_coulomb_repulsion():
    out = []
    # Each primitive: (xyz, norm, lmn, alpha)
    prims = [
        ([0.0, 0.0, 0.0], 1.0, [0, 0, 0], 1.2),
        ([0.5, 0.0, 0.0], 1.0, [1, 0, 0], 0.8),
        ([0.0, 0.6, 0.0], 1.0, [0, 1, 0], 1.5),
        ([0.0, 0.0, 0.7], 1.0, [0, 0, 1], 0.6),
        ([0.3, 0.3, 0.3], 1.0, [1, 1, 0], 1.1),
        ([0.2, -0.4, 0.1], 1.0, [2, 0, 0], 0.9),
        ([-0.3, 0.2, 0.5], 1.0, [0, 1, 1], 1.3),
    ]
    quads = [
        (0, 1, 2, 3),
        (0, 0, 0, 0),
        (4, 1, 2, 3),
        (5, 0, 6, 4),
        (1, 2, 3, 4),
        (6, 6, 5, 5),
        (4, 4, 4, 4),
    ]
    for a, b, c, d in quads:
        pa, pb, pc, pd = prims[a], prims[b], prims[c], prims[d]
        val = pyints.coulomb_repulsion(
            pa[0],
            pa[1],
            pa[2],
            pa[3],
            pb[0],
            pb[1],
            pb[2],
            pb[3],
            pc[0],
            pc[1],
            pc[2],
            pc[3],
            pd[0],
            pd[1],
            pd[2],
            pd[3],
        )
        out.append(
            {
                'a': pa,
                'b': pb,
                'c': pc,
                'd': pd,
                'val': float(val),
            }
        )
    return out


def gen_contr_coulomb():
    out = []
    # A contracted s and a contracted p, each with 3 primitives.
    s_exps = [3.0, 1.0, 0.3]
    s_coefs = [0.15, 0.5, 0.4]
    s_norms = [1.1, 0.9, 0.7]
    s_xyz = [0.0, 0.0, 0.0]
    s_pow = [[0, 0, 0]] * 3

    p_exps = [2.5, 0.8, 0.25]
    p_coefs = [0.2, 0.55, 0.35]
    p_norms = [1.0, 0.85, 0.65]
    p_xyz = [0.7, 0.1, -0.2]
    p_pow = [[1, 0, 0]] * 3

    d_exps = [2.0, 0.6]
    d_coefs = [0.6, 0.4]
    d_norms = [0.95, 0.7]
    d_xyz = [-0.3, 0.4, 0.5]
    d_pow = [[1, 1, 0]] * 2

    shells = {
        's': (s_exps, s_coefs, s_norms, s_xyz, s_pow),
        'p': (p_exps, p_coefs, p_norms, p_xyz, p_pow),
        'd': (d_exps, d_coefs, d_norms, d_xyz, d_pow),
    }
    combos = [
        ('s', 's', 's', 's'),
        ('s', 'p', 's', 'p'),
        ('p', 'p', 'p', 'p'),
        ('d', 's', 'd', 's'),
        ('d', 'p', 's', 'd'),
    ]
    for ka, kb, kc, kd in combos:
        a, b, c, d = shells[ka], shells[kb], shells[kc], shells[kd]
        val = pyints.contr_coulomb(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            c[0],
            c[1],
            c[2],
            c[3],
            c[4],
            d[0],
            d[1],
            d[2],
            d[3],
            d[4],
        )
        out.append(
            {
                'shells': [ka, kb, kc, kd],
                'a': {'exps': a[0], 'coefs': a[1], 'norms': a[2], 'xyz': a[3], 'pow': a[4]},
                'b': {'exps': b[0], 'coefs': b[1], 'norms': b[2], 'xyz': b[3], 'pow': b[4]},
                'c': {'exps': c[0], 'coefs': c[1], 'norms': c[2], 'xyz': c[3], 'pow': c[4]},
                'd': {'exps': d[0], 'coefs': d[1], 'norms': d[2], 'xyz': d[3], 'pow': d[4]},
                'val': float(val),
            }
        )
    return out


def main():
    data = {
        'gammln': gen_gammln(),
        'fgamma': gen_fgamma(),
        'binomial_prefactor': gen_binomial_prefactor(),
        'b_array': gen_b_array(),
        'coulomb_repulsion': gen_coulomb_repulsion(),
        'contr_coulomb': gen_contr_coulomb(),
    }
    out_path = os.path.join(HERE, '..', 'tests', 'golden.json')
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f'wrote {out_path}')
    for k, v in data.items():
        print(f'  {k}: {len(v)} cases')


if __name__ == '__main__':
    main()
