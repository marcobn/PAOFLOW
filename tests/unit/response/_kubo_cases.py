"""Model cases of the linear-response (Kubo-Bastin) routes of PAOFLOW.

Used twice:
- ``python tests/unit/response/_kubo_cases.py`` freezes the outputs of the routes as they were before the
  Kubo-Bastin campaign (14 November 2026) in ``data/kubo_legacy_reference.npz``;
- ``test_kubo_legacy_parity.py`` re-runs the cases and compares every output file with the frozen values.

Models (small grids, seconds each): Kane-Mele (2D, spinful, time-reversal symmetric), Haldane (2D, spinless,
Chern insulator), the two-band Weyl semimetal (3D, spinful, magnetic) and the magnetic Rashba gas of
O. Ly, J. Phys.: Condens. Matter 38, 435802 (2026), Eq. (17) (2D, spinful, magnetic).
"""

from __future__ import annotations

import glob
import os
import sys
from typing import Callable

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REFERENCE = os.path.join(HERE, 'data', 'kubo_legacy_reference.npz')


def rashba_gas(gamma: float = 1.0, delta: float = 0.2, alpha: float = 1.4, alat: float = 6.0):
    """Magnetic Rashba gas of Ly 2026, Eq. (17), as a spinful HoppingModel (one site, spin-orbitals up, down).

    H(k) = -2 gamma (1 + cos k_x + cos k_y) + Delta sigma_z + alpha (sin k_y sigma_x - sin k_x sigma_y).
    """
    from PAOFLOW.models.models import HoppingModel

    sx = np.array([[0, 1], [1, 0]], complex)
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.diag([1.0, -1.0]).astype(complex)
    blocks = {
        (0, 0, 0): -2 * gamma * np.eye(2) + delta * sz,
        (1, 0, 0): -gamma * np.eye(2) - alpha * sy / 2j,
        (-1, 0, 0): -gamma * np.eye(2) + alpha * sy / 2j,
        (0, 1, 0): -gamma * np.eye(2) + alpha * sx / 2j,
        (0, -1, 0): -gamma * np.eye(2) - alpha * sx / 2j,
    }
    hoppings = []
    for cell, block in blocks.items():
        for i in range(2):
            for j in range(2):
                if block[i, j] != 0:
                    hoppings.append((i, j, np.array(cell), complex(block[i, j])))
    lattice = np.diag([1.0, 1.0, 10.0])
    return HoppingModel('rashba_gas', tuple(hoppings), 2, lattice, alat, np.zeros((2, 3)), 1.0,
                        (3, 3, 1), spinful=True)  # fmt: skip


def sp_square_model(alat: float = 5.0):
    """Scalar s + p square lattice (one atom, orbitals s, p_z, p_x, p_y), Slater-Koster up to the
    second neighbours: orbital texture without spin-orbit coupling (orbital Hall effect)."""
    from PAOFLOW.models.models import HoppingModel

    onsite = np.diag([1.5, -0.6, 0.0, 0.0]).astype(complex)
    shells = [(1.0, dict(ss=-0.8, sp=0.9, pps=1.2, ppp=-0.3)), (np.sqrt(2.0), dict(ss=-0.2, sp=0.3, pps=0.4, ppp=-0.1))]  # fmt: skip
    hoppings = [(i, i, np.zeros(3, int), complex(onsite[i, i])) for i in range(4)]
    for cell in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        dist = np.hypot(*cell)
        v = next(p for d, p in shells if abs(d - dist) < 1e-9)
        l, m = cell[0] / dist, cell[1] / dist
        block = np.zeros((4, 4))
        block[0, 0] = v['ss']
        block[0, 2], block[0, 3] = l * v['sp'], m * v['sp']
        block[2, 0], block[3, 0] = -l * v['sp'], -m * v['sp']
        block[1, 1] = v['ppp']
        block[2, 2] = l * l * v['pps'] + (1 - l * l) * v['ppp']
        block[3, 3] = m * m * v['pps'] + (1 - m * m) * v['ppp']
        block[2, 3] = block[3, 2] = l * m * (v['pps'] - v['ppp'])
        R = np.array([cell[0], cell[1], 0])
        hoppings += [(i, j, R, complex(block[i, j])) for i in range(4) for j in range(4) if block[i, j] != 0]  # fmt: skip
    lattice = np.diag([1.0, 1.0, 10.0])
    return HoppingModel(
        'sp_square', tuple(hoppings), 4, lattice, alat, np.zeros((4, 3)), 2.0, (3, 3, 1)
    )


def model(name: str):
    from PAOFLOW.models import models

    if name == 'sp_square':
        return sp_square_model()

    if name == 'kane_mele':
        return models.kane_mele_model()
    if name == 'haldane':
        return models.honeycomb_model(t2=0.1, phi=np.pi / 2)
    if name == 'weyl':
        return models.weyl_model(alat=6.0)
    if name == 'rashba':
        return rashba_gas()
    raise KeyError(name)


GRIDS = {'kane_mele': (24, 24, 1), 'haldane': (24, 24, 1), 'weyl': (10, 10, 10), 'rashba': (32, 32, 1),
         'sp_square': (24, 24, 1)}  # fmt: skip
WINDOWS = {'kane_mele': (-3.0, 3.0), 'haldane': (-3.0, 3.0), 'weyl': (-1.0, 1.0), 'rashba': (-3.0, -1.0),
           'sp_square': (-2.0, 2.0)}  # fmt: skip
NE = 21


def prepared(name: str, outputdir: str):
    """A PAOFLOW instance on a model, ready for the response routes."""
    from PAOFLOW import PAOFLOW

    p = PAOFLOW.PAOFLOW(model=model(name), outputdir=outputdir, smearing='gauss', verbose=False)
    if name == 'sp_square':
        # orbital labels for the scalar L operator (PAOFLOW.orbital_operator)
        arry = p.data_controller.data_arrays
        arry['atoms'], arry['shells'] = ['X'], {'X': [0, 1]}
    n1, n2, n3 = GRIDS[name]
    p.interpolated_hamiltonian(nfft1=n1, nfft2=n2, nfft3=n3)
    p.pao_eigh()
    p.gradient_and_momenta()
    p.adaptive_smearing()
    return p


def _lr(response: str, **flags) -> Callable:
    def run(p, name):
        emin, emax = WINDOWS[name]
        p.linear_response(response=response, eminH=emin, emaxH=emax, esize=NE, **flags)

    return run


def _hall(method: str, **kwargs) -> Callable:
    def run(p, name):
        emin, emax = WINDOWS[name]
        getattr(p, method)(emin=emin, emax=emax, ne=NE, **kwargs)

    return run


SHC = dict(s_tensor=[[2, 0, 1]], a_tensor=[[0, 1]])  # linear_response: (spin, current, field)
REE = dict(s_tensor=[[2, 0, 1]], a_tensor=[[1, 0], [0, 1]])  # (spin, field)
AXY = dict(s_tensor=[[2, 0, 1]], a_tensor=[[0, 1], [0, 0]])

#: name -> (model, route). Spin-dependent routes run on spinful models only.
CASES: dict[str, tuple[str, Callable]] = {}
for _m in ('kane_mele', 'rashba'):
    CASES.update({
        f'{_m}/spin_Hall': (_m, _hall('spin_Hall', s_tensor=[[0, 1, 2]], do_ac=True)),
        f'{_m}/shc_eq3': (_m, _lr('shc', **SHC)),
        f'{_m}/shc_eq1_todd': (_m, _lr('shc', t_odd=True, **SHC)),
        f'{_m}/ree_eq1': (_m, _lr('ree', **REE)),
        f'{_m}/ree_eq3_todd': (_m, _lr('ree', t_odd=True, **REE)),
        f'{_m}/ree_eq2_todd': (_m, _lr('ree', t_odd=True, full_chi2=True, **REE)),
        f'{_m}/ree_eq4': (_m, _lr('ree', intraband=True, **REE)),
        f'{_m}/ree_eq5': (_m, _lr('ree', interband=True, **REE)),
        f'{_m}/cond_eq1': (_m, _lr('cond', **AXY)),
        f'{_m}/cond_eq4': (_m, _lr('cond', intraband=True, **AXY)),
        f'{_m}/cond_eq5': (_m, _lr('cond', interband=True, **AXY)),
        f'{_m}/ahc_eq3': (_m, _lr('ahc', **AXY)),
        f'{_m}/ahc_eq2': (_m, _lr('ahc', full_chi2=True, **AXY)),
    })  # fmt: skip
for _m in ('haldane', 'weyl'):
    CASES.update({
        f'{_m}/cond_eq1': (_m, _lr('cond', **AXY)),
        f'{_m}/ahc_eq3': (_m, _lr('ahc', **AXY)),
        f'{_m}/ahc_eq2': (_m, _lr('ahc', full_chi2=True, **AXY)),
    })  # fmt: skip
CASES['weyl/anomalous_Hall'] = ('weyl', _hall('anomalous_Hall', a_tensor=[[0, 1]], do_ac=True))
CASES['rashba/anomalous_Hall'] = ('rashba', _hall('anomalous_Hall', a_tensor=[[0, 1]], do_ac=True))


def run_case(case: str, workdir: str) -> dict[str, np.ndarray]:
    """Run one case in ``workdir``; return every ``.dat`` output as an array (file name -> data)."""
    name, route = CASES[case]
    outdir = os.path.join(workdir, case.replace('/', '__'))
    p = prepared(name, outdir)
    route(p, name)
    p.finish_execution()
    out = {}
    for path in sorted(glob.glob(os.path.join(outdir, '*.dat'))):
        out[os.path.basename(path)] = np.loadtxt(path, comments='#')
    return out


def freeze(workdir: str) -> None:
    """Write the reference file from the current code."""
    arrays = {}
    for case in CASES:
        for fname, data in run_case(case, workdir).items():
            arrays[f'{case}|{fname}'] = data
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    np.savez_compressed(REFERENCE, **arrays)
    print('wrote', REFERENCE, len(arrays), 'arrays from', len(CASES), 'cases')


if __name__ == '__main__':
    import tempfile

    freeze(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp())
