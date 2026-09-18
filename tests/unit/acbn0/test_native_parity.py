"""Parity tests: the optional Rust ERI backend must reproduce the pure-Python
``pyints`` kernel inside the real ACBN0 / eACBN0 Hartree drivers.

Skipped automatically when the ``paoflow_rs`` extension is not installed.
"""

from __future__ import annotations

import pickle
from os.path import join

import numpy as np
import pytest
from mpi4py import MPI

from PAOFLOW import acbn0_native
from PAOFLOW.ACBN0 import ACBN0_Hartree, eACBN0_Hartree
from PAOFLOW.utils.pyints import CGBF

pytestmark = pytest.mark.skipif(
    not acbn0_native.available(),
    reason='paoflow_rs extension not installed',
)


def _cgbf(origin, powers, exps, coefs, norms):
    b = CGBF(origin)
    b.powers = [tuple(p) for p in powers]
    b.pexps = list(exps)
    b.pcoefs = list(coefs)
    b.pnorms = list(norms)
    return b


def _d_basis():
    """Five d-like contracted Gaussians on a single centre."""
    dpow = [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1]]
    exps = [2.0, 0.6, 0.2]
    coefs = [0.5, 0.4, 0.1]
    norms = [0.95, 0.7, 0.5]
    return [_cgbf([0.1, -0.2, 0.05], [p, p, p], exps, coefs, norms) for p in dpow]


def _p_basis(origin):
    ppow = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    exps = [2.5, 0.8]
    coefs = [0.6, 0.4]
    norms = [1.0, 0.7]
    return [_cgbf(origin, [p, p], exps, coefs, norms) for p in ppow]


def _make_kernel(cls, data):
    obj = cls.__new__(cls)
    obj.comm = MPI.COMM_WORLD
    obj.rank = obj.comm.Get_rank()
    obj.size = obj.comm.Get_size()
    obj.data = data
    return obj


def test_hartree_energy_native_matches_pyints(tmp_path, monkeypatch):
    basis = _d_basis()
    n = len(basis)
    rng = np.random.default_rng(0)
    DR_up = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    DR_dn = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    data = {
        'DR_up': DR_up,
        'DR_dn': DR_dn,
        'basis': basis,
        'basis_2e': list(range(n)),
    }

    # Native path.
    _make_kernel(ACBN0_Hartree, data).hartree_energy(str(tmp_path))
    with open(join(tmp_path, 'tmp_uj.pkl'), 'rb') as fh:
        native = pickle.load(fh)

    # Forced pure-Python fallback.
    monkeypatch.setattr(acbn0_native, 'available', lambda: False)
    _make_kernel(ACBN0_Hartree, data).hartree_energy(str(tmp_path))
    with open(join(tmp_path, 'tmp_uj.pkl'), 'rb') as fh:
        ref = pickle.load(fh)

    assert native['U'] == pytest.approx(ref['U'], rel=1e-12, abs=1e-12)
    assert native['J'] == pytest.approx(ref['J'], rel=1e-12, abs=1e-12)


def test_intersite_energy_native_matches_pyints(tmp_path, monkeypatch):
    gauss_I = _p_basis([0.0, 0.0, 0.0])
    gauss_J = _p_basis([1.4, 0.2, -0.3])
    n_I, n_J = len(gauss_I), len(gauss_J)
    rng = np.random.default_rng(1)

    def cmat(a, b):
        return (rng.standard_normal((a, b)) + 1j * rng.standard_normal((a, b))) * 0.1

    data = {
        'gauss_I': gauss_I,
        'gauss_J': gauss_J,
        'P_II_up': cmat(n_I, n_I),
        'P_II_dn': cmat(n_I, n_I),
        'P_JJ_up': cmat(n_J, n_J),
        'P_JJ_dn': cmat(n_J, n_J),
        'P_IJ_up': cmat(n_I, n_J),
        'P_IJ_dn': cmat(n_I, n_J),
        'P_JI_up': cmat(n_J, n_I),
        'P_JI_dn': cmat(n_J, n_I),
    }

    _make_kernel(eACBN0_Hartree, data).intersite_energy(str(tmp_path))
    with open(join(tmp_path, 'tmp_v.pkl'), 'rb') as fh:
        native = pickle.load(fh)

    monkeypatch.setattr(acbn0_native, 'available', lambda: False)
    _make_kernel(eACBN0_Hartree, data).intersite_energy(str(tmp_path))
    with open(join(tmp_path, 'tmp_v.pkl'), 'rb') as fh:
        ref = pickle.load(fh)

    assert native['num'] == pytest.approx(ref['num'], rel=1e-12, abs=1e-12)
