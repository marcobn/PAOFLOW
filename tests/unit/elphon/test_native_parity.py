"""Parity of the Rust electron-phonon kernels with the NumPy reference (skipped without the extension)."""

import numpy as np
import pytest

from PAOFLOW import elphon_native
from PAOFLOW.elphon.phonon_assisted_absorption import indirect_absorption_kernel

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not elphon_native.available(), reason='paoflow_rs not built'),
]


def _inputs(seed=0, nk=12, nb=5, nmode=3):
    rng = np.random.default_rng(seed)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    energy_k = rng.uniform(-0.3, 0.3, (nk, nb))
    energy_kq = rng.uniform(-0.3, 0.3, (nk, nb))
    velocity_k = cplx(nk, 3, nb, nb)
    velocity_kq = cplx(nk, 3, nb, nb)
    coupling = 0.1 * cplx(nk, nmode, nb, nb)
    phonon = np.array([0.0005, 0.02, 0.03])  # the first mode is below eps_acoustic
    return {
        'energy_k': energy_k,
        'energy_kq': energy_kq,
        'velocity_k': velocity_k,
        'velocity_kq': velocity_kq,
        'coupling': coupling,
        'phonon_energy': phonon,
        'temperatures': [0.002, 0.005],
        'photon_energy': np.linspace(0.05, 0.6, 23),
        'etas': [0.001, 0.01],
        'degauss': 0.02,
        'fsthick': 0.5,
        'eps_acoustic': 0.001,
        'weight': 0.37,
        'spin_degeneracy': 2.0,
        'fermi_levels': [0.01, -0.02],
    }


def test_indirect_absorption_matches_numpy(monkeypatch):
    kw = _inputs()
    native = indirect_absorption_kernel(**kw)
    monkeypatch.setattr(elphon_native, 'available', lambda: False)
    reference = indirect_absorption_kernel(**kw)
    for a, b in zip(native, reference):
        assert a.shape == b.shape == (2, 2, 3, 23)
        assert np.abs(b).max() > 0
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12 * np.abs(b).max())


def test_indirect_absorption_unsorted_photon_grid(monkeypatch):
    kw = _inputs(seed=3)
    kw['photon_energy'] = kw['photon_energy'][::-1].copy()
    native = indirect_absorption_kernel(**kw)
    monkeypatch.setattr(elphon_native, 'available', lambda: False)
    reference = indirect_absorption_kernel(**kw)
    assert np.allclose(native[0], reference[0], rtol=1e-10, atol=1e-12 * np.abs(reference[0]).max())
