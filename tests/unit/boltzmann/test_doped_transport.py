"""``doped_transport``: the chemical potential of the doping, then the tensors there (bulk Si)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

SILICON = Path(__file__).resolve().parents[2] / 'data' / 'silicon'


@pytest.fixture(scope='module')
def silicon(tmp_path_factory):
    import os

    from PAOFLOW import PAOFLOW

    out = tmp_path_factory.mktemp('doped')
    cwd = os.getcwd()
    os.chdir(SILICON)
    try:
        p = PAOFLOW.PAOFLOW(
            savedir='silicon.save', outputdir=str(out), smearing=None, npool=1, verbose=False
        )
        p.read_atomic_proj_QE()
        p.projectability()
        p.pao_hamiltonian()
        p.pao_eigh()
        p.gradient_and_momenta()
    finally:
        os.chdir(cwd)
    return p, out


def test_rows_are_doping_then_transport(silicon):
    p, out = silicon
    window = dict(emin=-14.0, emax=4.0, ne=6000)
    rows = p.doped_transport(-1.0e19, tmin=300.0, tmax=600.0, nt=2, **window)
    assert rows.shape == (2, 9)
    mu = p.doping(tmin=300.0, tmax=600.0, nt=2, doping_conc=-1.0e19, **window)
    np.testing.assert_allclose(rows[:, 1], mu)
    assert np.all(np.diff(rows[:, 1]) < 0)  # n type: mu moves down from the conduction band
    p.transport(tmin=600.0, tmax=600.0, nt=1, emin=mu[1], emax=mu[1], ne=1,
                save_tensors=True, write_to_file=False)  # fmt: skip
    sigma = np.diag(p.data_controller.data_arrays['sigma'][:, :, 0])
    np.testing.assert_allclose(rows[1, 2:5], sigma, rtol=1e-12)
    np.testing.assert_allclose(rows[1, 5], 3.0e2 / sigma.sum(), rtol=1e-12)
    table = np.loadtxt(out / 'doped_transport_n1e+19.dat')
    np.testing.assert_allclose(table, rows, rtol=1e-5)


def test_collinear_magnets_are_refused(silicon):
    p, _ = silicon
    attributes = p.data_controller.data_attributes
    attributes['nspin'], saved = 2, attributes['nspin']
    try:
        with pytest.raises(ValueError, match='one spin channel'):
            p.doped_transport(-1.0e19)
    finally:
        attributes['nspin'] = saved
