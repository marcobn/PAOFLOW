"""PythTB models in PAOFLOW (build_from_pythTB): conventions consistent with the optical responses."""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW import PAOFLOW
from PAOFLOW.hamiltonian.bloch_derivatives import (
    bloch_terms_from_HRs,
    hamiltonian_block,
    orbital_positions_from_data,
)
from PAOFLOW.models import models
from PAOFLOW.response import nonlinear_optics as nlo

pytest.importorskip('pythtb', minversion='2.0')

pytestmark = pytest.mark.unit

PYTHTB_MODELS = {
    'graphene': lambda: models.graphene_pythtb(0.6, -1.0),
    'haldane': lambda: models.haldane_pythtb(0.3, -1.0, 0.15, np.pi / 2),
    'kane_mele': lambda: models.kane_mele_pythtb(0.4, -1.0, 0.06, 0.05),
    'fu_kane_mele': lambda: models.fu_kane_mele_pythtb(1.0, 0.1, [0.4, 0.0, 0.0, 0.0]),
    'checkerboard': lambda: models.checkerboard_pythtb(0.3, 0.2),
    'ssh': lambda: models.ssh_pythtb(0.5, 1.0),
    'cubium': lambda: models.cubium_pythtb(1.0),
    'cubium2': lambda: models.cubium2_pythtb(1.0, 0.5),
}


@pytest.mark.parametrize('name', sorted(PYTHTB_MODELS))
def test_bands_are_pythtb_bands_at_k(name):
    """E(k) of the HRs equals PythTB's own H(k) at +k (time-reversal-odd Haldane was at -k)."""
    source = PYTHTB_MODELS[name]()
    model = models.pythtb_to_hopping_model(source)
    ndim = source._lattice._lat_vectors.shape[0]
    k = np.random.default_rng(11).random((8, 3))
    k[:, ndim:] = 0.0
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    norb = model.number_of_orbitals
    reference = [np.linalg.eigvalsh(source.hamiltonian(q[:ndim]).reshape(norb, norb)) for q in k]
    assert np.allclose(np.linalg.eigvalsh(hamiltonian_block(terms, k)), reference, atol=1e-12)


def test_spinful_centres_and_legacy_arrays(tmp_path):
    """Spin-orbitals (A up, B up | A down, B down) sit at (A, B, A, B); Hks and Dnm stay."""
    params = {'label': 'kane_mele_pythtb', 'delta': 0.4, 't': -1.0, 'soc': 0.06, 'rashba': 0.05}
    pao = PAOFLOW.PAOFLOW(model=params, outputdir=str(tmp_path / 'out'), verbose=False)
    arrays, attributes = pao.data_controller.data_dicts()
    sites = np.array([[1 / 3, 1 / 3, 0.0], [2 / 3, 2 / 3, 0.0]])
    assert np.allclose(orbital_positions_from_data(pao.data_controller), np.vstack([sites] * 2))
    assert attributes['dftSO'] and attributes['adhoc_SO'] and attributes['norb'] == 4
    assert 'nelec' not in attributes
    assert np.allclose(arrays['Hks'], np.fft.fftn(arrays['HRs'], axes=(2, 3, 4)))
    tau = np.asarray(arrays['tau'])
    assert np.allclose(arrays['Dnm'][0, 1], tau[0] - tau[1])
    assert np.allclose(arrays['Dnm'][0, 2], 0.0)


@pytest.mark.parametrize(
    'params',
    [
        {'label': 'graphene_pythtb', 'delta': 0.6, 't': -1.0},
        {'label': 'kane_mele_pythtb', 'delta': 0.4, 't': -1.0, 'soc': 0.06, 'rashba': 0.05},
    ],
    ids=['graphene', 'kane_mele'],
)
def test_shift_current_obeys_the_point_group(tmp_path, params):
    """D3h / C3v: sigma^yyy = -sigma^yxx, sigma^xxx = 0 through PAOFLOW(model=...)."""
    pao = PAOFLOW.PAOFLOW(model=params, outputdir=str(tmp_path / 'out'), verbose=False)
    arrays, attributes = pao.data_controller.data_dicts()
    out = nlo.nonlinear_optical_spectra(
        arrays['HRs'],
        arrays['a_vectors'],
        attributes['alat'],
        float(attributes['omega']),
        orbital_positions_from_data(pao.data_controller),
        2.0,
        (30, 30, 1),
        np.array([1.2, 2.0]),
        components=['yyy', 'yxx', 'xxx'],
        eta=0.1,
        spin_factor=1.0 if attributes['dftSO'] else 2.0,
        want=('shift',),
    )
    yyy, yxx, xxx = out['shift']
    scale = np.abs(yyy).max()
    assert scale > 0
    assert np.allclose(yxx, -yyy, atol=1e-10 * scale)
    assert np.allclose(xxx, 0.0, atol=1e-10 * scale)
