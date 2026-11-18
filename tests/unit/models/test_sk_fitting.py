"""SKFitter: fit of a Slater-Koster model to H(R), export as a model dict, reload in PAOFLOW."""

from __future__ import annotations

import contextlib
import copy
import io

import numpy as np
import pytest

from PAOFLOW import PAOFLOW
from PAOFLOW.models.sk_fitting import SKFitter

pytestmark = pytest.mark.unit

FCC = [[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]]
ALAT = 10.26
ONSITE = {'s': -4.2, 'px': 1.7, 'py': 1.7, 'pz': 1.7}
NN = {'sss': -1.94, 'sps': 2.75, 'pps': 3.05, 'ppp': -1.08}


def _si_model(hoppings: dict) -> dict:
    atoms = {
        str(i): {'name': 'Si', 'tau': tau, 'orbitals': ['s', 'px', 'py', 'pz'], **ONSITE}
        for i, tau in enumerate(([0, 0, 0], [0.25, 0.25, 0.25]))
    }
    return {
        'label': 'Slater_Koster',
        'alat': ALAT,
        'model': {'a_vectors': FCC, 'atoms': atoms, 'hoppings': hoppings},
    }


def _load(model: dict) -> PAOFLOW.PAOFLOW:
    with contextlib.redirect_stdout(io.StringIO()):
        return PAOFLOW.PAOFLOW(model=copy.deepcopy(model), verbose=False)


def _fitter(p: PAOFLOW.PAOFLOW, configuration: bool = True, n_shells: int = 1) -> SKFitter:
    arrays = p.data_controller.data_arrays
    arryp = {
        'HRs': arrays['HRs'],
        'a_vectors': arrays['a_vectors'],
        'b_vectors': arrays['b_vectors'],
        'atoms': ['Si', 'Si'],
        'tau': np.array([[0, 0, 0], [0.25, 0.25, 0.25]]) * ALAT,
        'shells': {'Si': [0, 1]},
    }
    if configuration:
        arryp['configuration'] = {'Si': ['3S', '3P']}
    with contextlib.redirect_stdout(io.StringIO()):
        return SKFitter(
            arryp, {'alat': ALAT, 'natoms': 2}, n_shells=n_shells, nkfit=4, verbose=False
        )


@pytest.mark.parametrize('configuration', [True, False])
def test_fitted_model_dict_reloads_with_the_same_bands(configuration):
    """sp3 Si with one shell per l: the fit recovers the model exactly, and build_model_dict (shell-pair
    keys when the atoms carry a configuration, flat keys otherwise) is accepted by PAOFLOW(model=...)
    with the same eigenvalues (the dict was rejected with configuration labels: 'Invalid group-pair
    key sss')."""
    reference = _fitter(_load(_si_model(NN)), configuration)
    result = reference.fit(n_trials=6, seed=123)
    assert result['rmse'] < 1e-10
    model = reference.build_model_dict(result['p_opt'])
    first_shell = model['model']['hoppings']['Si-Si'][0]['params']
    assert ('3S-3P' in first_shell) is configuration
    reloaded = _fitter(_load(model), configuration)
    assert np.allclose(reloaded.E_pao, reference.E_pao, atol=1e-10)


def test_band_weights():
    """Unit weights are the unweighted fit; with a second-neighbour hopping the one-shell model cannot
    represent, weight on the four valence bands lowers their error; shapes are checked."""
    exact = _fitter(_load(_si_model(NN)))
    plain, unit = (
        exact.fit(n_trials=2, seed=4),
        exact.fit(n_trials=2, seed=4, band_weights=np.ones(8)),
    )
    assert np.allclose(plain['p_opt'], unit['p_opt'], atol=1e-12)
    assert unit['rmse'] == pytest.approx(plain['rmse'], abs=1e-12)

    second = {'Si-Si': [{'r_ref': 4.4427, 'params': NN},
                        {'r_ref': 7.2549, 'params': {'sss': -0.25, 'sps': 0.3, 'pps': 0.5, 'ppp': -0.2}}]}  # fmt: skip
    fitter = _fitter(_load(_si_model(second)))

    def valence_rmse(p_opt):
        return np.sqrt(np.mean((fitter.eigenvalues(p_opt) - fitter.E_pao)[:, :4] ** 2))

    all_bands = fitter.fit(n_trials=4, seed=2)
    valence = fitter.fit(n_trials=4, seed=2, band_weights=[1, 1, 1, 1, 0, 0, 0, 0])
    assert valence['rmse'] == pytest.approx(valence_rmse(valence['p_opt']), rel=1e-6)
    assert valence_rmse(valence['p_opt']) < 0.7 * valence_rmse(all_bands['p_opt'])

    with pytest.raises(ValueError):
        fitter.fit(n_trials=1, band_weights=np.ones(5))
    with pytest.raises(ValueError):
        fitter.fit(n_trials=1, band_weights=np.zeros(8))
