"""Hopping-list models of PAOFLOW.models.models: spectra, the DataController round trip, the dict labels."""

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

pytestmark = pytest.mark.unit


def bands(model: models.HoppingModel, k_cryst: np.ndarray) -> np.ndarray:
    terms = bloch_terms_from_HRs(
        model.real_space_hamiltonian(), model.lattice_vectors, model.alat, model.orbital_positions
    )
    return np.linalg.eigvalsh(hamiltonian_block(terms, np.atleast_2d(k_cryst)))


def test_honeycomb_gaps_at_the_valleys():
    """Gapped graphene: |Delta| at K and K'; Haldane: 2 |Delta/2 -+ 3 sqrt(3) t2 sin(phi)|."""
    valleys = np.array([[1 / 3, 2 / 3, 0.0], [2 / 3, 1 / 3, 0.0]])
    gaps = np.diff(bands(models.honeycomb_model(delta=0.6, t=-1.0), valleys), axis=1)[:, 0]
    assert np.allclose(gaps, 0.6, atol=1e-12)
    haldane = models.honeycomb_model(delta=0.2, t=-1.0, t2=0.1, phi=np.pi / 2)
    gaps = np.sort(np.diff(bands(haldane, valleys), axis=1)[:, 0])
    expected = np.sort(2.0 * np.abs([0.1 - 3 * np.sqrt(3) * 0.1, 0.1 + 3 * np.sqrt(3) * 0.1]))
    assert np.allclose(gaps, expected, atol=1e-12)


def test_weyl_nodes_and_rice_mele_gap():
    """Weyl: degenerate at k = (0, 0, +-1/4) with energies +-b; Rice-Mele: gap 2 sqrt(Delta^2 + 4 dt^2) at the zone edge."""
    weyl = models.weyl_model(t=1.0, b=0.3)
    nodes = bands(weyl, np.array([[0, 0, 0.25], [0, 0, -0.25]]))
    assert np.allclose(nodes, [[0.3, 0.3], [-0.3, -0.3]], atol=1e-12)
    chain = models.rice_mele_model(t=1.0, dt=0.3, delta=0.4)
    gap = np.diff(bands(chain, np.array([0.5, 0.0, 0.0])), axis=1)[0, 0]
    assert gap == pytest.approx(2.0 * np.sqrt(0.4**2 + (2 * 0.3) ** 2), abs=1e-12)


def test_single_band_dispersion():
    """The band is sum_R h(R) e^{2 pi i k.R} (partners added), the second orbital stays at 12 eV."""
    band = {(1, 0, 0): -1.0, (0, 1, 0): -0.8, (2, 0, 0): 0.25j}
    model = models.single_band_model(band, np.diag([1.0, 1.0, 10.0]), 5.0, 0.8, (5, 5, 1))
    k = np.array([[0.13, 0.27, 0.0], [-0.31, 0.05, 0.0]])
    exact = sum(2.0 * (h * np.exp(2j * np.pi * (k @ np.array(R)))).real for R, h in band.items())
    assert np.allclose(bands(model, k), np.stack([exact, np.full(2, 12.0)], axis=1), atol=1e-12)


def test_grid_too_small_is_refused():
    with pytest.raises(ValueError, match='cannot hold'):
        models.hoppings_to_HRs([(0, 0, (2, 0, 0), 1.0)], 1, (4, 1, 1))


@pytest.mark.parametrize(
    'params, reference',
    [
        (
            {'label': 'honeycomb', 'delta': 0.6, 't': -1.0, 't2': 0.15, 'phi': 1.0, 'strain': 0.2},
            models.honeycomb_model(0.6, -1.0, 0.15, 1.0, strain=0.2),
        ),
        ({'label': 'weyl', 't': 1.0, 'b': 0.3}, models.weyl_model(1.0, 0.3)),
        ({'label': 'Rice_Mele', 'delta': -0.4}, models.rice_mele_model(delta=-0.4)),
    ],
)
def test_paoflow_model_round_trip(tmp_path, params, reference):
    """PAOFLOW(model=...) stores the HRs, cell and orbital centres the optical responses read."""
    pao = PAOFLOW.PAOFLOW(model=params, outputdir=str(tmp_path / 'out'), verbose=False)
    arrays, attributes = pao.data_controller.data_dicts()
    assert np.array_equal(arrays['HRs'], reference.real_space_hamiltonian())
    assert np.allclose(arrays['a_vectors'], reference.lattice_vectors)
    assert attributes['alat'] == reference.alat
    assert attributes['omega'] == pytest.approx(reference.volume)
    assert attributes['nelec'] == reference.nelec
    assert attributes['dftSO'] is reference.spinful
    positions = orbital_positions_from_data(pao.data_controller)
    assert np.allclose(positions, reference.orbital_positions, atol=1e-12)


def test_unknown_parameter_is_refused(tmp_path):
    with pytest.raises(TypeError):
        PAOFLOW.PAOFLOW(
            model={'label': 'weyl', 'mass': 1.0}, outputdir=str(tmp_path / 'out'), verbose=False
        )
