"""Landauer bulk transmission of a 3D crystal: transverse hoppings and transverse k-mesh."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

HOPPING = -1.0


def _cubic_transport(
    tmp_path: Path,
    nk_parallel: int | tuple[int, int],
    configure_before_build: bool = True,
    double_x: bool = False,
):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.Transport import Transport
    from PAOFLOW.models.models import single_band_model

    hoppings = {(1, 0, 0): HOPPING, (0, 1, 0): HOPPING, (0, 0, 1): HOPPING}
    model = single_band_model(hoppings, np.eye(3), 5.0, 1.0, (4, 4, 4))
    nk = tuple(nk_parallel) if isinstance(nk_parallel, tuple) else (nk_parallel, nk_parallel)
    grid = dict(emin=-7.0, emax=7.0, ne=141, delta=0.0005, nk=nk)
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        if double_x:  # two sites in the transverse cell: complex on-site blocks H00(k_par)
            p.doubling_Hamiltonian(1, 0, 0)
        transport = Transport(p.data_controller)
        transport.define_partition(central_atoms='ALL', transport_direction='z')
        transport.configure_outputs(output_dir=str(tmp_path), postfix='_bulk')
        if configure_before_build:
            transport.configure_energy_grid(**grid)
        transport.build_hamiltonian_blocks(
            calculation_type='bulk', use_sym=False, do_overlap_transformation=False
        )
        if not configure_before_build:
            transport.configure_energy_grid(**grid)
        transport.compute_leads_self_energy(write=False)
        transport.compute_greens_functions(write=False)
        transport.compute_transmission()
    return np.loadtxt(tmp_path / 'conductance_bulk.dat')


def _channels_per_transverse_k(energies: np.ndarray, n_parallel: int) -> np.ndarray:
    """Exact average number of channels: 1 where |E - e_par(k_par)| < 2|t| on the transverse mesh."""
    k = 2 * np.pi * np.arange(n_parallel) / n_parallel
    e_parallel = (2 * HOPPING * (np.cos(k)[:, None] + np.cos(k)[None, :])).ravel()
    inside = np.abs(energies[:, None] - e_parallel[None, :]) < 2 * abs(HOPPING)
    away_from_edges = (
        np.abs(np.abs(energies[:, None] - e_parallel[None, :]) - 2 * abs(HOPPING)).min(axis=1)
        > 0.05
    )
    return inside.mean(axis=1), away_from_edges


@pytest.mark.parametrize('nk_parallel', [0, 8])
def test_simple_cubic_bulk_transmission_counts_the_channels(tmp_path, nk_parallel):
    conductance = _cubic_transport(tmp_path, nk_parallel)
    n_parallel = nk_parallel or 4  # default: the transverse mesh of the model
    exact, away = _channels_per_transverse_k(conductance[:, 0], n_parallel)
    assert np.abs(conductance[away, 1] - exact[away]).max() < 2e-3
    conducting = conductance[conductance[:, 1] > 1e-3, 0]
    assert conducting.min() < -5.5 and conducting.max() > 5.5  # the 3D band [-6, 6], not the chain


def test_late_transverse_mesh_request_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='before build_hamiltonian_blocks'):
        _cubic_transport(tmp_path, 8, configure_before_build=False)


def test_complex_transverse_blocks(tmp_path):
    """The cell doubled along x has complex H00(k_par); per transverse cell it carries twice the
    channels of the single cell (same transverse density of k-points)."""
    single = _cubic_transport(tmp_path / 'single', (8, 8))
    doubled = _cubic_transport(tmp_path / 'doubled', (4, 8), double_x=True)
    exact, away = _channels_per_transverse_k(single[:, 0], 8)
    assert np.abs(doubled[away, 1] - 2 * exact[away]).max() < 4e-3


def test_doubling_updates_the_reciprocal_vectors(tmp_path):
    from PAOFLOW import PAOFLOW
    from PAOFLOW.models.models import single_band_model

    model = single_band_model({(1, 0, 0): -1.0, (0, 0, 1): -0.5}, np.eye(3), 5.0, 1.0, (4, 4, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        p.doubling_Hamiltonian(0, 0, 1)
    arrays = p.data_controller.data_arrays
    assert np.allclose(
        np.asarray(arrays['a_vectors']) @ np.asarray(arrays['b_vectors']).T, np.eye(3)
    )


def test_conductor_with_an_impurity_plane(tmp_path):
    """Cubic lattice, on-site V on one plane: k_par is conserved and each transverse k sees a chain
    with one impurity, T = (4 - e^2) / (4 - e^2 + V^2), e = E - e_par(k_par) (t = -1)."""
    from PAOFLOW import PAOFLOW
    from PAOFLOW.Transport import Transport
    from PAOFLOW.models.models import single_band_model

    impurity = 1.5
    hoppings = {(1, 0, 0): HOPPING, (0, 1, 0): HOPPING, (0, 0, 1): HOPPING}
    model = single_band_model(hoppings, np.eye(3), 5.0, 1.0, (4, 4, 4))
    with contextlib.redirect_stdout(io.StringIO()):
        p = PAOFLOW.PAOFLOW(model=model, outputdir=str(tmp_path), verbose=False)
        p.doubling_Hamiltonian(0, 0, 3)  # 8 sites along z
        hrs = p.data_controller.data_arrays['HRs']
        band_orbitals = np.flatnonzero(np.abs(np.real(np.diag(hrs[:, :, 0, 0, 0, 0]))) < 1e-9)
        site = band_orbitals[len(band_orbitals) // 2]
        hrs[site, site, 0, 0, 0, 0] += impurity
        transport = Transport(p.data_controller)
        transport.define_partition(
            central_atoms='ALL', left_lead_layers=2, right_lead_layers=2, transport_direction='z'
        )
        transport.configure_outputs(output_dir=str(tmp_path), postfix='_imp')
        transport.configure_energy_grid(emin=-6.5, emax=6.5, ne=131, delta=0.0005, nk=(8, 8))
        transport.build_hamiltonian_blocks(
            calculation_type='conductor', use_sym=False, do_overlap_transformation=False
        )
        transport.compute_leads_self_energy(write=False)
        transport.compute_greens_functions(write=False)
        transport.compute_transmission()
    conductance = np.loadtxt(tmp_path / 'conductance_imp.dat')
    k = 2 * np.pi * np.arange(8) / 8
    e_parallel = (2 * HOPPING * (np.cos(k)[:, None] + np.cos(k)[None, :])).ravel()
    eps = conductance[:, 0][:, None] - e_parallel[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        chains = np.where(np.abs(eps) < 2, (4 - eps**2) / (4 - eps**2 + impurity**2), 0.0)
    away = np.abs(np.abs(eps) - 2).min(axis=1) > 0.05
    assert np.abs(conductance[away, 1] - chains.mean(axis=1)[away]).max() < 4e-3
