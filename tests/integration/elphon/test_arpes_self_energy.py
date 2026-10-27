"""Fan-Migdal self-energy at arbitrary k and the ARPES map of Si on the DFPT asset (6^3 k, 3^3 q)."""

import os
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.arpes import ArpesExperiment, BlochHamiltonian, PhotoemissionBasis, simulate
from PAOFLOW.arpes.radial import BOHR_ANGSTROM
from PAOFLOW.elphon.elph_bloch import atom_masses
from PAOFLOW.elphon.pao_electrons import pao_hamiltonian_from_save
from PAOFLOW.elphon.self_energy import TabulatedSelfEnergy, fan_migdal_self_energy

pytestmark = pytest.mark.integration
BASIS = str(Path(__file__).resolve().parents[3] / 'BASIS')
MASSES_AMU = [28.0855]
NELEC = 8


@pytest.fixture(scope='module')
def electrons(elphon_assets, tmp_path_factory):
    save = os.path.join(elphon_assets, 'si.save')
    work = tmp_path_factory.mktemp('arpes')
    el = pao_hamiltonian_from_save(
        save, basispath=BASIS, configuration='standard', pthr=0.95, expand_wedge=False,
        workpath=str(work), outputdir='output',
    )  # fmt: skip
    return el, save, os.path.join(elphon_assets, 'save')


def _gamma_x_path(nscf, npts=4):
    """Gamma -> X in crystal coordinates and the same points in Cartesian 1/A."""
    x_cryst = np.array([0.5, 0.0, 0.5])  # X of the fcc lattice (b1 + b3) / 2
    k_cryst = np.linspace(0.0, 1.0, npts)[:, None] * x_cryst[None, :]
    k_cart = k_cryst @ nscf['bg'] * 2.0 * np.pi / (nscf['alat'] * BOHR_ANGSTROM)
    return k_cryst, k_cart


def test_fan_migdal_self_energy_on_a_path(electrons):
    el, save, dfpt = electrons
    nscf = el.nscf
    k_cryst, _ = _gamma_x_path(nscf)
    w = np.linspace(-2.0, 1.0, 61)
    result = fan_migdal_self_energy(
        el.projections, el.HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'],
        nscf['omega'], dfpt, (3, 3, 3), (6, 6, 6),
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']), nelec=NELEC,
        k_cryst=k_cryst, w_ev=w, temps_k=[300.0], nq_dense=6, eta_ev=0.02,
        coupling_source='dfpt', save_dir=save, tau_cryst=nscf['tau_cryst'],
        orbital_positions=nscf['orbital_positions'], allow_missing_long_range=True,
    )  # fmt: skip
    sigma = result['sigma_ev']
    assert sigma.shape == (1, 4, result['bands'].size, 61)
    assert np.all(np.isfinite(sigma)) and np.all(sigma.imag <= 1e-12)
    assert not result['metal']
    # the top valence band at Gamma: a phonon linewidth of a few to a few tens of meV at 300 K
    energies = result['energies_ev']
    top = int(np.argmax(np.where(energies[0] < 0.0, energies[0], -np.inf)))
    iw = int(np.argmin(np.abs(w - energies[0, top])))
    linewidth = -sigma[0, 0, top, iw].imag
    assert 1e-3 < linewidth < 0.3
    # the self-energy of the conduction states is non-zero as well (absorption + emission)
    assert np.abs(sigma[0, :, energies[0] > 0.0, :]).max() > 1e-4
    # Kramers-Kronig: Re Sigma follows from Im Sigma for a Lorentzian kernel (checked loosely:
    # the real part vanishes far from the bands)
    assert np.abs(sigma.real).max() < 10.0 * np.abs(sigma.imag).max() + 0.05


def test_arpes_map_with_the_tabulated_self_energy(electrons):
    el, save, dfpt = electrons
    nscf = el.nscf
    k_cryst, k_cart = _gamma_x_path(nscf)
    w = np.linspace(-2.0, 1.0, 61)
    result = fan_migdal_self_energy(
        el.projections, el.HRs, nscf['kpts_cryst'], nscf['bg'], nscf['at'], nscf['alat'],
        nscf['omega'], dfpt, (3, 3, 3), (6, 6, 6),
        masses_amu=atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names']), nelec=NELEC,
        k_cryst=k_cryst, w_ev=w, temps_k=[300.0], nq_dense=6, eta_ev=0.02,
        coupling_source='dfpt', save_dir=save, tau_cryst=nscf['tau_cryst'],
        orbital_positions=nscf['orbital_positions'], allow_missing_long_range=True,
    )  # fmt: skip
    basis = PhotoemissionBasis.from_data_controller(el.data_controller)
    assert basis.size == el.HRs.shape[0]
    assert all(o.radial is not None for o in basis.orbitals)
    bloch = BlochHamiltonian.from_data_controller(el.data_controller)
    # the Bloch Hamiltonian reproduces the energies of the self-energy run on the path
    e_path, _ = bloch.eigenstates(k_cryst)
    assert np.allclose(e_path[:, result['bands']], result['energies_abs_ev'], atol=1e-8)
    experiment = ArpesExperiment(
        photon_ev=30.0, work_function_ev=4.5, temperature_k=300.0, energy=(-2.0, 1.0, 61),
        kpoints=k_cart, resolution=(0.05, 0.0), polarization=(1.0, 0.0, 1.0),
    )  # fmt: skip
    bare = simulate(experiment, basis, bloch)
    dressed = simulate(experiment, basis, bloch, TabulatedSelfEnergy.from_result(result, k_cart))
    assert bare.intensity.shape == dressed.intensity.shape == (4, 61)
    assert np.all(np.isfinite(dressed.intensity)) and dressed.intensity.max() > 0.0
    # the phonon self-energy broadens the valence features: the maximum EDC intensity drops
    assert dressed.raw.max() < bare.raw.max()
    # occupied states only, up to the thermal tail
    assert dressed.raw[:, w > 0.3].max() < 1e-3 * dressed.raw.max()
