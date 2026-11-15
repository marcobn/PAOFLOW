"""Spectral assembly, self-energy helpers and the simulation driver of the ARPES module."""

import numpy as np

from PAOFLOW.arpes import (
    ArpesExperiment,
    BlochHamiltonian,
    ConstantSelfEnergy,
    GridSelfEnergy,
    IntensityMap,
    Orbital,
    PhotoemissionBasis,
    fermi_function,
    kramers_kronig_self_energy,
    simulate,
    spectral_function,
)
from PAOFLOW.arpes.kinematics import kz_free_electron
from PAOFLOW.arpes.spectral import assemble_intensity, gaussian_broadening


def test_spectral_function_is_normalised_and_centred():
    w = np.linspace(-5.0, 5.0, 20001)
    A = spectral_function(np.array([[-0.3, 0.2]]), -0.05j, w)
    assert A.shape == (1, 2, w.size)
    assert np.allclose(np.trapezoid(A, w, axis=-1), 1.0, atol=1e-2)  # Lorentzian tails beyond 5 eV
    assert np.allclose(w[np.argmax(A, axis=-1)], [[-0.3, 0.2]], atol=1e-3)


def test_assemble_intensity_matches_a_literal_state_loop():
    rng = np.random.default_rng(0)
    nk, nb = 7, 3
    w = np.linspace(-1.0, 0.2, 121)
    weights = rng.uniform(0.0, 1.0, size=(nk, nb))
    energies = rng.uniform(-0.8, 0.1, size=(nk, nb))
    sigma = GridSelfEnergy(w, -0.02j - 0.01 * w)
    out = assemble_intensity(weights, energies, w, sigma, temperature_k=50.0, kblock=3)
    fermi = fermi_function(w, 50.0)
    literal = np.zeros((nk, w.size))
    for ik in range(nk):
        for n in range(nb):
            literal[ik] += (
                weights[ik, n]
                * spectral_function(energies[ik : ik + 1, n : n + 1], sigma(w), w)[0, 0]
                * fermi
            )
    assert np.allclose(out, literal, rtol=1e-12, atol=1e-14)
    # a band-resolved array (nk, nb, nw) is accepted
    resolved = np.broadcast_to(sigma(w), (nk, nb, w.size)).copy()
    assert np.allclose(assemble_intensity(weights, energies, w, resolved, 50.0), out)
    assert ConstantSelfEnergy(0.02)(w) == -0.02j


def test_kramers_kronig_recovers_the_real_part_of_a_lorentzian():
    w = np.linspace(-4.0, 4.0, 4001)
    a, b = 0.05, 0.6
    im = -a * b**2 / (w**2 + b**2)
    sigma = kramers_kronig_self_energy(w, im, pad=50)
    expected_re = a * b * w / (w**2 + b**2)
    assert np.allclose(sigma.imag, im)
    assert np.allclose(sigma.real, expected_re, atol=2e-3)


def test_gaussian_broadening_keeps_the_integral():
    rng = np.random.default_rng(1)
    cube = rng.uniform(size=(20, 15, 100))
    out = gaussian_broadening(cube, (0.1, 0.0, 0.02), (0.02, None, 0.005))
    assert out.shape == cube.shape and np.isclose(out.sum(), cube.sum(), rtol=1e-6)


def _cubic_model():
    """One s and one p_z orbital on a simple cubic lattice (alat 5 bohr), energies in eV."""
    ng = (3, 3, 3)
    HRs = np.zeros((2, 2) + ng + (1,), dtype=complex)
    HRs[0, 0, 0, 0, 0, 0] = -1.0
    HRs[1, 1, 0, 0, 0, 0] = 1.5
    for axis in range(3):
        for sign in (1, -1):
            idx = [0, 0, 0]
            idx[axis] = sign % 3
            HRs[0, 0, idx[0], idx[1], idx[2], 0] = -0.6
            HRs[1, 1, idx[0], idx[1], idx[2], 0] = 0.3
            if axis == 2:  # s-p_z hopping along z, odd in R
                HRs[0, 1, idx[0], idx[1], idx[2], 0] = 0.4 * sign
                HRs[1, 0, idx[0], idx[1], idx[2], 0] = 0.4 * sign
    bloch = BlochHamiltonian.from_hrs(HRs, np.eye(3), 5.0, fermi_level=0.0)
    r = np.linspace(0.0, 30.0, 3001)
    orbitals = [
        Orbital(0, 0, 'Si', np.zeros(3), 0, 1, n=3, label='3S', radial=(r, 2.0 * r * np.exp(-r))),
        Orbital(
            1, 0, 'Si', np.zeros(3), 1, 1, n=3, label='3P', radial=(r, r * r * np.exp(-r / 2.0))
        ),
    ]
    return bloch, PhotoemissionBasis(orbitals)


def test_simulate_mesh_path_and_inner_potential(tmp_path):
    bloch, basis = _cubic_model()
    exp = ArpesExperiment(
        photon_ev=30.0, work_function_ev=4.0, temperature_k=20.0, energy=(-4.0, 1.0, 120),
        kx=(-1.0, 1.0, 9), ky=(-0.5, 0.5, 5), kz=0.3, resolution=(0.05, 0.05),
    )  # fmt: skip
    out = simulate(exp, basis, bloch, ConstantSelfEnergy(0.02))
    assert isinstance(out, IntensityMap)
    assert out.intensity.shape == (45, 120) and out.grid == (9, 5)
    assert np.all(np.isfinite(out.intensity)) and out.intensity.max() > 0.0
    assert out.bands.tolist() == [0, 1]
    # the Fermi factor removes the intensity above E_F
    above = out.w > 0.1
    assert out.raw[:, above].max() < 1e-6 * out.raw.max()
    cube = out.as_grid()
    assert cube.shape == (9, 5, 120)
    assert out.energy_distribution_curve(3).shape == (120,)
    assert out.momentum_distribution_curve(-1.0).shape == (45,)
    path = out.save(str(tmp_path / 'map.npz'))
    assert np.load(path)['intensity'].shape == (45, 120)
    # an explicit path, with the free-electron kz from the inner potential
    kpath = np.stack([np.linspace(0.0, 1.0, 12), np.zeros(12), np.zeros(12)], axis=1)
    exp_path = ArpesExperiment(
        photon_ev=30.0, inner_potential_ev=10.0, energy=(-4.0, 1.0, 60), kpoints=kpath,
        resolution=(0.0, 0.0), radial='slater', position_phase=True,
    )  # fmt: skip
    out_path = simulate(exp_path, basis, bloch)
    assert out_path.grid is None and out_path.intensity.shape == (12, 60)
    assert np.allclose(out_path.kpoints[:, 2], kz_free_electron(30.0, kpath[:, 0], 4.0, 10.0))
    assert np.allclose(out_path.intensity, out_path.raw)  # no broadening requested


def test_polarisation_selects_orbitals():
    """At normal emission along z, s and p_z states emit with z-polarised light only."""
    bloch, basis = _cubic_model()
    base = dict(
        photon_ev=30.0,
        energy=(-4.0, 1.0, 200),
        kx=(0.0, 0.0, 1),
        ky=(0.0, 0.0, 1),
        kz=0.2,
        resolution=(0.0, 0.0),
    )
    z_pol = simulate(
        ArpesExperiment(polarization=(0.0, 0.0, 1.0), **base),
        basis,
        bloch,
        ConstantSelfEnergy(0.05),
    )
    x_pol = simulate(
        ArpesExperiment(polarization=(1.0, 0.0, 0.0), **base),
        basis,
        bloch,
        ConstantSelfEnergy(0.05),
    )
    assert z_pol.intensity.max() > 0.0
    assert x_pol.intensity.max() < 1e-10 * z_pol.intensity.max()
