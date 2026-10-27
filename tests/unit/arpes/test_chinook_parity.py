"""Parity of the matrix elements with chinook on its square-lattice p-orbital tutorial model.

The fixture ``data/chinook_square_lattice.npz`` was produced by
``data/chinook_square_lattice_reference.py`` with chinook 1.1.3 (git 24913de) in a throwaway
environment: two Sb atoms with 5p orbitals, Slater-Koster nearest and next-nearest neighbour
hoppings, no spin, fixed radial integrals, a 6 x 5 momentum mesh at k_z = 0, h nu = 21.2 eV,
W = 4 eV.  chinook's hopping list is in its atomic gauge (connecting vectors include the orbital
positions); the PAOFLOW Hamiltonian is rebuilt in the cell-periodic gauge and the eigenvectors
are transformed back with ``e^{-i k.tau_i}`` before the matrix elements, which is the gauge
relation documented in :mod:`PAOFLOW.arpes.matrix_elements`.
"""

import os

import numpy as np

from PAOFLOW.arpes.basis import Orbital, PhotoemissionBasis
from PAOFLOW.arpes.bloch import BlochHamiltonian
from PAOFLOW.arpes.kinematics import final_state_directions, kinetic_energy, photoelectron_momentum
from PAOFLOW.arpes.matrix_elements import dipole_matrix_elements, polarization_weights
from PAOFLOW.arpes.radial import BOHR_ANGSTROM, RadialIntegrals

FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'data', 'chinook_square_lattice.npz'
)
QE_INDEX = {'x': 2, 'y': 3, 'z': 1}  # QE order of the p harmonics: z, x, y


def _model(d):
    """Cell-periodic HRs on a 3^3 grid from chinook's atomic-gauge hopping list (eV, Angstrom)."""
    avec = d['avec']
    pos = d['pos']
    nawf = pos.shape[0]
    ng = (3, 3, 3)
    HRs = np.zeros((nawf, nawf) + ng + (1,), dtype=complex)
    inv = np.linalg.inv(avec)
    for i, j, rx, ry, rz, re, im in d['hops']:
        i, j = int(i), int(j)
        t = re + 1j * im
        lattice = np.array([rx, ry, rz]) - (pos[j] - pos[i])
        n = np.rint(lattice @ inv).astype(int)
        assert np.allclose(lattice @ inv, n, atol=1e-3)  # chinook rounds positions to 1e-4 A
        # chinook: H_ij(k) = t e^{i k.R_ij} on the upper triangle; PAOFLOW: sum_R H(R) e^{-2 pi i k.R}
        HRs[i, j][tuple((-n) % 3)] += t
        if i != j:
            HRs[j, i][tuple(n % 3)] += np.conj(t)
    alat = 1.0 / BOHR_ANGSTROM  # 1 Angstrom lattice unit
    bloch = BlochHamiltonian.from_hrs(HRs, avec, alat_bohr=alat)
    orbitals = [
        Orbital(
            i,
            0 if i < 3 else 1,
            'Sb',
            pos[i],
            1,
            QE_INDEX[str(d['labels'][i])[-1]],
            n=5,
            label='5P',
        )
        for i in range(nawf)
    ]
    return bloch, PhotoemissionBasis(orbitals).with_atomic_radial('slater')


def test_hamiltonian_and_matrix_elements_match_chinook():
    d = np.load(FIXTURE)
    bloch, basis = _model(d)
    kpts = np.stack([d['state_kx'], d['state_ky'], np.zeros(d['state_kx'].size)], axis=1)
    unique, first = np.unique(kpts, axis=0, return_index=True)
    k_cart = kpts[np.sort(first)]  # (30, 3) 1/A
    energies, vectors = bloch.eigenstates(bloch.crystal_coordinates(k_cart))
    # the fixture lists the states k by k, band-sorted
    nb = energies.shape[1]
    ref = d['state_energy'].reshape(-1, nb)
    assert np.allclose(np.sort(energies, axis=1), np.sort(ref, axis=1), atol=1e-8)
    gaps = np.min(np.diff(np.sort(energies, axis=1), axis=1))
    assert gaps > 1e-4  # no degeneracy: states compare one to one
    # atomic gauge: c^at_i = c^cell_i e^{-i k.tau_i}
    phases = np.exp(-1j * (k_cart @ basis.positions.T))  # (nk, nawf)
    vectors_atomic = vectors * phases[:, :, None]
    kinetic = kinetic_energy(float(d['hv']), energies, float(d['W']))
    kf_norm = photoelectron_momentum(kinetic)
    directions, valid = final_state_directions(
        np.broadcast_to(k_cart[:, None, :2], energies.shape + (2,)), kf_norm
    )
    assert valid.all()
    k_final = directions * kf_norm[..., None]
    key = basis.orbitals[0].radial_key
    radial = RadialIntegrals(
        basis.radial_functions(), 2.0 * float(kf_norm.max()) * BOHR_ANGSTROM, nk=50,
        fixed={(key, 0): complex(d['fixed_l0']), (key, 2): complex(d['fixed_l2'])},
    )  # fmt: skip
    M = dipole_matrix_elements(basis, vectors_atomic, k_final, radial)
    theirs = d['weights']  # (npol, nstates) in the same state order (k by k, band-sorted)
    # equal up to one global constant (the normalisation of the fixed radial integrals and of
    # the dipole operator differ between the two codes); chinook rounds the orbital positions
    # of its hoppings to 1e-4 A, which limits the agreement to ~1e-4
    linear = [0, 1, 2, 4]  # x, y, z, (x + z) / sqrt 2
    ours = np.array([polarization_weights(M, d['pols'][ip]).ravel() for ip in linear])
    scale = np.sum(theirs[linear] * ours) / np.sum(ours * ours)
    assert scale > 0.0
    assert np.allclose(scale * ours, theirs[linear], rtol=2e-4, atol=1e-7 * theirs.max())
    # circular light: chinook's spherical components label the helicities the other way round,
    # its (x + i y) / sqrt 2 is the matrix element of (x - i y) / sqrt 2 here
    circular = np.conj(d['pols'][3])
    ours_circular = polarization_weights(M, circular).ravel()
    assert np.allclose(scale * ours_circular, theirs[3], rtol=2e-4, atol=1e-7 * theirs.max())
    ours_same = polarization_weights(M, d['pols'][3]).ravel()
    assert not np.allclose(scale * ours_same, theirs[3], rtol=1e-2, atol=1e-7 * theirs.max())
