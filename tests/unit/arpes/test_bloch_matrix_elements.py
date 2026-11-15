"""Bloch Hamiltonian, matrix elements and kinematics of the ARPES module."""

import numpy as np
import pytest

from PAOFLOW.arpes.basis import Orbital, PhotoemissionBasis
from PAOFLOW.arpes.bloch import BlochHamiltonian
from PAOFLOW.arpes.kinematics import (
    KINETIC_PREFACTOR,
    final_state_directions,
    kz_free_electron,
    momentum_mesh,
    photoelectron_momentum,
)
from PAOFLOW.arpes.matrix_elements import dipole_matrix_elements, polarization_weights
from PAOFLOW.arpes.radial import BOHR_ANGSTROM, RadialIntegrals, radial_integral


def _random_hks(nawf=3, ng=4, seed=0):
    rng = np.random.default_rng(seed)
    axes = np.arange(ng) / ng
    K = np.stack(np.meshgrid(axes, axes, axes, indexing='ij'), axis=-1).reshape(-1, 3)
    # a smooth hermitian H(k) from a few real-space hoppings
    H = np.zeros((nawf, nawf, K.shape[0]), dtype=complex)
    cells = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    hops = [rng.normal(size=(nawf, nawf)) + 1j * rng.normal(size=(nawf, nawf)) for _ in cells]
    hops[0] = hops[0] + hops[0].conj().T
    for cell, t in zip(cells, hops):
        phase = np.exp(-2j * np.pi * (K @ np.array(cell)))
        H += t[:, :, None] * phase[None, None, :]
        if any(cell):
            H += t.conj().T[:, :, None] * np.conj(phase)[None, None, :]
    return K, H.reshape(nawf, nawf, ng, ng, ng)


def test_bloch_hamiltonian_reproduces_the_grid_hamiltonian():
    K, Hks = _random_hks()
    HRs = np.fft.ifftn(Hks, axes=(2, 3, 4))[..., None]  # PAOFLOW convention, one spin
    bloch = BlochHamiltonian.from_hrs(HRs, np.eye(3), alat_bohr=5.0)
    Hk = bloch.hamiltonian(K)
    assert np.allclose(np.transpose(Hk, (1, 2, 0)), Hks.reshape(3, 3, -1), atol=1e-10)
    e, v = bloch.eigenstates(K[:5])
    assert e.shape == (5, 3) and v.shape == (5, 3, 3)
    # crystal coordinates of a Cartesian momentum: k . a / 2 pi
    k_cart = np.array([[2.0 * np.pi / (5.0 * BOHR_ANGSTROM), 0.0, 0.0]])
    assert np.allclose(bloch.crystal_coordinates(k_cart), [[1.0, 0.0, 0.0]])


def _s_basis(position=(0.0, 0.0, 0.0), spin=0):
    r = np.linspace(0.0, 30.0, 3001)
    chi = 2.0 * r * np.exp(-r)
    return PhotoemissionBasis(
        [
            Orbital(
                0,
                0,
                'H',
                np.array(position, dtype=float),
                0,
                1,
                n=1,
                label='1S',
                radial=(r, chi),
                spin=spin,
            )
        ]
    )


def test_s_orbital_emission_is_isotropic_and_scales_with_the_radial_integral():
    basis = _s_basis()
    radial = RadialIntegrals(basis.radial_functions(), 4.0, nk=801)
    rng = np.random.default_rng(2)
    kf = rng.normal(size=(5, 2, 3))
    kf *= (1.5 / np.linalg.norm(kf, axis=-1))[..., None]  # |k_f| = 1.5 1/A everywhere
    c = np.ones((5, 1, 2), dtype=complex)
    M = dipole_matrix_elements(basis, c, kf, radial)
    assert M.shape == (5, 2, 1, 3)
    total = np.sum(np.abs(M) ** 2, axis=(2, 3))
    B = radial_integral(*basis.orbitals[0].radial, 1, 1.5 * BOHR_ANGSTROM)[0]
    assert np.allclose(total, np.abs(B) ** 2 / (4.0 * np.pi), rtol=5e-3)
    # linear polarisation along the emission direction picks the z-like component only
    weights = polarization_weights(M, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(
        weights, np.abs(B) ** 2 * (3.0 / (4.0 * np.pi)) / 3.0 * (kf[..., 0] / 1.5) ** 2, rtol=5e-3
    )


def test_position_phase_and_escape_depth():
    basis = _s_basis(position=(0.3, -0.2, 1.1))
    radial = RadialIntegrals(basis.radial_functions(), 4.0, nk=401)
    kf = np.array([[[0.2, 0.1, 1.0]]])
    k = np.array([[0.2, 0.1, 0.0]])
    c = np.ones((1, 1, 1), dtype=complex)
    plain = dipole_matrix_elements(basis, c, kf, radial)
    phased = dipole_matrix_elements(basis, c, kf, radial, k_hamiltonian=k)
    expected = np.exp(-1j * np.dot(kf[0, 0] - k[0], basis.orbitals[0].position))
    assert np.allclose(phased, plain * expected)
    deep = PhotoemissionBasis(
        [
            Orbital(
                0,
                0,
                'H',
                np.zeros(3),
                0,
                1,
                n=1,
                label='1S',
                radial=basis.orbitals[0].radial,
                depth=4.0,
            )
        ]
    )
    attenuated = dipole_matrix_elements(
        deep, c, kf, RadialIntegrals(deep.radial_functions(), 4.0, nk=401), escape_depth=8.0
    )
    assert np.allclose(attenuated, plain * np.exp(-4.0 / 16.0))


def test_closed_p_shell_emission_is_isotropic():
    r = np.linspace(0.0, 30.0, 3001)
    chi = r * r * np.exp(-r / 2.0)
    orbitals = [
        Orbital(i, 0, 'C', np.zeros(3), 1, i + 1, n=2, label='2P', radial=(r, chi))
        for i in range(3)
    ]
    basis = PhotoemissionBasis(orbitals)
    radial = RadialIntegrals(basis.radial_functions(), 4.0, nk=801)
    rng = np.random.default_rng(3)
    kf = rng.normal(size=(6, 1, 3))
    kf *= (1.2 / np.linalg.norm(kf, axis=-1))[..., None]
    kf = np.repeat(kf, 3, axis=1)  # the same photoelectron momentum for the three bands
    c = np.tile(np.eye(3, dtype=complex)[None], (6, 1, 1))  # the three p states as bands
    M = dipole_matrix_elements(basis, c, kf, radial)
    total = np.sum(np.abs(M) ** 2, axis=(1, 2, 3))  # summed over the shell and polarisations
    assert np.allclose(total, total[0], rtol=1e-6)


def test_spinor_basis_fills_two_spin_components():
    up = _s_basis(spin=1).orbitals[0]
    down = Orbital(1, 0, 'H', np.zeros(3), 0, 1, n=1, label='1S', radial=up.radial, spin=-1)
    basis = PhotoemissionBasis([up, down])
    radial = RadialIntegrals(basis.radial_functions(), 4.0, nk=401)
    c = np.array([[[1.0], [0.5j]]])
    M = dipole_matrix_elements(basis, c, np.array([[[0.0, 0.0, 1.0]]]), radial)
    assert M.shape == (1, 1, 2, 3)
    assert np.allclose(M[0, 0, 1], 0.5j * M[0, 0, 0])


def test_kinematics():
    assert np.isclose(photoelectron_momentum(1.0), KINETIC_PREFACTOR)
    assert np.isclose(
        kz_free_electron(21.2, 0.0, 4.5, 12.0), KINETIC_PREFACTOR * np.sqrt(21.2 - 4.5 + 12.0)
    )
    assert kz_free_electron(21.2, 10.0, 4.5, 0.0) == 0.0  # evanescent
    kp, kx, ky = momentum_mesh((-1.0, 1.0, 3), (0.0, 0.5, 2), kz=0.7, azimuth=np.pi / 2)
    assert kp.shape == (6, 3) and np.allclose(kp[:, 2], 0.7)
    # a laboratory point on +x lies on -y in the sample frame after a 90 degree azimuth
    assert np.allclose(kp[np.argmin(np.abs(kx - 1.0)) * 2], [0.0, -1.0, 0.7], atol=1e-12)
    directions, valid = final_state_directions(
        np.array([[0.3, 0.4], [3.0, 4.0]]), np.array([1.0, 1.0])
    )
    assert valid.tolist() == [True, False]
    assert np.allclose(directions[0], [0.3, 0.4, np.sqrt(1.0 - 0.25)]) and np.allclose(
        directions[1], 0.0
    )


def test_basis_helpers():
    basis = _s_basis()
    assert not basis.is_spinor and basis.size == 1
    slater = basis.with_atomic_radial('slater')
    assert slater.orbitals[0].radial[0].size == 2000
    with pytest.raises(ValueError):
        PhotoemissionBasis([Orbital(0, 0, 'H', np.zeros(3), 0, 1)]).radial_functions()
    stacked = PhotoemissionBasis(
        [
            Orbital(
                i,
                i,
                'H',
                np.array([0.0, 0.0, 2.0 * i]),
                0,
                1,
                n=1,
                label='1S',
                radial=basis.orbitals[0].radial,
            )
            for i in range(3)
        ]
    ).with_depths([0.0, 0.0, 1.0])
    assert [o.depth for o in stacked.orbitals] == [4.0, 2.0, 0.0]


def test_slab_of_a_cubic_s_band_is_a_finite_chain():
    """At k_parallel = 0 the slab levels are those of an open chain, -2t cos(pi j / (N + 1))."""
    from PAOFLOW.arpes.slab import build_slab

    ng = (3, 3, 3)
    t = 0.4
    HRs = np.zeros((1, 1) + ng + (1,), dtype=complex)
    for axis in range(3):
        for sign in (1, -1):
            idx = [0, 0, 0]
            idx[axis] = sign % 3
            HRs[0, 0, idx[0], idx[1], idx[2], 0] = -t
    bloch = BlochHamiltonian.from_hrs(HRs, np.eye(3), alat_bohr=4.0)
    basis = _s_basis()
    slab, slab_basis = build_slab(bloch, basis, nlayers=5, axis=2)
    assert slab.nawf == 5 and slab_basis.size == 5
    e, _ = slab.eigenstates(np.array([[0.0, 0.0, 0.0]]))
    chain = np.sort(-2.0 * t * 2.0 - 2.0 * t * np.cos(np.pi * np.arange(1, 6) / 6.0))
    assert np.allclose(np.sort(e[0]), chain, atol=1e-12)
    # in-plane dispersion unchanged, k_z irrelevant
    e2, _ = slab.eigenstates(np.array([[0.25, 0.0, 0.0], [0.25, 0.0, 0.37]]))
    assert np.allclose(e2[0], e2[1])
    assert np.isclose(
        e2[0].min(), chain[0] + 2.0 * t * (1.0 - np.cos(2.0 * np.pi * 0.25)), atol=1e-12
    )
    # depths: layer 0 on top (0 A), one lattice constant per layer
    depths = [o.depth for o in slab_basis.orbitals]
    assert np.allclose(depths, 4.0 * BOHR_ANGSTROM * np.arange(5))
    assert np.allclose(slab_basis.positions[:, 2], -4.0 * BOHR_ANGSTROM * np.arange(5))


def test_spinor_shell_emission_equals_the_scalar_shell():
    """The 2(2l+1) spinor functions of a p shell (|l j m_j>) span the same space as the 3 real
    harmonics x 2 spins: the emission summed over the shell is identical, and isotropic."""
    from PAOFLOW.arpes.harmonics import spinor_components

    r = np.linspace(0.0, 30.0, 3001)
    chi = r * r * np.exp(-r / 2.0)
    comps = spinor_components(1)  # (6, 2, 3)
    assert np.allclose(np.einsum('isq,jsq->ij', comps, np.conj(comps)), np.eye(6))  # unitary
    spinor = []
    for p in range(6):
        terms = [
            (q + 1, +1 if s == 0 else -1, complex(comps[p, s, q]))
            for s in range(2)
            for q in range(3)
        ]
        spinor.append(
            Orbital(
                p,
                0,
                'C',
                np.zeros(3),
                1,
                1,
                n=2,
                label='2P',
                radial=(r, chi),
                spin=1,
                components=terms,
                j=0.5 if p < 2 else 1.5,
            )
        )
    spinor_basis = PhotoemissionBasis(spinor)
    assert spinor_basis.is_spinor
    scalar = [
        Orbital(
            i,
            0,
            'C',
            np.zeros(3),
            1,
            (i % 3) + 1,
            n=2,
            label='2P',
            radial=(r, chi),
            spin=+1 if i < 3 else -1,
        )
        for i in range(6)
    ]
    scalar_basis = PhotoemissionBasis(scalar)
    rng = np.random.default_rng(7)
    kf = rng.normal(size=(5, 1, 3))
    kf *= (1.3 / np.linalg.norm(kf, axis=-1))[..., None]
    kf = np.repeat(kf, 6, axis=1)
    c = np.tile(np.eye(6, dtype=complex)[None], (5, 1, 1))
    radial_s = RadialIntegrals(spinor_basis.radial_functions(), 4.0, nk=801)
    radial_c = RadialIntegrals(scalar_basis.radial_functions(), 4.0, nk=801)
    M_spinor = dipole_matrix_elements(spinor_basis, c, kf, radial_s)
    M_scalar = dipole_matrix_elements(scalar_basis, c, kf, radial_c)
    total_spinor = np.sum(np.abs(M_spinor) ** 2, axis=(1, 2, 3))
    total_scalar = np.sum(np.abs(M_scalar) ** 2, axis=(1, 2, 3))
    assert np.allclose(total_spinor, total_scalar, rtol=1e-10)
    assert np.allclose(total_spinor, total_spinor[0], rtol=1e-6)  # closed shell: isotropic
    # per polarisation as well (the two bases are unitarily related)
    for eps in ([1.0, 0.0, 0.0], [0.0, 1.0, 1j]):
        w_s = polarization_weights(M_spinor, np.array(eps)).sum(axis=1)
        w_c = polarization_weights(M_scalar, np.array(eps)).sum(axis=1)
        assert np.allclose(w_s, w_c, rtol=1e-10)
