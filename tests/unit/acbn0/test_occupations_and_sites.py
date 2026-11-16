"""ACBN0 occupations and Hubbard-site selection.

- ``ACBN0.Dk`` / ``Nmm``: the bare occupations (denominators of U and J) include every k-point, the first one too.
- ``_local_hubbard_manifolds`` / ``_projwfc_hubbard_manifolds``: the Hubbard site of the two-electron integrals is
  the shell of one atom, even when the shells of two atoms of the species are adjacent in the PAO list.
"""

from __future__ import annotations

import numpy as np

from PAOFLOW.ACBN0 import ACBN0


def _random_bands(nbasis: int, nk: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hermitian H(k) with the Fermi level inside the spectrum, S(k) = 1, uneven weights."""
    rng = np.random.default_rng(seed)
    hks = np.empty((nbasis, nbasis, nk), dtype=complex)
    for ik in range(nk):
        a = rng.normal(size=(nbasis, nbasis)) + 1j * rng.normal(size=(nbasis, nbasis))
        hks[:, :, ik] = 0.5 * (a + a.conj().T)
    sks = np.repeat(np.eye(nbasis, dtype=complex)[:, :, None], nk, axis=2)
    weights = rng.uniform(0.5, 2.0, size=nk)
    return hks, sks, weights


def test_bare_occupations_include_the_first_kpoint():
    hks, sks, weights = _random_bands(nbasis=6, nk=5, seed=3)
    basis_dm = np.array([0, 1, 2, 3])
    basis_2e = np.array([0, 1, 2])
    driver = ACBN0.__new__(ACBN0)
    _, nlm_k = driver.Dk(basis_dm, basis_2e, hks, sks)
    occupations = driver.Nmm(nlm_k, hks, weights)

    expected = np.zeros(len(basis_2e))
    for ik in range(hks.shape[2]):
        energies, vectors = np.linalg.eigh(hks[:, :, ik])
        occupied = vectors[:, energies <= 0.0]
        expected += weights[ik] * np.sum(np.abs(occupied[basis_2e]) ** 2, axis=1)
    expected /= weights.sum()
    assert np.allclose(occupations.real, expected, atol=1e-12)


def test_local_basis_site_is_one_atom_when_shells_are_adjacent(tmp_path):
    # two Ni atoms whose 3d shells follow each other in the PAO list, then two O 2p shells
    rows = []
    index = 0
    for atom, elem, label, l in [
        (1, 'Ni', '3D', 2),
        (2, 'Ni', '3D', 2),
        (3, 'O', '2P', 1),
        (4, 'O', '2P', 1),
    ]:
        for m in range(2 * l + 1):
            rows.append(f'{index} {atom} {elem} {l} {m} {label}')
            index += 1
    (tmp_path / 'pao_basis.dat').write_text('\n'.join(rows) + '\n')
    driver = ACBN0.__new__(ACBN0)
    driver.outputdir = str(tmp_path)
    driver._pao_meta = None
    driver.uVals = {'Ni-3d': 1.0, 'O-2p': 1.0}
    manifolds = driver._local_hubbard_manifolds()

    basis_dm, basis_2e = manifolds['Ni-3d']
    assert list(basis_dm) == list(range(10))
    assert list(basis_2e) == list(range(5))
    basis_dm, basis_2e = manifolds['O-2p']
    assert list(basis_dm) == list(range(10, 16))
    assert list(basis_2e) == list(range(10, 13))


def test_projwfc_site_is_one_atom_when_shells_are_adjacent():
    lines = []
    state = 1
    for atom, elem in [(1, 'Ni'), (2, 'Ni')]:
        for m in range(1, 6):
            lines.append(
                f'     state #{state:4d}: atom {atom:3d} ({elem}  ), wfc  1 (l=2 m= {m})\n'
            )
            state += 1
    driver = ACBN0.__new__(ACBN0)
    driver.uVals = {'Ni-3d': 1.0}
    driver.basis_labels = {}
    basis_dm, basis_2e = driver._projwfc_hubbard_manifolds(lines)['Ni-3d']
    assert list(basis_dm) == list(range(10))
    assert list(basis_2e) == list(range(5))
