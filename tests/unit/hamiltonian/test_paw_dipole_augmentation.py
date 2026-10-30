"""PAW augmentation of the position dipoles: radial integrals, angular factors, on-site reconstruction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.elphon.pseudopotential_terms import qe_simpson
from PAOFLOW.hamiltonian import position_dipoles as pd
from PAOFLOW.inputs.read_upf import UPF

REPO_ROOT = Path(__file__).resolve().parents[3]
GA_PAW = REPO_ROOT / 'examples' / 'qe_examples' / 'example10' / 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF'


@pytest.fixture(scope='module')
def ga():
    if not GA_PAW.exists():
        pytest.skip('example10 Ga PAW file missing')
    return UPF(str(GA_PAW))


def test_angular_dipole_factors_are_the_known_ones():
    # <Y_00| r_hat_alpha |Y_1m> = 1/sqrt(3) for the matching p orbital (QE order z, x, y)
    a = pd._angular_dipole(0, 1)  # (3, 1, 3)
    expected = np.zeros((3, 1, 3))
    expected[2, 0, 0] = expected[0, 0, 1] = expected[1, 0, 2] = 1.0 / np.sqrt(3.0)
    assert np.allclose(a, expected, atol=1e-10)
    # parity: no s-s or p-p dipole; d-p and p-d are transposes
    assert np.allclose(pd._angular_dipole(1, 1), 0.0, atol=1e-12)
    assert np.allclose(pd._angular_dipole(2, 1), np.transpose(pd._angular_dipole(1, 2), (0, 2, 1)))


def test_partial_wave_dipoles_couple_adjacent_l_only(ga):
    radial = pd.partial_wave_dipole_radial(ga)
    assert radial.shape == (6, 6) and np.allclose(radial, radial.T)
    lll = [b['l'] for b in ga.beta]
    for i in range(6):
        for j in range(6):
            if abs(lll[i] - lll[j]) != 1:
                # same-l pairs still integrate to something, but the angular factor kills them;
                # the s-p and p-d radial integrals are the physical ones and are not small
                continue
            assert abs(radial[i, j]) > 1e-3


def test_on_site_dipole_of_the_reconstructed_all_electron_orbitals(ga):
    """<4s|r|4p> of the all-electron orbitals reconstructed from the pseudo-orbitals with the
    partial waves equals the pseudo value plus the projector augmentation (PAW identity)."""
    r, rab = ga.r, ga.rab
    chi = {c['label'].upper(): np.asarray(c['wfc'], dtype=float) for c in ga.pswfc}
    betas = [np.asarray(b['wfc'], dtype=float) for b in ga.beta]
    cut = [int(b['cutoff_index']) for b in ga.beta]

    def ip(u, v, n=None):
        n = n or r.size
        return qe_simpson((u * v)[:n], rab[:n])

    u_s, u_p = chi['4S'], chi['4P']
    lll = [int(b['l']) for b in ga.beta]
    # the projectors act within their own angular channel (the angular factors select l)
    becp_s = np.array([ip(b, u_s, c) if l == 0 else 0.0 for b, c, l in zip(betas, cut, lll)])
    becp_p = np.array([ip(b, u_p, c) if l == 1 else 0.0 for b, c, l in zip(betas, cut, lll)])
    # reconstructed all-electron radial functions
    ae_s = u_s + sum(becp_s[i] * (ga.aewfc[i] - ga.pswfc_partial[i]) for i in range(6))
    ae_p = u_p + sum(becp_p[i] * (ga.aewfc[i] - ga.pswfc_partial[i]) for i in range(6))
    direct = qe_simpson(ae_s * ae_p * r, rab)
    radial = pd.partial_wave_dipole_radial(ga)
    augmented = qe_simpson(u_s * u_p * r, rab) + becp_s @ radial @ becp_p
    assert abs(direct - augmented) < 2e-3 * abs(direct), (direct, augmented)
    # and the correction is not negligible for a semicore-augmented element
    assert abs(becp_s @ radial @ becp_p) > 1e-2 * abs(direct)


class _IsolatedGa:
    """A Ga atom in a 30 bohr box with its UPF pseudo-orbitals as the basis."""

    def __init__(self):
        self.rank = 0
        self.data_arrays = {
            'atoms': ['Ga'], 'tau': np.zeros((1, 3)),
            'species': [('Ga', 'Ga.pbe-dn-kjpaw_psl.1.0.0.UPF')],
            'configuration': {'Ga': ['4S', '4P', '3D']},
            'a_vectors': np.eye(3), 'b_vectors': np.eye(3),
        }  # fmt: skip
        self.data_attributes = {
            'ecutrho': 100.0, 'omega': 30.0**3, 'verbose': False, 'dftSO': False,
            'fpath': str(GA_PAW.parent), 'basispath': str(REPO_ROOT / 'BASIS'), 'alat': 30.0,
        }  # fmt: skip

    def data_dicts(self):
        return self.data_arrays, self.data_attributes


def test_isolated_atom_augmentation_reproduces_the_radial_references(ga):
    """On-site Delta S and Delta X of the module equal the radial PAW corrections of the
    pseudo-orbitals: S-orthonormality of the pseudo-orbitals and the all-electron <4s|r|4p>."""
    from PAOFLOW.elphon.pseudopotential_terms import qe_simpson
    from PAOFLOW.hamiltonian.nonlocal_velocity import load_pao_orbitals
    from PAOFLOW.projection.do_atwfc_proj import build_basis_from_sources

    dc = _IsolatedGa()
    basis, _ = build_basis_from_sources(dc, {'Ga': ['4S', '4P', '3D']})
    dc.data_arrays['atomic_basis'] = basis
    catalog = load_pao_orbitals(dc)
    pairs = pd.enumerate_pao_pairs(catalog, np.eye(3) * 30.0)
    tables = pd.build_position_tables(catalog, pairs)
    aug = pd.build_projector_augmentation(dc, catalog)
    assert aug is not None and aug.q_blocks[0].shape == (18, 18)
    k0 = np.zeros((1, 3))
    S, _, X = pd.assemble_position_kspace(tables, k0)
    dS, _, dX = pd.augmentation_kspace(aug, k0, tables.tau_orbitals)
    labels = [(b['label'], b['m']) for b in basis]
    s4, px, d3 = labels.index(('4S', 1)), labels.index(('4P', 2)), labels.index(('3D', 1))
    r, rab = ga.r, ga.rab
    chi = {c['label'].upper(): np.asarray(c['wfc'], dtype=float) for c in ga.pswfc}
    # the pseudo-orbitals are S-orthonormal: S + Delta S is the identity on site
    assert np.allclose(S[0] + dS[0], np.eye(S.shape[1]), atol=2e-3)
    assert abs(S[0, d3, d3].real - qe_simpson(chi['3D'] ** 2, rab)) < 1e-3
    # the dipole correction of <4s|x|4p_x>: (AE - pseudo) / sqrt(3) of the radial test above
    radial = pd.partial_wave_dipole_radial(ga)
    lll = [int(b['l']) for b in ga.beta]
    betas = [np.asarray(b['wfc'], dtype=float) for b in ga.beta]
    cut = [int(b['cutoff_index']) for b in ga.beta]
    bs = np.array(
        [
            qe_simpson((b * chi['4S'])[:c], rab[:c]) if l == 0 else 0.0
            for b, c, l in zip(betas, cut, lll)
        ]
    )
    bp = np.array(
        [
            qe_simpson((b * chi['4P'])[:c], rab[:c]) if l == 1 else 0.0
            for b, c, l in zip(betas, cut, lll)
        ]
    )
    expected = (bs @ radial @ bp) / np.sqrt(3.0)
    assert (
        abs(X[0, 0, s4, px].real - qe_simpson(chi['4S'] * chi['4P'] * r, rab) / np.sqrt(3.0)) < 2e-3
    )
    assert abs(dX[0, 0, s4, px].real - expected) < 2e-4, (dX[0, 0, s4, px].real, expected)
    assert np.allclose(dX[0, 1:, s4, px], 0.0, atol=1e-6)
    assert pd.hermiticity_residual(dX) < 1e-12
