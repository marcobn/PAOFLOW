"""Molecular post-processing of the second-order responses and the per-channel electron counts."""

import numpy as np
import pytest

from PAOFLOW.response import molecular_nlo as mol
from PAOFLOW.response.nonlinear_optics import channel_electron_counts, component_label

pytestmark = pytest.mark.unit


def test_conversion_constants():
    assert abs(mol.BETA_AU_PER_BOHR3_PMV - 0.040919) < 2e-6
    assert abs(mol.BETA_AU_SI - 3.2063613e-53) < 1e-58
    assert abs(mol.BETA_AU_TO_ESU - 8.6392e-33) < 2e-36
    chi = np.array([[1.0 + 0.5j]])
    beta = mol.hyperpolarizability(chi, 1000.0)
    assert np.isclose(beta[0, 0], -2.0 * 0.040919 * 1000.0 * (1.0 + 0.5j), rtol=1e-4)


def test_orientational_averages_of_a_single_component_tensor():
    """beta_zzz only: <b_ZZZ^2> = b^2/7, <b_XZZ^2> = b^2/35, depolarisation ratio 5, beta_par = 3b/5."""
    beta = np.zeros((3, 3, 3))
    beta[2, 2, 2] = 4.0
    zzz, xzz = mol.hrs_averages(beta)
    assert np.isclose(zzz, 16.0 / 7.0, rtol=1e-12)
    assert np.isclose(xzz, 16.0 / 35.0, rtol=1e-12)
    hrs = mol.hrs_quantities(beta)
    assert np.isclose(hrs['depolarization_ratio'], 5.0, rtol=1e-12)
    assert np.isclose(hrs['beta_hrs'], 4.0 * np.sqrt(6.0 / 35.0), rtol=1e-12)
    assert np.isclose(mol.beta_parallel(beta, np.array([0.0, 0.0, 2.0])), 12.0 / 5.0)
    assert np.isclose(mol.beta_parallel(beta, np.array([0.0, 0.0, -2.0])), -12.0 / 5.0)
    # the averages are rotation invariant (a rotated single-component tensor gives the same numbers)
    rng = np.random.default_rng(0)
    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    rotated = np.einsum('ia,jb,kc,abc->ijk', Q, Q, Q, beta)
    zzz_r, xzz_r = mol.hrs_averages(rotated)
    assert np.isclose(zzz_r, zzz, rtol=1e-12) and np.isclose(xzz_r, xzz, rtol=1e-12)
    # frequency axis and complex values are carried through
    spec = beta[..., None] * np.array([1.0, 1.0 + 1.0j])[None, None, None, :]
    zzz_s, _ = mol.hrs_averages(spec)
    assert np.allclose(zzz_s, [16.0 / 7.0, 32.0 / 7.0])


def test_tensor_from_components_and_missing_component():
    labels = [component_label((a, b, c)) for a in range(3) for b in range(3) for c in range(3)]
    values = np.arange(27.0)[:, None] * np.ones((1, 2))
    tensor = mol.tensor_from_components(values, labels)
    assert tensor.shape == (3, 3, 3, 2)
    assert tensor[1, 2, 0, 1] == labels.index('yzx')
    with pytest.raises(ValueError):
        mol.tensor_from_components(values[:5], labels[:5])


def test_ground_state_dipole_of_a_two_site_model():
    """Two sites at +-1 bohr, on-site asymmetry: the occupied bonding state leans to the lower site."""
    H = np.array([[-1.0, -0.5], [-0.5, 1.0]])
    tau = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mu = mol.ground_state_dipole(H, tau, 1, 2.0)
    _, U = np.linalg.eigh(H)
    expected = -2.0 * (abs(U[0, 0]) ** 2 * (-1.0) + abs(U[1, 0]) ** 2 * 1.0)
    assert np.isclose(mu[0], expected) and mu[0] > 0.0  # electrons at -x: dipole toward +x
    assert mu[1] == 0.0 and mu[2] == 0.0
    # ionic charges make a neutral molecule's dipole origin-independent (here they cancel: +1 at +-1 bohr)
    total = mol.ground_state_dipole(H, tau, 1, 2.0, ionic=(tau, np.array([1.0, 1.0])))
    shifted = mol.ground_state_dipole(H, tau + 5.0, 1, 2.0, ionic=(tau + 5.0, np.array([1.0, 1.0])))
    assert np.allclose(total, shifted) and np.isclose(total[0], expected)


def test_minimum_image_violations():
    H = np.array([[0.0, 0.5, 0.001], [0.5, 0.0, 0.3], [0.001, 0.3, 0.0]])
    pos = np.array(
        [[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.7, 0.1, 0.1]]
    )  # 1-2 close, 2-3 at 0.5 of the box
    bad = mol.minimum_image_violations(H, pos, tol_ev=0.01, fraction=0.45)
    assert [(i, j) for i, j, _, _ in bad] == [(1, 2)]
    assert np.isclose(bad[0][2], 0.3) and np.isclose(bad[0][3], 0.5)
    assert mol.minimum_image_violations(H, pos, tol_ev=0.01, fraction=0.6) == []


def test_molecular_summary_shapes():
    labels = [component_label((a, b, c)) for a in range(3) for b in range(3) for c in range(3)]
    ne = 3
    shg = np.zeros((27, ne), dtype=complex)
    shg[labels.index('xxx')] = 2.0
    results = {'components': labels, 'shg': shg, 'shift': np.ones((27, ne))}
    H = np.diag([-1.0, 0.5, 1.0])[:, :, None]
    pos_bohr = np.array([[-2.0, 0, 0], [0, 0, 0], [2.0, 0, 0]], dtype=float)
    out = mol.molecular_summary(results, H, pos_bohr, pos_bohr / 40.0, 1000.0, [1], 2.0)
    assert out['beta_T_au'].shape == (27, ne)
    assert out['beta_parallel_au'].shape == (ne,) and out['beta_hrs_au'].shape == (ne,)
    assert np.allclose(out['dipole_ebohr'], [4.0, 0.0, 0.0])  # both electrons on the -2 bohr site
    assert np.allclose(
        out['beta_parallel_au'], 3.0 / 5.0 * out['beta_T_au'][labels.index('xxx')].real
    )
    assert out['dmudt_si'].shape == (27, ne)
    assert out['minimum_image_violations'] == []


def test_channel_electron_counts():
    """Two channels with different fillings: the counts come from the bands below E_F = 0."""
    n = 4
    HRs = np.zeros((n, n, 1, 1, 1, 2), dtype=complex)
    HRs[:, :, 0, 0, 0, 0] = np.diag([-2.0, -1.0, 0.5, 1.0])  # 2 occupied
    HRs[:, :, 0, 0, 0, 1] = np.diag([-2.0, -1.0, -0.3, 1.0])  # 3 occupied
    assert channel_electron_counts(HRs) == [2.0, 3.0]
    assert channel_electron_counts(HRs, nelec=(1, 2)) == [1.0, 2.0]
    # a dispersive channel crossing E_F raises
    HRs_k = np.zeros((2, 2, 2, 1, 1, 2), dtype=complex)
    HRs_k[:, :, 0, 0, 0, :] = np.diag([-1.0, 0.1])[:, :, None]
    HRs_k[:, :, 1, 0, 0, 0] = np.diag([0.0, 0.2])  # H(k = 1/2) = H(0) - H(R=1): band 2 crosses E_F
    with pytest.raises(ValueError):
        channel_electron_counts(HRs_k)
