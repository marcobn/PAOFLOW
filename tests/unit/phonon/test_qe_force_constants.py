"""q2r.x + matdyn.x from QE dynamical matrices (Si, ph.x on a 3x3x3 q-grid)."""

import os

import numpy as np
import pytest

from PAOFLOW.elphon.qe_matdyn import read_qe_ifc
from PAOFLOW.phonon.qe_force_constants import (
    THZ_TO_CM1,
    is_polar,
    phonon_density_of_states,
    phonon_interpolator,
    q2r_matdyn,
    read_dynamical_matrices,
    real_space_force_constants,
    write_qe_force_constants,
)

DATA = os.path.join(os.path.dirname(__file__), 'data', 'si_dyn')

# QE 7.5 q2r.x (zasr='simple') + matdyn.x (asr='simple') on the same files, cm^-1;
# q in Cartesian 2 pi / alat.
QE_Q_CART = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5], [0.123, 0.456, 0.789]])
QE_FREQ_CM1 = np.array([
    [0.0, 0.0, 0.0, 503.7166, 503.7166, 503.7166],
    [128.8731, 128.8731, 398.0933, 398.0933, 439.3614, 439.3614],
    [106.0762, 106.0762, 372.8260, 392.7676, 475.0475, 475.0475],
    [157.4828, 200.2152, 330.5239, 378.4607, 456.2530, 466.4807],
])  # fmt: skip


@pytest.fixture(scope='module')
def silicon():
    return read_dynamical_matrices('Si2', DATA)


def test_read_dynamical_matrices(silicon):
    assert silicon['qgrid'] == (3, 3, 3) and silicon['nat'] == 2 and silicon['ibrav'] == 2
    assert silicon['force_constants_q'].shape == (3, 3, 3, 6, 6)
    np.testing.assert_allclose(np.diag(silicon['epsilon']), 12.968650486075)
    assert not silicon['polar']  # Z* ~ 3e-4: numerical noise
    np.testing.assert_allclose(silicon['tau_cryst'][1] % 1.0, [0.25, 0.25, 0.25], atol=1e-10)
    np.testing.assert_allclose(silicon['at'] @ silicon['bg'].T, np.eye(3), atol=1e-12)


def test_frequencies_match_qe_matdyn(silicon):
    phonon_at_q = phonon_interpolator(
        real_space_force_constants(silicon['force_constants_q']), silicon['masses_ry'],
        silicon['qgrid'], silicon['at'], atom_positions=silicon['tau_cryst'],
    )  # fmt: skip
    q_cryst = QE_Q_CART @ silicon['at'].T
    frequencies, eigenvectors = phonon_at_q(q_cryst)
    np.testing.assert_allclose(frequencies * THZ_TO_CM1, QE_FREQ_CM1, atol=2e-3)
    single, vectors = phonon_at_q(q_cryst[3])
    np.testing.assert_allclose(single, frequencies[3], rtol=1e-12)
    np.testing.assert_allclose(vectors @ vectors.conj().T, np.eye(6), atol=1e-12)


def test_force_constant_file_reproduces_grid_matrices(silicon, tmp_path):
    force_constants_r = real_space_force_constants(silicon['force_constants_q'])
    path = write_qe_force_constants(str(tmp_path / 'Si2.fc'), silicon, force_constants_r)
    ifc = read_qe_ifc(path)
    assert ifc['nr'] == [3, 3, 3] and ifc['ibrav'] == 2
    np.testing.assert_allclose(ifc['masses_amu'], 28.085, rtol=1e-4)
    # QE convention: C(q) = sum_m Phi(m) exp(-2 pi i q.m), exact on the grid
    cells = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)
    phi = ifc['frc'].reshape(3, 3, 2, 2, -1)
    for label in [(0, 0, 0), (1, 0, 2), (2, 2, 1)]:
        q = np.array(label) / 3.0
        phase = np.exp(-2j * np.pi * cells @ q)
        C = np.einsum('abijm,m->iajb', phi, phase).reshape(6, 6)
        np.testing.assert_allclose(C, silicon['force_constants_q'][label], atol=1e-10)


def test_q2r_matdyn_outputs(tmp_path):
    result = q2r_matdyn('Si2', DATA, outdir=str(tmp_path), fname='si', force_constants_file='Si2.fc',
                        npoints=5, mesh=(8, 8, 8), sigma_thz=0.2, units='cm-1')  # fmt: skip
    names = sorted(os.path.basename(p) for p in result['files'])
    assert names == ['Si2.fc', 'si_band.dat', 'si_band.labels', 'si_dos.dat']
    band = np.loadtxt(tmp_path / 'si_band.dat')
    assert band.shape[1] == 7 and np.all(np.diff(band[:, 0]) >= -1e-12)
    np.testing.assert_allclose(band[0, 1:4], 0.0, atol=1e-3)  # acoustic sum rule at Gamma
    labels = open(tmp_path / 'si_band.labels').read().split('\n')[1:]
    assert labels[0].split()[1] == r'$\Gamma$'
    dos = np.loadtxt(tmp_path / 'si_dos.dat')
    np.testing.assert_allclose(np.trapezoid(dos[:, 1], dos[:, 0]), 6.0, rtol=1e-6)


def test_density_of_states_normalised(silicon):
    phonon_at_q = phonon_interpolator(
        real_space_force_constants(silicon['force_constants_q']), silicon['masses_ry'],
        silicon['qgrid'], silicon['at'], atom_positions=silicon['tau_cryst'],
    )  # fmt: skip
    dos = phonon_density_of_states(phonon_at_q, (6, 6, 6), sigma_thz=0.3)
    np.testing.assert_allclose(np.trapezoid(dos['dos'], dos['frequency_thz']), 6.0, rtol=1e-6)


def test_is_polar():
    assert not is_polar(None)
    assert not is_polar(np.full((2, 3, 3), 2.0))  # removed by charge neutrality
    charges = np.zeros((2, 3, 3))
    charges[0] = 2.0 * np.eye(3)
    charges[1] = -2.0 * np.eye(3)
    assert is_polar(charges)


# ----------------------------------------------------------------------------- polar crystal (3C-SiC)
DATA_SIC = os.path.join(os.path.dirname(__file__), 'data', 'sic_dyn')


def _matdyn_path(at):
    """The q-list of data/sic_dyn/matdyn.in (band form, Cartesian 2 pi / a): 20 + 20 + 20 + 1 points."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    q_cart = []
    for start, stop in zip(points[:-1], points[1:]):
        q_cart += [start + (stop - start) * i / 20.0 for i in range(20)]
    q_cart.append(points[-1])
    return np.array(q_cart) @ at.T  # crystal coordinates


def test_sic_dispersion_matches_matdyn_with_the_nonanalytic_term(tmp_path):
    from PAOFLOW.phonon.qe_force_constants import gamma_limit_points

    result = q2r_matdyn('sic', DATA_SIC, outdir=str(tmp_path), fname='sic', mesh=None, npoints=5)
    model = result['long_range']
    assert model is not None and model.is_polar()
    np.testing.assert_allclose(model.born_charges.sum(axis=0), 0.0, atol=1e-12)  # zasr = 'simple'
    dyn = result['dynamical_matrices']
    reference = np.loadtxt(os.path.join(DATA_SIC, 'matdyn_freq.gp'))[:, 1:]  # cm^-1
    q_cryst = gamma_limit_points(_matdyn_path(dyn['at']), dyn['bg'])
    frequencies, _ = result['phonon_at_q'](q_cryst)
    np.testing.assert_allclose(
        np.sort(frequencies, axis=1) * THZ_TO_CM1, np.sort(reference, axis=1), atol=0.05
    )
    # LO-TO splitting at Gamma along Gamma-X (matdyn: 790 / 952 cm^-1); without the model LO = TO
    gamma = np.sort(frequencies[0]) * THZ_TO_CM1
    assert gamma[-1] - gamma[-2] > 150.0
    with pytest.warns(UserWarning, match='LO-TO'):
        plain = q2r_matdyn(
            'sic', DATA_SIC, outdir=str(tmp_path / 'plain'), mesh=None, npoints=5, long_range=False
        )
    plain_gamma = np.sort(plain['phonon_at_q'](q_cryst[:1])[0][0]) * THZ_TO_CM1
    assert abs(plain_gamma[-1] - plain_gamma[-2]) < 1.0


def test_sic_force_constant_file_carries_the_tensors_and_reproduces_the_grid(tmp_path):
    result = q2r_matdyn(
        'sic',
        DATA_SIC,
        outdir=str(tmp_path),
        fname='sic',
        force_constants_file='sic.fc',
        mesh=None,
        npoints=5,
    )
    dyn, model = result['dynamical_matrices'], result['long_range']
    path = os.path.join(str(tmp_path), 'sic.fc')
    ifc = read_qe_ifc(path)
    assert ifc['has_zeu']
    lines = open(path).read().splitlines()
    flag = [ln for ln in lines if ln.strip().startswith('T')][0].split()
    np.testing.assert_allclose(float(flag[1]), model.alpha)
    np.testing.assert_allclose(model.alpha, (dyn['alat'] / (2 * np.pi)) ** 2)  # q2r.x's choice
    # short-range part from the file + the model restores the coarse matrices exactly
    cells = np.stack(np.meshgrid(*[np.arange(3)] * 3, indexing='ij'), -1).reshape(-1, 3)
    phi = ifc['frc'].reshape(3, 3, 2, 2, -1)
    for label in [(0, 0, 0), (1, 0, 2), (2, 2, 1)]:
        q = np.array(label) / 3.0
        phase = np.exp(-2j * np.pi * cells @ q)
        C = np.einsum('abijm,m->iajb', phi, phase).reshape(6, 6) + model.dynamical_matrix(q)
        np.testing.assert_allclose(C, dyn['force_constants_q'][label], atol=1e-8)
