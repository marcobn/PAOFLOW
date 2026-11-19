"""Unit tests of the special displacement method (ZG configuration)."""

import os

import numpy as np
import pytest

from PAOFLOW.phonon.special_displacement import (
    AMU_SI,
    BOHR_M,
    HBAR_SI,
    KB_SI,
    THZ,
    bose_occupation,
    configuration_mean_square_displacement,
    optimise_signs,
    sign_table,
    split_qpoints,
    supercell_qpoints,
    thermal_mean_square_displacement,
    zg_amplitudes,
    zg_displacements,
    zg_error,
    zg_sign_sequence,
)


def test_supercell_qpoints_order_and_gamma_first():
    q = supercell_qpoints((2, 3, 4))
    assert q.shape == (24, 3)
    np.testing.assert_allclose(q[0], 0.0)
    np.testing.assert_allclose(q[1], [0.0, 0.0, 0.25])  # third index fastest
    np.testing.assert_allclose(q[4], [0.0, 1.0 / 3.0, 0.0])
    assert np.all((q >= 0.0) & (q < 1.0))


@pytest.mark.parametrize(
    'dims, n_a, n_b', [((2, 2, 2), 8, 0), ((3, 3, 3), 1, 13), ((4, 4, 4), 8, 28)]
)
def test_split_qpoints_sets_a_and_b(dims, n_a, n_b):
    q_ab, is_a = split_qpoints(supercell_qpoints(dims))
    assert is_a.sum() == n_a and (~is_a).sum() == n_b
    np.testing.assert_allclose(q_ab[0], 0.0)  # Gamma first
    for q in q_ab[is_a]:
        assert np.allclose(2 * q - np.round(2 * q), 0.0)
    # one member of every +-q pair: the partner of a B point is not in the set
    for q in q_ab[~is_a]:
        partner = (-q) % 1.0
        assert not any(np.allclose(partner, p) for p in q_ab)


def test_bose_occupation_limits():
    f = np.array([1.0, 2.0])
    assert np.all(bose_occupation(f, 0.0) == 0.0)
    n = bose_occupation(f, 300.0)
    x = HBAR_SI * 2 * np.pi * f * THZ / (KB_SI * 300.0)
    np.testing.assert_allclose(n, 1.0 / np.expm1(x))
    assert bose_occupation(np.array([-1.0, 0.0]), 300.0).tolist() == [0.0, 0.0]


def test_zg_amplitudes_zero_point_classical_limit_and_frozen_modes():
    f = np.array([[0.0, 0.0, 0.0, 2.0], [1.0, 1.5, 2.0, 3.0]])
    s0 = zg_amplitudes(f, 0.0, is_gamma=np.array([True, False]))
    omega = 2 * np.pi * 2.0 * THZ
    np.testing.assert_allclose(s0[0, 3], np.sqrt(HBAR_SI / (2 * omega * AMU_SI)) / BOHR_M)
    assert np.all(s0[0, :3] == 0.0)  # Gamma acoustic modes frozen (zero frequency anyway)
    # three lowest modes at Gamma are frozen even when finite
    g = zg_amplitudes(np.array([[0.5, 0.6, 0.7, 2.0]]), 0.0, is_gamma=np.array([True]))
    assert np.all(g[0, :3] == 0.0) and g[0, 3] > 0.0
    # classical limit: sigma^2 -> 2 k T / (m_u omega^2)
    T = 1.0e5
    s = zg_amplitudes(f, T)[1]
    omega = 2 * np.pi * f[1] * THZ
    np.testing.assert_allclose(s**2, KB_SI * T / (AMU_SI * omega**2) / BOHR_M**2, rtol=1.0e-2)
    # soft/imaginary modes frozen
    assert zg_amplitudes(np.array([[-0.5, 1.0e-5, 1.0]]), 300.0)[0, :2].tolist() == [0.0, 0.0]


def test_sign_table_binary_pattern_and_antithetic_rows():
    t = sign_table(3)
    assert t.shape == (8, 3) and set(np.unique(t).tolist()) == {-1, 1}
    pattern = np.array(
        [
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [1, -1, -1],
            [-1, 1, 1],
            [-1, 1, -1],
            [-1, -1, 1],
            [-1, -1, -1],
        ]
    )
    expected = pattern.copy()
    expected[1::2] *= -1
    np.testing.assert_array_equal(t, expected)
    assert sign_table(13).shape == (2 * 2**11, 13)  # capped at 2^11 rows per half


def test_sign_sequence_cycles_and_negates_every_block():
    nmode, nq = 3, 10
    s = zg_sign_sequence(nmode, nq)
    assert s.shape == (nq, nmode) and set(np.unique(s).tolist()) <= {-1, 1}
    # the first block of pn = 4 q-points uses four distinct rows; the next block is the
    # rotated, negated set
    rows = {tuple(r) for r in s[:4]}
    assert len(rows) == 4
    rows_next = {tuple(-r) for r in s[4:8]}
    assert rows_next == rows
    perm = np.array([3, 2, 1, 0])
    s2 = zg_sign_sequence(nmode, nq, perm)
    assert {tuple(r) for r in s2[:4]} == rows  # a permutation of the same rows


def _random_modes(nq, nat, rng, real=False):
    nmode = 3 * nat
    eig = np.empty((nq, nat, 3, nmode), complex)
    for iq in range(nq):
        a = rng.standard_normal((nmode, nmode)) + (
            0.0 if real else 1j * rng.standard_normal((nmode, nmode))
        )
        h = a + a.conj().T
        _, v = np.linalg.eigh(h)
        eig[iq] = v.reshape(nat, 3, nmode)
    freq = rng.uniform(1.0, 3.0, size=(nq, nmode))
    return freq, eig


def test_zg_error_vanishes_without_cross_terms_and_optimisation_reduces_it():
    rng = np.random.default_rng(1)
    nq, nat = 14, 2
    freq, eig = _random_modes(nq, nat, rng)
    sigma = zg_amplitudes(freq, 300.0)
    # eigenvectors = identity: every component carried by one branch -> no cross terms
    ident = np.tile(np.eye(3 * nat).reshape(nat, 3, 3 * nat), (nq, 1, 1, 1)).astype(complex)
    total, ratio = zg_error(np.ones((nq, 3 * nat), int), sigma, ident)
    assert total == pytest.approx(0.0, abs=1.0e-12)
    signs, err, attempts = optimise_signs(sigma, eig, niters=200, error_thresh=0.05, seed=0)
    err_plus, _ = zg_error(np.ones((nq, 3 * nat), int), sigma, eig)
    assert err <= err_plus and attempts >= 1
    assert zg_error(signs, sigma, eig)[0] == pytest.approx(err)


@pytest.mark.parametrize('dims', [(3, 3, 3), (4, 4, 4)])
def test_configuration_msd_equals_thermal_msd_times_one_plus_error(dims):
    """Exact identity of the construction: <u^2>_config = <u^2>_thermal (1 + ratio)."""
    rng = np.random.default_rng(7)
    q_ab, is_a = split_qpoints(supercell_qpoints(dims))
    nat, masses = 1, np.array([207.2])
    nq = len(q_ab)
    freq, eig = _random_modes(nq, nat, rng)
    freq_a, eig_a = _random_modes(nq, nat, rng, real=True)
    eig[is_a] = eig_a[is_a]  # real eigenvectors at the time-reversal-invariant points
    sigma = zg_amplitudes(freq, 300.0, is_gamma=np.array([np.allclose(q, 0.0) for q in q_ab]))
    signs, err, _ = optimise_signs(sigma, eig, niters=20, error_thresh=0.3, seed=3, is_a=is_a)
    cells = np.array(
        [[i, j, k] for i in range(dims[0]) for j in range(dims[1]) for k in range(dims[2])]
    )
    s2p = np.zeros(len(cells), int)
    u = zg_displacements(q_ab, is_a, eig, sigma, signs, masses, cells, s2p, len(cells))
    msd_c = configuration_mean_square_displacement(u, s2p, nat)
    msd_t = thermal_mean_square_displacement(is_a, eig, sigma, masses, len(cells))
    _, ratio = zg_error(signs, sigma, eig, is_a=is_a)
    np.testing.assert_allclose(
        msd_c, msd_t * (1.0 + ratio.reshape(nat, 3)), rtol=1.0e-9, atol=1.0e-14
    )
    assert np.all(msd_t > 0.0)


def test_zg_configuration_with_phonopy(tmp_path):
    phonopy = pytest.importorskip('phonopy')  # noqa: F841
    from PAOFLOW.phonon.do_phonopy import init_phonopy
    from PAOFLOW.phonon.special_displacement import zg_configuration

    class _Stub:
        def __init__(self, arry, attr):
            self._arry, self._attr, self.rank = arry, attr, 0

        def data_dicts(self):
            return self._arry, self._attr

    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
    arry = {'a_vectors': a_vectors, 'tau': np.zeros((1, 3)), 'atoms': ['Al']}
    attr = {
        'alat': 7.6326928726,
        'natoms': 1,
        'opath': str(tmp_path),
        'phonon_supercell_matrix': 3,
        'verbose': False,
    }
    phonon = init_phonopy(_Stub(arry, attr))
    nsc = len(phonon.supercell)
    fc = np.zeros((nsc, nsc, 3, 3))
    pos = phonon.supercell.positions
    cell = phonon.supercell.cell
    # nearest-neighbour springs (minimum image) -> a stable, translationally invariant fc2
    for i in range(nsc):
        for j in range(i + 1, nsc):
            d = pos[j] - pos[i]
            frac = np.linalg.solve(cell.T, d)
            frac -= np.round(frac)
            if np.linalg.norm(frac @ cell) < 0.75 * 7.6326928726:
                k = 0.05 * np.eye(3)
                fc[i, i] += k
                fc[j, j] += k
                fc[i, j] -= k
                fc[j, i] -= k
    phonon.force_constants = fc
    out = zg_configuration(phonon, 3, 300.0, niters=30, error_thresh=0.3, seed=0)
    assert out['displacements'].shape == (nsc, 3) and np.all(np.isfinite(out['displacements']))
    np.testing.assert_allclose(
        out['cell'].positions, phonon.supercell.positions + out['displacements']
    )
    assert out['q_ab'].shape[0] == 14 and out['is_a'].sum() == 1
    # the configuration reproduces the thermal mean-square displacement within the error
    rel = np.abs(out['msd_configuration'] - out['msd_thermal']) / out['msd_thermal']
    assert np.all(
        rel <= out['error'] + 1.0e-9 + 3 * np.abs(rel).max() * 0
    )  # identity holds per component
    out0 = zg_configuration(phonon, 3, 0.0, niters=5, seed=0)
    assert np.linalg.norm(out0['displacements']) > 0.0  # zero-point motion
    assert np.linalg.norm(out['displacements']) > np.linalg.norm(out0['displacements'])


class _StubController:
    def __init__(self, arry, attr):
        self._arry, self._attr, self.rank = arry, attr, 0

    def data_dicts(self):
        return self._arry, self._attr


def _aluminium_controller(tmp_path, supercell_matrix=2):
    alat = 7.6326928726
    a_vectors = np.array([[-0.5, 0.0, 0.5], [0.0, 0.5, 0.5], [-0.5, 0.5, 0.0]])
    arry = {
        'a_vectors': a_vectors,
        'tau': np.zeros((1, 3)),
        'atoms': ['Al'],
        'species': [('Al', 'Al.upf')],
    }
    attr = {
        'alat': alat, 'natoms': 1, 'omega': alat**3 / 4.0, 'opath': str(tmp_path), 'savedir': 'Al1.save',
        'fpath': str(tmp_path), 'ecutwfc': 44.0, 'nspin': 1, 'insulator': False, 'degauss': 0.05,
        'nk1': 12, 'nk2': 12, 'nk3': 12, 'phonon_supercell_matrix': supercell_matrix, 'verbose': False,
    }  # fmt: skip
    return _StubController(arry, attr)


def _spring_force_constants(phonon, k=0.05, cutoff=0.75 * 7.6326928726):
    nsc = len(phonon.supercell)
    fc = np.zeros((nsc, nsc, 3, 3))
    pos, cell = phonon.supercell.positions, phonon.supercell.cell
    for i in range(nsc):
        for j in range(i + 1, nsc):
            frac = np.linalg.solve(cell.T, pos[j] - pos[i])
            frac -= np.round(frac)
            if np.linalg.norm(frac @ cell) < cutoff:
                fc[i, i] += k * np.eye(3)
                fc[j, j] += k * np.eye(3)
                fc[i, j] -= k * np.eye(3)
                fc[j, i] -= k * np.eye(3)
    return fc


def test_zg_supercell_independent_of_the_force_constant_supercell(tmp_path):
    pytest.importorskip('phonopy')
    from PAOFLOW.phonon.do_phonopy import init_phonopy
    from PAOFLOW.phonon.special_displacement import zg_configuration

    phonon = init_phonopy(_aluminium_controller(tmp_path, supercell_matrix=2))
    phonon.force_constants = _spring_force_constants(phonon)
    out = zg_configuration(
        phonon, 3, 300.0, niters=20, seed=0
    )  # 3x3x3 ZG cell from 2x2x2 force constants
    assert len(out['cell']) == 27 and out['dims'] == (3, 3, 3) and out['q_ab'].shape[0] == 14
    np.testing.assert_allclose(
        out['cell'].positions - out['reference'].positions, out['displacements']
    )
    rel = np.abs(out['msd_configuration'] - out['msd_thermal']) / out['msd_thermal']
    assert np.all(rel <= out['error'] + 1.0e-9)


def test_write_zg_inputs_writes_files_manifest_and_npz(tmp_path):
    pytest.importorskip('phonopy')
    import json

    from PAOFLOW.phonon.do_phonopy import init_phonopy
    from PAOFLOW.phonon.special_displacement import MANIFEST, write_zg_inputs, zg_configuration

    dc = _aluminium_controller(tmp_path, supercell_matrix=2)
    phonon = init_phonopy(dc)
    phonon.force_constants = _spring_force_constants(phonon)
    confs = [zg_configuration(phonon, 2, T, niters=5, seed=0) for T in (0.0, 300.0)]
    paths = write_zg_inputs(dc, confs, zg_dir='zg', nbnd=40)
    zdir = tmp_path / 'zg'
    assert [os.path.basename(p) for p in paths] == ['Al1_zg_T0K.in', 'Al1_zg_T300K.in']
    text = (zdir / 'Al1_zg_T300K.in').read_text()
    assert "calculation = 'scf'" in text and 'nbnd = 40' in text and 'nat = 8' in text
    assert (
        'K_POINTS automatic\n 6 6 6 0 0 0' in text
    )  # unit-cell 12^3 grid divided by the dimensions
    manifest = json.loads((zdir / MANIFEST).read_text())
    assert manifest['dims'] == [2, 2, 2] and manifest['natoms'] == 8 and manifest['nbnd'] == 40
    assert [c['temperature_k'] for c in manifest['configurations']] == [0.0, 300.0]
    assert (
        manifest['configurations'][1]['rms_displacement_bohr']
        > manifest['configurations'][0]['rms_displacement_bohr']
    )
    npz = np.load(zdir / 'Al1_zg_T300K.npz')
    np.testing.assert_allclose(npz['positions'] - npz['reference_positions'], npz['displacements'])
    assert npz['signs'].shape == (confs[1]['q_ab'].shape[0], 3)
    # the undisplaced reference cell: no temperature, zero displacements, its own prefix
    from PAOFLOW.phonon.special_displacement import reference_configuration

    ref = reference_configuration(confs[1])
    assert ref['temperature_k'] is None and ref['error'] == 0.0
    np.testing.assert_allclose(ref['displacements'], 0.0)
    paths = write_zg_inputs(dc, confs + [ref], zg_dir='zg_ref', nbnd=40)
    assert os.path.basename(paths[-1]) == 'Al1_zg_ref.in'
    manifest = json.loads((tmp_path / 'zg_ref' / MANIFEST).read_text())
    assert manifest['configurations'][-1]['temperature_k'] is None
    assert manifest['configurations'][-1]['rms_displacement_bohr'] == 0.0
    npz = np.load(tmp_path / 'zg_ref' / 'Al1_zg_ref.npz')
    np.testing.assert_allclose(npz['positions'], npz['reference_positions'])


def test_spectral_function_normalisation_and_peak_width():
    from PAOFLOW.elphon.workflows.thermal import (
        peak_width_at_crossing,
        spectral_function_from_weights,
    )

    grid = np.linspace(-0.5, 0.5, 2001)
    e = np.array([[-0.2, 0.1, 0.3], [0.0, 0.05, 0.4]])
    w = np.array([[1.0, 0.5, 0.0], [0.7, 0.3, 1.0]])
    a = spectral_function_from_weights(e, w, grid, 0.01)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    np.testing.assert_allclose(trapezoid(a, grid, axis=1), w.sum(axis=1), rtol=1.0e-6)
    assert grid[np.argmax(a[0])] == pytest.approx(-0.2, abs=1.0e-3)
    kd = np.linspace(0.0, 1.0, 5)
    E_pc = np.stack([np.linspace(-0.2, 0.2, 5), np.linspace(0.3, 0.5, 5)], axis=1)
    e_sc = np.repeat(E_pc[:, :1], 2, axis=1)
    res = {
        'kdist': kd,
        'energy_ev': grid,
        'spectral': spectral_function_from_weights(e_sc, np.full(e_sc.shape, 0.5), grid, 0.02),
        'E_pc': E_pc,
    }
    pk = peak_width_at_crossing(res, energy_ev=0.0)
    assert pk['band'] == 0 and pk['k_index'] == 2 and pk['E_pc_ev'] == 0.0
    assert pk['fwhm_ev'] == pytest.approx(2.3548 * 0.02, rel=0.05)
    assert pk['weight'] == pytest.approx(1.0, rel=0.02)
    assert pk['shift_ev'] == pytest.approx(0.0, abs=1.0e-3)
    # the path discretisation does not enter: a state 30 meV above a band sampled 45 meV off E_F
    E_off = E_pc + 0.055
    res_off = dict(
        res,
        E_pc=E_off,
        spectral=spectral_function_from_weights(
            np.repeat(E_off[:, :1], 2, axis=1) + 0.03, np.full(e_sc.shape, 0.5), grid, 0.02
        ),
    )
    pk = peak_width_at_crossing(res_off, energy_ev=0.0, half_window_ev=0.1)
    assert pk['k_index'] == 1 and pk['E_pc_ev'] == pytest.approx(-0.045)
    assert pk['shift_ev'] == pytest.approx(0.03, abs=1.0e-3)


def test_write_zg_inputs_accumulates_the_manifest(tmp_path):
    pytest.importorskip('phonopy')
    import json

    from PAOFLOW.phonon.do_phonopy import init_phonopy
    from PAOFLOW.phonon.special_displacement import MANIFEST, write_zg_inputs, zg_configuration

    dc = _aluminium_controller(tmp_path, supercell_matrix=2)
    phonon = init_phonopy(dc)
    phonon.force_constants = _spring_force_constants(phonon)
    first = [zg_configuration(phonon, 2, 100.0, niters=5, seed=0)]
    write_zg_inputs(dc, first, zg_dir='zg', nbnd=40)
    second = [zg_configuration(phonon, 2, 300.0, niters=5, seed=0)]
    write_zg_inputs(dc, second, zg_dir='zg', nbnd=40)
    manifest = json.loads((tmp_path / 'zg' / MANIFEST).read_text())
    temps = sorted(e['temperature_k'] for e in manifest['configurations'])
    assert temps == [100.0, 300.0]
    # Rewriting a temperature replaces its entry instead of duplicating it.
    write_zg_inputs(dc, second, zg_dir='zg', nbnd=40)
    manifest = json.loads((tmp_path / 'zg' / MANIFEST).read_text())
    assert sorted(e['temperature_k'] for e in manifest['configurations']) == [100.0, 300.0]
