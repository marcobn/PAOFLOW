"""Long-range (dipole / quadrupole) electron-phonon model: Ewald sums, limits, symmetry, I/O."""

import numpy as np
import pytest

from PAOFLOW.elphon.long_range import LongRangeModel, read_quadrupole_fmt

pytestmark = pytest.mark.unit

AMU_RY = 911.444243  # amu in Rydberg mass units
AT_FCC = 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]])


def rocksalt_model(z=1.1, eps=3.0, alat=10.0, quadrupoles=None, **kw):
    """Two-atom cubic rigid-ion model (charges +z / -z, scalar dielectric tensor)."""
    bg = np.linalg.inv(AT_FCC).T
    tau = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    born = np.array([z * np.eye(3), -z * np.eye(3)])
    return LongRangeModel(
        eps * np.eye(3), born, tau, alat, AT_FCC, bg, quadrupoles=quadrupoles, **kw
    )


def q_along(model, direction, magnitude):
    """Crystal coordinates of a q of ``magnitude`` (2 pi / alat) along a Cartesian direction."""
    d = np.asarray(direction, dtype=float)
    return (magnitude * d / np.linalg.norm(d)) @ np.linalg.inv(model.bg)


def test_g_sum_is_converged_at_the_default_cutoff():
    model = rocksalt_model()
    q = np.array([0.2, 0.1, 0.0])
    c_default, _ = model.coefficients(q)
    c_large, _ = rocksalt_model(gmax=30.0).coefficients(q)
    assert np.abs(c_default - c_large).max() / np.abs(c_large).max() < 1e-5
    d_default = model.dynamical_matrix(q)
    d_large = rocksalt_model(gmax=30.0).dynamical_matrix(q)
    assert np.abs(d_default - d_large).max() / np.abs(d_large).max() < 1e-4  # e^-14 truncation


def test_frohlich_limit_and_lo_to_splitting():
    z, eps = 1.1, 3.0
    model = rocksalt_model(z, eps)
    masses = np.array([23.0, 35.45]) * AMU_RY
    mu = 1.0 / (1.0 / masses[0] + 1.0 / masses[1])
    lst = 8.0 * np.pi * z**2 / (model.omega * mu * eps)  # 4 pi e^2 Z*^2 / (Omega mu eps), e^2 = 2
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses, 3))
    for qmag in (1e-3, 1e-4):
        q = q_along(model, [1, 0, 0], qmag)
        coeff, coeff_r = model.coefficients(q)
        assert coeff_r is None
        frohlich = 8.0 * np.pi / model.omega * z / (eps * qmag * model.tpiba)
        assert abs(abs(coeff[0]) / frohlich - 1.0) < 1e-5  # atom 1, x: 1/|q| divergence
        assert abs(coeff[1]) < 1e-6 * abs(coeff[0])  # transverse displacements do not couple
        dyn = model.dynamical_matrix(q)
        w2 = np.linalg.eigvalsh(0.5 * (dyn + dyn.conj().T) * np.outer(inv_sqrt_m, inv_sqrt_m))
        # three acoustic modes at zero (the self term is the acoustic sum rule); the G != 0 sum
        # shifts LO and TO alike, so LO - TO = Lyddane-Sachs-Teller and the TO pair is degenerate
        acoustic = np.abs(w2) < 1e-4 * lst
        assert acoustic.sum() == 3
        optical = np.sort(w2[~acoustic])
        assert abs((optical[2] - optical[1]) / lst - 1.0) < 1e-5
        assert abs(optical[1] - optical[0]) < 1e-8 * lst


def test_dynamical_matrix_is_hermitian_and_covariant():
    model = rocksalt_model()
    q = np.array([0.3, 0.1, 0.2])
    dyn = model.dynamical_matrix(q)
    assert np.abs(dyn - dyn.conj().T).max() < 1e-12 * np.abs(dyn).max()
    # a cubic rotation about the origin maps both atoms onto themselves: D(Rq) = R D(q) R^T blockwise
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90 deg about z
    q_cart = q @ model.bg
    q_rot = (rot @ q_cart) @ np.linalg.inv(model.bg)
    big = np.kron(np.eye(2), rot)
    assert np.allclose(
        model.dynamical_matrix(q_rot), big @ dyn @ big.T, atol=1e-10 * np.abs(dyn).max()
    )
    c, _ = model.coefficients(q)
    c_rot, _ = model.coefficients(q_rot)
    assert np.allclose(c_rot, big @ c, atol=1e-10 * np.abs(c).max())


def test_quadrupole_term_is_finite_at_small_q():
    quad = np.zeros((2, 3, 3, 3))
    for a in range(3):
        for p in range(3):
            if p != a:
                quad[0, a, a, p] = quad[0, a, p, a] = 2.0  # Q_{a, ap} = Q_{a, pa}: tetrahedral-like
                quad[1, a, a, p] = quad[1, a, p, a] = -2.0
    z0 = np.zeros((2, 3, 3))
    bg = np.linalg.inv(AT_FCC).T
    tau = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    model = LongRangeModel(12.0 * np.eye(3), z0, tau, 10.0, AT_FCC, bg, quadrupoles=quad)
    assert model.has_quadrupoles and not model.is_polar()
    small = np.abs(model.coefficients(q_along(model, [1, 1, 0], 1e-3))[0]).max()
    smaller = np.abs(model.coefficients(q_along(model, [1, 1, 0], 1e-4))[0]).max()
    assert small > 0 and abs(small / smaller - 1.0) < 1e-3  # |q|^0 at small q
    coeff, coeff_r = model.coefficients(np.array([0.1, 0.2, 0.0]))
    assert coeff_r is not None and coeff_r.shape == (3, 6)
    dyn = model.dynamical_matrix(np.array([0.1, 0.2, 0.0]))
    assert np.abs(dyn - dyn.conj().T).max() < 1e-12 * np.abs(dyn).max()
    lr = model.vertex_pao(
        np.array([0.1, 0.2, 0.0]), 4, orbital_positions=np.array([tau[0]] * 2 + [tau[1]] * 2)
    )
    assert lr.shape == (4, 4, 6) and np.abs(lr - np.einsum('ijc,ij->ijc', lr, np.eye(4))).max() == 0


def test_vertex_pao_is_diagonal_and_identical_for_every_orbital_without_quadrupoles():
    model = rocksalt_model()
    lr = model.vertex_pao(np.array([0.25, 0.0, 0.0]), 5)
    assert lr.shape == (5, 5, 6)
    assert np.allclose(lr[np.arange(5), np.arange(5)], lr[0, 0][None, :])
    assert np.abs(lr - np.einsum('ijc,ij->ijc', lr, np.eye(5))).max() == 0
    assert (
        np.abs(lr).max() == 0 or np.abs(model.vertex_pao(np.zeros(3), 5)).max() < np.abs(lr).max()
    )


def test_quadrupole_fmt_round_trip(tmp_path):
    lines = ['atom dir Qxx Qyy Qzz Qyz Qxz Qxy']
    values = {}
    rng = np.random.default_rng(1)
    for na in (1, 2):
        for idir in (1, 2, 3):
            v = np.round(rng.normal(size=6), 4)
            values[(na, idir)] = v
            lines.append('%d %d ' % (na, idir) + ' '.join('%.4f' % x for x in v))
    path = tmp_path / 'quadrupole.fmt'
    path.write_text('\n'.join(lines) + '\n')
    quad = read_quadrupole_fmt(str(path), 2)
    for (na, idir), (qxx, qyy, qzz, qyz, qxz, qxy) in values.items():
        block = quad[na - 1, idir - 1]
        assert np.allclose(block, block.T)
        assert np.allclose(np.diag(block), [qxx, qyy, qzz])
        assert np.allclose([block[1, 2], block[0, 2], block[0, 1]], [qyz, qxz, qxy])
    with pytest.raises(ValueError):
        read_quadrupole_fmt(str(path), 3)


def test_signature_tracks_the_parameters():
    a, b = rocksalt_model(), rocksalt_model(alpha=0.5)
    assert a.signature() == rocksalt_model().signature()
    assert a.signature() != b.signature()


# ----------------------------------------------------------------------------- hooks
def test_vertex_pao_r_subtracts_the_model_in_the_home_cell_only():
    from PAOFLOW.elphon.elph_bloch import kq_index_map, vertex_pao_R

    rng = np.random.default_rng(5)
    model = rocksalt_model()
    ng = (2, 2, 2)
    nk, nbnd, nawf, ncart = 8, 3, 4, 6
    kpts = np.stack(np.meshgrid(*[np.arange(n) for n in ng], indexing='ij'), -1).reshape(-1, 3) / ng
    kidx = np.round(kpts * ng).astype(int)
    q = np.array([0.5, 0.0, 0.0])
    ikq, _ = kq_index_map(kpts, q, ng)
    d = rng.normal(size=(nk, nbnd, nbnd, ncart)) + 1j * rng.normal(size=(nk, nbnd, nbnd, ncart))
    A = rng.normal(size=(nbnd, nawf, nk)) + 1j * rng.normal(size=(nbnd, nawf, nk))
    lr = model.vertex_pao(q, nawf)
    full = vertex_pao_R(d, A, ikq, kidx, ng)
    short = vertex_pao_R(d, A, ikq, kidx, ng, long_range=lr)
    diff = full - short  # a k-independent matrix transforms to the R_e = 0 cell only
    assert np.allclose(diff[..., 0, 0, 0], lr, atol=1e-12)
    diff[..., 0, 0, 0] = 0.0
    assert np.abs(diff).max() < 1e-12


def test_band_vertex_restores_the_model_after_the_rotations():
    from PAOFLOW.elphon.elph_bloch import band_vertex

    rng = np.random.default_rng(6)
    nb, nws, nawf, ncart, nmode = 3, 5, 4, 6, 6
    gn_flat = rng.normal(size=(nws, nawf * nawf * ncart)) + 0j
    phkg = np.exp(2j * np.pi * rng.uniform(size=(nb, nws)))
    Vkq = rng.normal(size=(nb, nawf, nawf)) + 1j * rng.normal(size=(nb, nawf, nawf))
    Vk = rng.normal(size=(nb, nawf, nawf)) + 1j * rng.normal(size=(nb, nawf, nawf))
    zmass = rng.normal(size=(nmode, ncart))
    lr = rocksalt_model().vertex_pao(np.array([0.1, 0.2, 0.0]), nawf)
    plain = band_vertex(gn_flat, phkg, Vkq, Vk, zmass)
    with_lr = band_vertex(gn_flat, phkg, Vkq, Vk, zmass, long_range=lr)
    added = np.einsum('bim,ijc,vc,bjn->bvmn', np.conj(Vkq), lr, zmass, Vk)
    assert np.allclose(with_lr - plain, added, atol=1e-12)


def _rocksalt_force_constants(model, masses_ry, qgrid, spring=0.02):
    """Coarse C(q): the long-range model plus a nearest-neighbour spring between the two atoms."""
    nat = 2
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in qgrid], indexing='ij'), -1).reshape(-1, 3)
    neighbours = np.array([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, -1, -1]])
    cgrid = np.zeros(tuple(qgrid) + (3 * nat, 3 * nat), dtype=complex)
    for lab in grid:
        q = lab / np.asarray(qgrid)
        c = np.zeros((2, 3, 2, 3), dtype=complex)
        phase = np.exp(
            2j * np.pi * (neighbours @ q)
        ).sum()  # sum over images of atom 2 around atom 1
        c[0, :, 1, :] = -spring * phase * np.eye(3)
        c[1, :, 0, :] = -spring * np.conj(phase) * np.eye(3)
        c[0, :, 0, :] = spring * len(neighbours) * np.eye(3)
        c[1, :, 1, :] = spring * len(neighbours) * np.eye(3)
        cgrid[tuple(lab)] = c.reshape(6, 6) + model.dynamical_matrix(q)
    return cgrid


def test_phonon_interpolation_restores_the_lo_to_splitting():
    from PAOFLOW.elphon.do_pao_eph_dense_q import _phonon_interp_from_force_constants
    from PAOFLOW.elphon.elph_bloch import RY_TO_THZ

    model = rocksalt_model(z=1.1, eps=3.0)
    masses_ry = np.array([23.0, 35.45]) * AMU_RY
    qgrid = (3, 3, 3)
    cgrid = _rocksalt_force_constants(model, masses_ry, qgrid)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(masses_ry, 3))
    with_model = _phonon_interp_from_force_constants(
        cgrid, masses_ry, qgrid, at=AT_FCC, long_range=model
    )
    plain = _phonon_interp_from_force_constants(cgrid, masses_ry, qgrid, at=AT_FCC)
    assert with_model.long_range is model and plain.long_range is None
    # exact on the coarse grid (the acoustic sum rule of the interpolator is applied at Gamma only)
    lab = np.array([1, 2, 0])
    q = lab / np.asarray(qgrid)
    direct = np.linalg.eigvalsh(
        0.5 * (cgrid[tuple(lab)] + cgrid[tuple(lab)].conj().T) * np.outer(inv_sqrt_m, inv_sqrt_m)
    )
    direct = np.sign(direct) * np.sqrt(np.abs(direct)) * RY_TO_THZ
    freq, _ = with_model(q)
    assert np.allclose(np.sort(freq), np.sort(direct), atol=1e-8)
    # near Gamma the plain interpolation collapses LO onto TO; the model keeps the splitting
    q_small = q_along(model, [1, 0, 0], 1e-3)
    freq_model, _ = with_model(q_small)
    freq_plain, _ = plain(q_small)
    optical_model = np.sort(freq_model)[-3:]
    optical_plain = np.sort(freq_plain)[-3:]
    assert optical_model[2] - optical_model[1] > 0.1 * optical_model[2]
    assert abs(optical_plain[2] - optical_plain[1]) < 0.02 * optical_plain[2]


def test_polar_check_accepts_a_model_and_warns_for_missing_quadrupoles():
    from PAOFLOW.elphon.do_pao_eph import check_long_range_terms

    polar = rocksalt_model()
    eps = polar.epsilon
    with pytest.raises(NotImplementedError):
        check_long_range_terms(polar.born_charges, eps)
    check_long_range_terms(polar.born_charges, eps, long_range=polar)  # no raise, no warning
    nonpolar = LongRangeModel(
        eps, np.zeros((2, 3, 3)), polar.tau_alat, polar.alat, polar.at, polar.bg
    )
    with pytest.warns(UserWarning, match='quadrupoles'):
        check_long_range_terms(nonpolar.born_charges, eps, long_range=nonpolar)
