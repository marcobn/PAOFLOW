"""Unit tests for :mod:`PAOFLOW.inputs.lattice_format`.

These tests pin down the QE ``latgen`` lattice-vector conventions for every
supported ``ibrav`` against hardcoded reference vectors (Bohr) derived from
the Quantum ESPRESSO ``INPUT_PW`` documentation, so the conversion is
validated without any QE runtime dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from PAOFLOW.inputs.lattice_format import (
    BOHR_RADIUS_ANGS,
    bravais_to_ibrav,
    cell_lengths_angles,
    celldm_from_namelist,
    lattice_format_QE,
    qe_ibrav_from_lattice,
)

# Common celldm parameters reused across the orthorhombic/monoclinic/triclinic
# reference cells.
A = 5.0
BOA = 1.3  # b / a
COA = 1.7  # c / a
B = BOA * A
C = COA * A


def _ref(ibrav):
    """Hardcoded reference lattice vectors (rows, Bohr) per QE INPUT_PW."""
    if ibrav == 1:  # simple cubic
        return A * np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    if ibrav == 2:  # fcc
        return (A / 2.0) * np.array([[-1, 0, 1], [0, 1, 1], [-1, 1, 0]], dtype=float)
    if ibrav == 3:  # bcc
        return (A / 2.0) * np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1]], dtype=float)
    if ibrav == -3:  # bcc, symmetric axes
        return (A / 2.0) * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
    if ibrav == 4:  # hexagonal
        return A * np.array([[1, 0, 0], [-0.5, np.sqrt(3.0) / 2.0, 0], [0, 0, COA]], dtype=float)
    if ibrav == 6:  # tetragonal P
        return A * np.array([[1, 0, 0], [0, 1, 0], [0, 0, COA]], dtype=float)
    if ibrav == 7:  # tetragonal I
        return (A / 2.0) * np.array([[1, -1, COA], [1, 1, COA], [-1, -1, COA]], dtype=float)
    if ibrav == 8:  # orthorhombic P
        return np.array([[A, 0, 0], [0, B, 0], [0, 0, C]], dtype=float)
    if ibrav == 9:  # base-centred orthorhombic, C-type
        return np.array([[A / 2.0, B / 2.0, 0], [-A / 2.0, B / 2.0, 0], [0, 0, C]], dtype=float)
    if ibrav == -9:  # base-centred orthorhombic, C-type (alternate)
        return np.array([[A / 2.0, -B / 2.0, 0], [A / 2.0, B / 2.0, 0], [0, 0, C]], dtype=float)
    if ibrav == 91:  # base-centred orthorhombic, A-type
        return np.array([[A, 0, 0], [0, B / 2.0, -C / 2.0], [0, B / 2.0, C / 2.0]], dtype=float)
    if ibrav == 10:  # face-centred orthorhombic
        return np.array(
            [
                [A / 2.0, 0, C / 2.0],
                [A / 2.0, B / 2.0, 0],
                [0, B / 2.0, C / 2.0],
            ],
            dtype=float,
        )
    if ibrav == 11:  # body-centred orthorhombic
        return np.array(
            [
                [A / 2.0, B / 2.0, C / 2.0],
                [-A / 2.0, B / 2.0, C / 2.0],
                [-A / 2.0, -B / 2.0, C / 2.0],
            ],
            dtype=float,
        )
    raise KeyError(ibrav)


@pytest.mark.parametrize('ibrav', [1, 2, 3, -3, 4, 6, 7, 8, 9, -9, 91, 10, 11])
def test_lattice_matches_reference(ibrav):
    celldm = np.array([A, BOA, COA, 0.0, 0.0, 0.0])
    out = lattice_format_QE(ibrav, celldm)
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, _ref(ibrav), rtol=0, atol=1e-10)


def test_trigonal_R_ibrav5():
    cg = 0.3  # cos(gamma)
    celldm = np.array([A, 0.0, 0.0, cg, 0.0, 0.0])
    out = lattice_format_QE(5, celldm)
    # All three vectors have length a; pairwise angle has cosine = cg.
    lengths = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(lengths, [A, A, A], atol=1e-10)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cos = out[i] @ out[j] / (lengths[i] * lengths[j])
        assert cos == pytest.approx(cg, abs=1e-10)


def test_trigonal_R_ibravm5():
    cg = 0.3
    celldm = np.array([A, 0.0, 0.0, cg, 0.0, 0.0])
    out = lattice_format_QE(-5, celldm)
    lengths = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(lengths, [A, A, A], atol=1e-10)
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        cos = out[i] @ out[j] / (lengths[i] * lengths[j])
        assert cos == pytest.approx(cg, abs=1e-10)


def test_monoclinic_P_unique_axis_c():
    cg = 0.2  # cos(gamma) = cos(ab)
    celldm = np.array([A, BOA, COA, cg, 0.0, 0.0])
    out = lattice_format_QE(12, celldm)
    np.testing.assert_allclose(out[0], [A, 0, 0], atol=1e-10)
    sg = np.sqrt(1 - cg * cg)
    np.testing.assert_allclose(out[1], [B * cg, B * sg, 0], atol=1e-10)
    np.testing.assert_allclose(out[2], [0, 0, C], atol=1e-10)
    # Angle between a1 and a2 has cosine cg.
    cos = out[0] @ out[1] / (np.linalg.norm(out[0]) * np.linalg.norm(out[1]))
    assert cos == pytest.approx(cg, abs=1e-10)


def test_monoclinic_P_unique_axis_b():
    cb = 0.25  # cos(beta) = cos(ac)
    celldm = np.array([A, BOA, COA, 0.0, cb, 0.0])
    out = lattice_format_QE(-12, celldm)
    np.testing.assert_allclose(out[0], [A, 0, 0], atol=1e-10)
    np.testing.assert_allclose(out[1], [0, B, 0], atol=1e-10)
    sb = np.sqrt(1 - cb * cb)
    np.testing.assert_allclose(out[2], [C * cb, 0, C * sb], atol=1e-10)


def test_monoclinic_base_centred_unique_axis_c():
    cg = 0.2
    celldm = np.array([A, BOA, COA, cg, 0.0, 0.0])
    out = lattice_format_QE(13, celldm)
    sg = np.sqrt(1 - cg * cg)
    ref = np.array([[A / 2.0, 0, -C / 2.0], [B * cg, B * sg, 0], [A / 2.0, 0, C / 2.0]])
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_monoclinic_base_centred_unique_axis_b():
    cb = 0.25
    celldm = np.array([A, BOA, COA, 0.0, cb, 0.0])
    out = lattice_format_QE(-13, celldm)
    sb = np.sqrt(1 - cb * cb)
    ref = np.array([[A / 2.0, B / 2.0, 0], [-A / 2.0, B / 2.0, 0], [C * cb, 0, C * sb]])
    np.testing.assert_allclose(out, ref, atol=1e-10)


def test_triclinic_ibrav14():
    calpha, cbeta, cgamma = 0.1, 0.2, 0.3
    celldm = np.array([A, BOA, COA, calpha, cbeta, cgamma])
    out = lattice_format_QE(14, celldm)
    la, lb, lc = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose([la, lb, lc], [A, B, C], atol=1e-10)
    # Recover the three input cosines from the vector geometry.
    assert out[1] @ out[2] / (lb * lc) == pytest.approx(calpha, abs=1e-10)
    assert out[0] @ out[2] / (la * lc) == pytest.approx(cbeta, abs=1e-10)
    assert out[0] @ out[1] / (la * lb) == pytest.approx(cgamma, abs=1e-10)


@pytest.mark.parametrize(
    'ibrav', [1, 2, 3, -3, 4, 5, -5, 6, 7, 8, 9, -9, 91, 10, 11, 12, -12, 13, -13, 14]
)
def test_all_ibrav_positive_volume(ibrav):
    celldm = np.array([A, BOA, COA, 0.3, 0.25, 0.2])
    out = lattice_format_QE(ibrav, celldm)
    vol = abs(np.linalg.det(out))
    assert vol > 0.0


def test_ibrav0_raises():
    with pytest.raises(ValueError):
        lattice_format_QE(0, np.array([A, 0, 0, 0, 0, 0]))


def test_unknown_ibrav_raises():
    with pytest.raises(ValueError):
        lattice_format_QE(99, np.array([A, 0, 0, 0, 0, 0]))


# --------------------------------------------------------------------------- #
# celldm_from_namelist                                                        #
# --------------------------------------------------------------------------- #
def test_celldm_from_celldm_keys():
    block = {
        'celldm(1)': '5.0',
        'celldm(2)': '1.3',
        'celldm(3)': '1.7',
        'celldm(4)': '0.2',
    }
    celldm = celldm_from_namelist(block, ibrav=12)
    np.testing.assert_allclose(celldm, [5.0, 1.3, 1.7, 0.2, 0.0, 0.0], atol=1e-12)


def test_celldm_fortran_exponent():
    block = {'celldm(1)': '5.0d0', 'celldm(2)': '1.0D0'}
    celldm = celldm_from_namelist(block, ibrav=8)
    assert celldm[0] == pytest.approx(5.0)
    assert celldm[1] == pytest.approx(1.0)


def test_celldm_from_ABC_matches_celldm_orthorhombic():
    # A/B/C convention (Ångström) vs equivalent celldm(i).
    a_ang = A * BOHR_RADIUS_ANGS
    block_abc = {
        'a': str(a_ang),
        'b': str(B * BOHR_RADIUS_ANGS),
        'c': str(C * BOHR_RADIUS_ANGS),
    }
    block_celldm = {
        'celldm(1)': str(A),
        'celldm(2)': str(BOA),
        'celldm(3)': str(COA),
    }
    cd_abc = celldm_from_namelist(block_abc, ibrav=8)
    cd_cd = celldm_from_namelist(block_celldm, ibrav=8)
    np.testing.assert_allclose(cd_abc, cd_cd, rtol=1e-12, atol=1e-12)
    # And the resulting lattices agree.
    np.testing.assert_allclose(
        lattice_format_QE(8, cd_abc), lattice_format_QE(8, cd_cd), atol=1e-10
    )


def test_celldm_from_ABC_unique_axis_b_uses_cosac():
    a_ang = A * BOHR_RADIUS_ANGS
    block = {
        'a': str(a_ang),
        'b': str(B * BOHR_RADIUS_ANGS),
        'c': str(C * BOHR_RADIUS_ANGS),
        'cosac': '0.25',
    }
    celldm = celldm_from_namelist(block, ibrav=-12)
    assert celldm[4] == pytest.approx(0.25)
    assert celldm[3] == 0.0


def test_celldm_from_ABC_triclinic_all_cosines():
    a_ang = A * BOHR_RADIUS_ANGS
    block = {
        'a': str(a_ang),
        'b': str(B * BOHR_RADIUS_ANGS),
        'c': str(C * BOHR_RADIUS_ANGS),
        'cosbc': '0.1',
        'cosac': '0.2',
        'cosab': '0.3',
    }
    celldm = celldm_from_namelist(block, ibrav=14)
    assert celldm[3] == pytest.approx(0.1)  # cosBC
    assert celldm[4] == pytest.approx(0.2)  # cosAC
    assert celldm[5] == pytest.approx(0.3)  # cosAB


def test_celldm_missing_raises():
    with pytest.raises(ValueError):
        celldm_from_namelist({'nat': '2'}, ibrav=1)


# ---------------------------------------------------------------------------
# Inverse map: explicit lattice -> QE ibrav + celldm
# ---------------------------------------------------------------------------

# (ibrav, celldm) cases exercising every supported Bravais form.  The negative
# axis variants describe the *same* physical lattice as their positive
# counterpart, so the inverse map is allowed to return either.
_INVERSE_CASES = [
    (1, [8.0, 0, 0, 0, 0, 0], {1}),
    (2, [8.0, 0, 0, 0, 0, 0], {2}),
    (3, [8.0, 0, 0, 0, 0, 0], {3}),
    (-3, [8.0, 0, 0, 0, 0, 0], {3, -3}),
    (4, [6.0, 0, 1.6, 0, 0, 0], {4}),
    (5, [7.0, 0, 0, 0.3, 0, 0], {5}),
    (-5, [7.0, 0, 0, 0.3, 0, 0], {5, -5}),
    (6, [6.0, 0, 1.6, 0, 0, 0], {6}),
    (7, [6.0, 0, 1.6, 0, 0, 0], {7}),
    (8, [6.0, 1.2, 1.7, 0, 0, 0], {8}),
    (9, [6.0, 1.2, 1.7, 0, 0, 0], {9}),
    (-9, [6.0, 1.2, 1.7, 0, 0, 0], {9, -9}),
    (91, [6.0, 1.2, 1.7, 0, 0, 0], {91}),
    (10, [6.0, 1.2, 1.7, 0, 0, 0], {10}),
    (11, [6.0, 1.2, 1.7, 0, 0, 0], {11}),
    (12, [6.0, 1.2, 1.7, 0.2, 0, 0], {12}),
    (-12, [6.0, 1.2, 1.7, 0, 0.2, 0], {12, -12}),
    (13, [6.0, 1.2, 1.7, 0.2, 0, 0], {13}),
    (-13, [6.0, 1.2, 1.7, 0, 0.2, 0], {13, -13}),
    (14, [6.0, 1.2, 1.7, 0.1, 0.2, 0.15], {14}),
]

_INVERSE_HINTS = {
    1: 'CUB',
    2: 'FCC',
    3: 'BCC',
    -3: 'BCC',
    4: 'HEX',
    5: 'RHL',
    -5: 'RHL',
    6: 'TET',
    7: 'BCT',
    8: 'ORC',
    9: 'ORCC',
    -9: 'ORCC',
    91: 'ORCA',
    10: 'ORCF',
    11: 'ORCI',
    12: 'MCL',
    -12: 'MCL',
    13: 'MCLC',
    -13: 'MCLC',
    14: 'TRI',
}


def _rotation(seed):
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def _unimodular(seed):
    rng = np.random.default_rng(seed + 1000)
    while True:
        m = rng.integers(-1, 2, size=(3, 3))
        if abs(round(float(np.linalg.det(m)))) == 1:
            return m


@pytest.mark.parametrize('ibrav, celldm, allowed', _INVERSE_CASES)
def test_inverse_roundtrip(ibrav, celldm, allowed):
    """Scramble (re-pick primitive cell + rotate) and recover the ibrav."""
    lat0 = lattice_format_QE(ibrav, celldm)
    seed = abs(ibrav) * 7 + (1 if ibrav < 0 else 0)
    lat = (_unimodular(seed) @ lat0) @ _rotation(seed).T

    res = qe_ibrav_from_lattice(lat, bravais_hint=_INVERSE_HINTS[ibrav], symprec=1e-6)
    assert res['ibrav'] in allowed, f'ibrav {ibrav} -> {res["ibrav"]}'

    # The returned (ibrav, celldm, M) must reproduce the input lattice metric.
    lat_qe = lattice_format_QE(res['ibrav'], res['celldm'])
    m = np.asarray(res['M'], dtype=float)
    g_in = lat @ lat.T
    np.testing.assert_allclose(m @ g_in @ m.T, lat_qe @ lat_qe.T, atol=1e-3)


def test_inverse_position_remap_preserves_geometry():
    """f_qe = f_in @ inv(M) must keep interatomic distances unchanged."""
    lat0 = lattice_format_QE(2, [7.0, 0, 0, 0, 0, 0])
    lat = lat0 @ _rotation(42).T
    frac_in = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    cart_in = frac_in @ lat

    res = qe_ibrav_from_lattice(lat, bravais_hint='FCC', symprec=1e-6)
    lat_qe = lattice_format_QE(res['ibrav'], res['celldm'])
    frac_qe = frac_in @ np.linalg.inv(np.asarray(res['M'], dtype=float))
    cart_qe = frac_qe @ lat_qe

    d_in = np.linalg.norm(cart_in[1] - cart_in[0])
    d_qe = np.linalg.norm(cart_qe[1] - cart_qe[0])
    assert d_in == pytest.approx(d_qe, abs=1e-9)


def test_inverse_skew_lattice_falls_back():
    """A generic skew cell with an incompatible hint returns ibrav=0."""
    skew = np.array([[5.0, 0.3, 0.1], [0.2, 5.5, 0.4], [0.1, 0.2, 6.0]])
    res = qe_ibrav_from_lattice(skew, bravais_hint='CUB', symprec=1e-5)
    assert res['ibrav'] == 0
    assert res['celldm'] is None
    assert res['M'] is None


def test_inverse_generic_skew_is_triclinic():
    skew = np.array([[5.0, 0.3, 0.1], [0.2, 5.5, 0.4], [0.1, 0.2, 6.0]])
    res = qe_ibrav_from_lattice(skew, symprec=1e-5)
    assert res['ibrav'] == 14


def test_cell_lengths_angles_cubic():
    a, b, c, al, be, ga = cell_lengths_angles(np.eye(3) * 4.0)
    assert (a, b, c) == pytest.approx((4.0, 4.0, 4.0))
    assert (al, be, ga) == pytest.approx((90.0, 90.0, 90.0))


def test_bravais_to_ibrav_known_symbols():
    assert bravais_to_ibrav('FCC') == 2
    assert bravais_to_ibrav('BCC') == 3
    assert bravais_to_ibrav('HEX') == 4
    assert bravais_to_ibrav('cF') == 2
    assert bravais_to_ibrav('unknown-symbol') is None
