"""Geometry tests for the ``fermi-plotter`` supercell and SKEAF field overlays.

Mayavi is never imported: ``fermi_plotter`` defers it to the body of
``plot_fermi_surface``, so every helper exercised here is pure NumPy.  The field
tests use a deliberately triclinic reciprocal cell so that an ``hvd`` selector
aligned with :math:`\\mathbf b_i` cannot accidentally pass through a Cartesian
axis.
"""

import math

import numpy as np
import pytest

from PAOFLOW.gen.fermi_plotter import (
    FieldSpec,
    _box_edges,
    _build_field,
    _build_parser,
    _cell_edges,
    _parse_supercell,
    central_replica,
    field_direction,
    field_sweep_arc,
    plane_basis,
    resolve_field_angles,
    supercell_translations,
    tile_mesh,
)

# Triclinic: no reciprocal vector is parallel to a Cartesian axis.
RECIP = np.array(
    [
        [1.3, 0.2, -0.1],
        [0.3, 0.9, 0.25],
        [-0.15, 0.4, 1.1],
    ]
)


def _sorted_rows(points, decimals=6):
    rounded = np.round(np.asarray(points, dtype=float), decimals)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


# --------------------------------------------------------------------------- #
# Box edges
# --------------------------------------------------------------------------- #


def test_box_edges_count_and_lengths():
    edges = _box_edges(np.zeros(3), *RECIP)
    assert len(edges) == 12
    lengths = sorted(float(np.linalg.norm(p1 - p0)) for p0, p1 in edges)
    expected = sorted(np.repeat(np.linalg.norm(RECIP, axis=1), 4).tolist())
    np.testing.assert_allclose(lengths, expected, atol=1e-12)


def test_cell_edges_is_box_at_origin():
    for (a0, a1), (b0, b1) in zip(_cell_edges(RECIP), _box_edges(np.zeros(3), *RECIP)):
        np.testing.assert_allclose(a0, b0)
        np.testing.assert_allclose(a1, b1)


def test_box_edges_translate_rigidly():
    shift = np.array([0.7, -1.2, 0.3])
    for (p0, p1), (q0, q1) in zip(_box_edges(np.zeros(3), *RECIP), _box_edges(shift, *RECIP)):
        np.testing.assert_allclose(q0 - p0, shift)
        np.testing.assert_allclose(q1 - p1, shift)


# --------------------------------------------------------------------------- #
# Supercell replication
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'ncell,expected',
    [((1, 1, 1), (0, 0, 0)), ((2, 2, 2), (1, 1, 1)), ((3, 1, 4), (1, 0, 2))],
)
def test_central_replica(ncell, expected):
    assert central_replica(ncell) == expected


def test_translations_count_and_origin_first():
    trans = supercell_translations(RECIP, (2, 3, 4))
    assert trans.shape == (24, 3)
    np.testing.assert_allclose(trans[0], np.zeros(3), atol=1e-12)


def test_translations_are_integer_lattice_vectors():
    trans = supercell_translations(RECIP, (2, 2, 2), center=True)
    frac = trans @ np.linalg.inv(RECIP)
    np.testing.assert_allclose(frac, np.round(frac), atol=1e-12)


def test_centering_puts_central_replica_at_origin():
    ncell = (3, 2, 4)
    trans = supercell_translations(RECIP, ncell, center=True)
    offset = np.asarray(central_replica(ncell), dtype=float) @ RECIP
    np.testing.assert_allclose(trans[0], -offset, atol=1e-12)
    # the central replica itself now sits exactly on the origin
    assert np.isclose(np.linalg.norm(trans, axis=1).min(), 0.0, atol=1e-12)


def test_centering_is_noop_for_single_cell():
    plain = supercell_translations(RECIP, (1, 1, 1), center=False)
    centred = supercell_translations(RECIP, (1, 1, 1), center=True)
    np.testing.assert_allclose(plain, centred)
    np.testing.assert_allclose(centred, np.zeros((1, 3)))


def test_translations_reject_zero_repetition():
    with pytest.raises(ValueError):
        supercell_translations(RECIP, (2, 0, 1))


# --------------------------------------------------------------------------- #
# Mesh tiling
# --------------------------------------------------------------------------- #


def _toy_mesh():
    cart = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    scalars = np.array([1.0, 2.0, 3.0, 4.0])
    return cart, faces, scalars


def test_tile_mesh_identity_for_single_cell():
    cart, faces, scalars = _toy_mesh()
    out_c, out_f, out_s = tile_mesh(cart, faces, scalars, np.zeros((1, 3)))
    np.testing.assert_allclose(out_c, cart)
    np.testing.assert_array_equal(out_f, faces)
    np.testing.assert_allclose(out_s, scalars)


def test_tile_mesh_shapes_and_offsets():
    cart, faces, scalars = _toy_mesh()
    trans = supercell_translations(RECIP, (2, 2, 1))
    ncopy = trans.shape[0]
    out_c, out_f, out_s = tile_mesh(cart, faces, scalars, trans)

    assert out_c.shape == (ncopy * len(cart), 3)
    assert out_f.shape == (ncopy * len(faces), 3)
    assert out_s.shape == (ncopy * len(scalars),)

    for copy in range(ncopy):
        block = out_c[copy * len(cart) : (copy + 1) * len(cart)]
        np.testing.assert_allclose(block, cart + trans[copy])
        np.testing.assert_allclose(out_s[copy * len(scalars) : (copy + 1) * len(scalars)], scalars)


def test_tile_mesh_face_indices_stay_in_range():
    cart, faces, scalars = _toy_mesh()
    trans = supercell_translations(RECIP, (2, 2, 2))
    out_c, out_f, _ = tile_mesh(cart, faces, scalars, trans)
    assert out_f.min() >= 0
    assert out_f.max() == len(out_c) - 1
    # every replica must reference only its own vertices
    per_copy = np.split(out_f, trans.shape[0])
    for copy, block in enumerate(per_copy):
        assert block.min() == copy * len(cart)
        assert block.max() == (copy + 1) * len(cart) - 1


def test_tiling_a_periodic_isosurface_is_seamless():
    """Translation tiling relies on the BXSF wrap plane closing the surface."""
    measure = pytest.importorskip('skimage.measure')

    n = 16
    frac = np.linspace(0.0, 1.0, n + 1)  # wrap-closed, as written by write2bxsf
    fx, fy, fz = np.meshgrid(frac, frac, frac, indexing='ij')
    energy = np.cos(2 * np.pi * fx) + np.cos(2 * np.pi * fy) + np.cos(2 * np.pi * fz)

    verts, _, _, _ = measure.marching_cubes(energy, level=0.0)
    cart = (verts / n) @ RECIP

    tol = 1e-9
    on_low = cart[np.abs(verts[:, 0]) < tol]
    on_high = cart[np.abs(verts[:, 0] - n) < tol]
    assert len(on_low) > 0

    # The +b0 neighbour's low face must land exactly on this cell's high face.
    np.testing.assert_allclose(_sorted_rows(on_low + RECIP[0]), _sorted_rows(on_high), atol=1e-9)


# --------------------------------------------------------------------------- #
# SKEAF field geometry
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'theta_deg,phi_deg,expected',
    [
        (0.0, 0.0, [0.0, 0.0, 1.0]),
        (0.0, 90.0, [1.0, 0.0, 0.0]),
        (90.0, 90.0, [0.0, 1.0, 0.0]),
        (180.0, 90.0, [-1.0, 0.0, 0.0]),
    ],
)
def test_field_direction_known_angles(theta_deg, phi_deg, expected):
    vec = field_direction(math.radians(theta_deg), math.radians(phi_deg))
    np.testing.assert_allclose(vec, expected, atol=1e-12)
    assert np.isclose(np.linalg.norm(vec), 1.0)


@pytest.mark.parametrize('hvd,index', [('a', 0), ('b', 1), ('c', 2)])
def test_principal_axis_field_follows_reciprocal_vector(hvd, index):
    theta, phi = resolve_field_angles(RECIP, hvd)
    vec = field_direction(theta, phi)
    axis = RECIP[index] / np.linalg.norm(RECIP[index])
    np.testing.assert_allclose(np.cross(vec, axis), np.zeros(3), atol=1e-12)
    assert np.dot(vec, axis) > 0.0


def test_non_principal_field_passes_angles_through():
    theta, phi = resolve_field_angles(RECIP, 'n', math.radians(30.0), math.radians(45.0))
    assert np.isclose(theta, math.radians(30.0))
    assert np.isclose(phi, math.radians(45.0))


def test_angles_round_trip_through_set_field_angle():
    theta_in, phi_in = math.radians(37.0), math.radians(64.0)
    vec = field_direction(theta_in, phi_in)
    theta_out, phi_out = resolve_field_angles(vec[None, :].repeat(3, axis=0), 'a')
    assert np.isclose(theta_out, theta_in)
    assert np.isclose(phi_out, phi_in)


def test_sweep_arc_endpoints_and_sampling():
    t0, p0 = math.radians(0.0), math.radians(45.0)
    t1, p1 = math.radians(90.0), math.radians(15.0)
    arc = field_sweep_arc(t0, p0, t1, p1, 7)

    assert arc.shape == (7, 3)
    np.testing.assert_allclose(np.linalg.norm(arc, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(arc[0], field_direction(t0, p0), atol=1e-12)
    np.testing.assert_allclose(arc[-1], field_direction(t1, p1), atol=1e-12)


def test_sweep_arc_matches_runner_linspace_stepping():
    """The arc must trace the angles run_angle_sweep actually visits."""
    t0, p0, t1, p1, n = 0.1, 0.2, 1.4, 0.9, 5
    arc = field_sweep_arc(t0, p0, t1, p1, n)
    expected = np.array(
        [field_direction(t, p) for t, p in zip(np.linspace(t0, t1, n), np.linspace(p0, p1, n))]
    )
    np.testing.assert_allclose(arc, expected, atol=1e-12)


def test_sweep_arc_needs_at_least_two_points():
    assert field_sweep_arc(0.0, 0.0, 1.0, 1.0, 1).shape == (2, 3)


@pytest.mark.parametrize('normal', [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.3, -0.8, 0.5]])
def test_plane_basis_is_orthonormal(normal):
    u, v = plane_basis(np.array(normal))
    n = np.array(normal) / np.linalg.norm(normal)
    assert np.isclose(np.linalg.norm(u), 1.0)
    assert np.isclose(np.linalg.norm(v), 1.0)
    assert np.isclose(np.dot(u, v), 0.0, atol=1e-12)
    assert np.isclose(np.dot(u, n), 0.0, atol=1e-12)
    assert np.isclose(np.dot(v, n), 0.0, atol=1e-12)


# --------------------------------------------------------------------------- #
# CLI parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    'spec,expected',
    [('2', (2, 2, 2)), ('2,3,4', (2, 3, 4)), (' 1 2 3 ', (1, 2, 3)), ('2x2x2', (2, 2, 2))],
)
def test_parse_supercell(spec, expected):
    assert _parse_supercell(spec) == expected


@pytest.mark.parametrize('spec', ['0', '2,3', '-1', 'abc', '1,2,3,4'])
def test_parse_supercell_rejects_bad_input(spec):
    with pytest.raises(SystemExit):
        _parse_supercell(spec)


def _field_from_argv(argv):
    return _build_field(_build_parser().parse_args(['dummy.bxsf', *argv]))


def test_no_field_by_default():
    assert _field_from_argv([]) is None


def test_principal_field_flags_map_to_hvd():
    for name, hvd in (('b1', 'a'), ('b2', 'b'), ('b3', 'c')):
        spec = _field_from_argv(['--b-field', name])
        assert isinstance(spec, FieldSpec)
        assert spec.hvd == hvd


def test_non_principal_field_converts_degrees_to_radians():
    spec = _field_from_argv(['--b-field', 'non_principal', '--azimuthal', '30', '--polar', '45'])
    assert spec.hvd == 'n'
    assert np.isclose(spec.theta, math.radians(30.0))
    assert np.isclose(spec.phi, math.radians(45.0))
    assert spec.num_angles == 1


def test_angle_pair_implies_rotation():
    spec = _field_from_argv(
        ['--b-field', 'b1', '--azimuthal', '0,90', '--polar', '45,45', '--num-angles', '7']
    )
    assert spec.hvd == 'r'
    assert np.isclose(spec.theta_end, math.radians(90.0))
    assert np.isclose(spec.phi_end, math.radians(45.0))
    assert spec.num_angles == 7


def test_rotation_requires_angle_pairs():
    with pytest.raises(SystemExit):
        _field_from_argv(['--b-field', 'rotation', '--azimuthal', '0', '--polar', '45'])


def test_rotation_requires_two_angles():
    with pytest.raises(SystemExit):
        _field_from_argv(['--b-field', 'rotation', '--azimuthal', '0,90', '--polar', '0,45'])


def test_mismatched_angle_arity_is_rejected():
    with pytest.raises(SystemExit):
        _field_from_argv(['--b-field', 'b1', '--azimuthal', '0,90', '--polar', '45'])


def test_plane_and_label_flags_are_opt_in():
    default = _field_from_argv(['--b-field', 'b1'])
    assert not default.show_plane
    assert not default.show_label
    both = _field_from_argv(['--b-field', 'b1', '--field-plane', '--field-label'])
    assert both.show_plane
    assert both.show_label
