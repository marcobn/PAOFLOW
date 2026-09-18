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
    brillouin_zone,
    bz_edges,
    bz_planes,
    central_replica,
    clip_to_planes,
    field_direction,
    field_sweep_arc,
    fold_into_bz,
    plane_basis,
    reciprocal_neighbours,
    reduce_basis,
    resolve_field_angles,
    select_near_bz,
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

# One representative reciprocal basis per Bravais family, plus a real
# face-centred orthorhombic (ORCF1) cell taken from a Nb1Sn2 QE run and a
# deliberately non-reduced basis of the simple cubic lattice.
LATTICES = {
    'cubic': np.eye(3),
    'tetragonal': np.diag([1.0, 1.0, 0.4]),
    'orthorhombic': np.diag([0.7, 1.1, 1.6]),
    'bcc_recip': np.array([[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]),
    'fcc_recip': np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]),
    'hexagonal': np.array(
        [[1.0, -1.0 / math.sqrt(3.0), 0.0], [0.0, 2.0 / math.sqrt(3.0), 0.0], [0.0, 0.0, 0.6]]
    ),
    'triclinic': RECIP,
    'orcf1_nb1sn2': np.array(
        [
            [-3.172282, 1.556733, 0.527974],
            [3.172282, -1.556733, 0.527974],
            [3.172282, 1.556733, -0.527974],
        ]
    ),
    'non_reduced_cubic': np.array([[1.0, 0.0, 0.0], [3.0, 1.0, 0.0], [0.0, 4.0, 1.0]]),
}


def _sorted_rows(points, decimals=6):
    rounded = np.round(np.asarray(points, dtype=float), decimals)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


def _triangle_area(tri):
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1).sum()


def _periodic_isosurface(recip, n=20):
    """A closed, lattice-periodic sheet meshed on the wrap-closed cell grid."""
    measure = pytest.importorskip('skimage.measure')
    frac = np.linspace(0.0, 1.0, n + 1)
    fx, fy, fz = np.meshgrid(frac, frac, frac, indexing='ij')
    energy = np.cos(2 * np.pi * fx) + np.cos(2 * np.pi * fy) + np.cos(2 * np.pi * fz)
    verts, faces, _, _ = measure.marching_cubes(energy, level=0.0)
    cart = (verts / n) @ recip
    return cart, faces, np.linalg.norm(cart, axis=1)


# --------------------------------------------------------------------------- #
# Brillouin zone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('name', sorted(LATTICES))
def test_bz_closes_for_every_lattice(name):
    """The zone must tile space, i.e. carry exactly the primitive cell volume."""
    hull_mod = pytest.importorskip('scipy.spatial')
    recip = LATTICES[name]
    verts, faces = brillouin_zone(recip)
    volume = hull_mod.ConvexHull(verts).volume
    assert np.isclose(volume, abs(np.linalg.det(recip)), rtol=1e-6)
    assert len(faces) >= 6


@pytest.mark.parametrize('name', sorted(LATTICES))
def test_bz_is_a_convex_polyhedron(name):
    """Euler's formula V - E + F = 2 holds for any convex polyhedron."""
    verts, faces = brillouin_zone(LATTICES[name])
    edges = bz_edges(verts, faces)
    assert len(verts) - len(edges) + len(faces) == 2


@pytest.mark.parametrize('name', sorted(LATTICES))
def test_bz_corners_are_closest_to_gamma(name):
    """Defining property of the Wigner-Seitz cell."""
    recip = LATTICES[name]
    verts, _ = brillouin_zone(recip)
    neighbours = reciprocal_neighbours(reduce_basis(recip), 2)
    to_gamma = np.linalg.norm(verts, axis=1)
    to_others = np.linalg.norm(verts[:, None, :] - neighbours[None, :, :], axis=2).min(axis=1)
    assert np.all(to_gamma <= to_others + 1e-9)


@pytest.mark.parametrize('name', sorted(LATTICES))
def test_bz_is_inversion_symmetric(name):
    verts, _ = brillouin_zone(LATTICES[name])
    rounded = np.round(verts, 8)
    assert {tuple(v) for v in rounded} == {tuple(-v) for v in rounded}


@pytest.mark.parametrize(
    'name,nfaces,nverts',
    [
        # simple cubic -> cube
        ('cubic', 6, 8),
        # reciprocal of fcc is bcc -> truncated octahedron
        ('bcc_recip', 14, 24),
        # reciprocal of bcc is fcc -> rhombic dodecahedron
        ('fcc_recip', 12, 14),
        # hexagonal -> hexagonal prism
        ('hexagonal', 8, 12),
    ],
)
def test_bz_matches_textbook_shapes(name, nfaces, nverts):
    verts, faces = brillouin_zone(LATTICES[name])
    assert len(faces) == nfaces
    assert len(verts) == nverts


def test_bz_is_a_lattice_property_not_a_basis_property():
    """A non-reduced basis of the cubic lattice must give the same cube."""
    plain, _ = brillouin_zone(LATTICES['cubic'])
    skewed, _ = brillouin_zone(LATTICES['non_reduced_cubic'])
    np.testing.assert_allclose(_sorted_rows(plain), _sorted_rows(skewed), atol=1e-9)


def test_reduce_basis_preserves_the_lattice():
    for name, recip in LATTICES.items():
        reduced = reduce_basis(recip)
        # the change of basis must be an integer, unimodular matrix
        transform = reduced @ np.linalg.inv(recip)
        np.testing.assert_allclose(transform, np.round(transform), atol=1e-9, err_msg=name)
        assert np.isclose(abs(np.linalg.det(transform)), 1.0), name


def test_reduce_basis_shortens_a_skewed_basis():
    reduced = reduce_basis(LATTICES['non_reduced_cubic'])
    assert np.allclose(np.sort(np.linalg.norm(reduced, axis=1)), [1.0, 1.0, 1.0])


@pytest.mark.parametrize('name', sorted(LATTICES))
def test_bz_planes_bound_the_zone(name):
    verts, faces = brillouin_zone(LATTICES[name])
    normals, offsets = bz_planes(verts, faces)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(offsets > 0.0)
    assert np.all(verts @ normals.T <= offsets[None, :] + 1e-9)


# --------------------------------------------------------------------------- #
# Clipping and folding into the zone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize('name', ['cubic', 'bcc_recip', 'orcf1_nb1sn2', 'triclinic'])
def test_clipping_keeps_geometry_inside_the_zone(name):
    recip = LATTICES[name]
    verts, faces = brillouin_zone(recip)
    normals, offsets = bz_planes(verts, faces)

    rng = np.random.default_rng(5)
    tri = rng.random((500, 3, 3)) @ recip
    values = rng.random((500, 3))
    clipped, kept = clip_to_planes(tri, values, normals, offsets)

    assert np.all(clipped.reshape(-1, 3) @ normals.T <= offsets[None, :] + 1e-9)
    # interpolated scalars can only ever lie between the original corner values
    assert kept.min() >= values.min() - 1e-9
    assert kept.max() <= values.max() + 1e-9


def test_clipping_a_fully_interior_triangle_is_a_no_op():
    verts, faces = brillouin_zone(LATTICES['cubic'])
    normals, offsets = bz_planes(verts, faces)
    tri = np.array([[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]])
    values = np.array([[1.0, 2.0, 3.0]])
    clipped, kept = clip_to_planes(tri, values, normals, offsets)
    np.testing.assert_allclose(clipped, tri)
    np.testing.assert_allclose(kept, values)


def test_clipping_drops_a_fully_exterior_triangle():
    verts, faces = brillouin_zone(LATTICES['cubic'])
    normals, offsets = bz_planes(verts, faces)
    far = np.array([[[9.0, 9.0, 9.0], [9.1, 9.0, 9.0], [9.0, 9.1, 9.0]]])
    clipped, _ = clip_to_planes(far, np.ones((1, 3)), normals, offsets)
    assert clipped.shape[0] == 0


def test_clipping_halves_a_straddling_triangle():
    """Cut a big triangle with the x = 1/2 face of the cubic zone."""
    normals = np.array([[1.0, 0.0, 0.0]])
    offsets = np.array([0.5])
    tri = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    clipped, _ = clip_to_planes(tri, np.ones((1, 3)), normals, offsets)
    assert np.all(clipped[..., 0] <= 0.5 + 1e-12)
    # the half-plane cuts off exactly 1/4 of this right triangle's area
    assert np.isclose(_triangle_area(clipped), 0.75 * _triangle_area(tri))


def test_select_near_bz_keeps_only_touching_triangles():
    recip = LATTICES['cubic']
    verts, faces = brillouin_zone(recip)
    normals, offsets = bz_planes(verts, faces)

    cart = np.array(
        [
            [0.0, 0.0, 0.0],  # inside
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [9.0, 9.0, 9.0],  # far outside
            [9.1, 9.0, 9.0],
            [9.0, 9.1, 9.0],
        ]
    )
    tri_faces = np.array([[0, 1, 2], [3, 4, 5]])
    kept_cart, kept_faces, _ = select_near_bz(cart, tri_faces, np.ones(len(cart)), normals, offsets)
    assert kept_faces.shape[0] == 1
    np.testing.assert_allclose(_sorted_rows(kept_cart), _sorted_rows(cart[:3]))


@pytest.mark.parametrize('name', ['cubic', 'tetragonal', 'bcc_recip', 'fcc_recip', 'orcf1_nb1sn2'])
def test_folding_conserves_surface_area(name):
    """A periodic sheet has the same area in the zone as in the cell."""
    recip = LATTICES[name]
    cart, faces, scalars = _periodic_isosurface(recip)
    verts, bz_faces = brillouin_zone(recip)
    normals, offsets = bz_planes(verts, bz_faces)

    folded_cart, folded_faces, _ = fold_into_bz(cart, faces, scalars, recip, normals, offsets)

    area_cell = _triangle_area(cart[faces])
    area_zone = _triangle_area(folded_cart[folded_faces])
    assert np.isclose(area_zone, area_cell, rtol=2e-3)


@pytest.mark.parametrize('name', ['cubic', 'bcc_recip', 'orcf1_nb1sn2'])
def test_folded_sheet_stays_inside_the_zone(name):
    recip = LATTICES[name]
    cart, faces, scalars = _periodic_isosurface(recip)
    verts, bz_faces = brillouin_zone(recip)
    normals, offsets = bz_planes(verts, bz_faces)

    folded_cart, _, _ = fold_into_bz(cart, faces, scalars, recip, normals, offsets)
    assert np.all(folded_cart @ normals.T <= offsets[None, :] + 1e-8)


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


def test_bz_defaults_to_the_true_brillouin_zone():
    assert _build_parser().parse_args(['dummy.bxsf']).bz == 'ws'


@pytest.mark.parametrize('choice', ['ws', 'cell', 'none'])
def test_bz_choices_are_accepted(choice):
    assert _build_parser().parse_args(['dummy.bxsf', '--bz', choice]).bz == choice


def test_bz_rejects_unknown_choice():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(['dummy.bxsf', '--bz', 'octahedron'])
