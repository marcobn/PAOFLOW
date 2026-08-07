"""Unit tests for the surface-projected transverse k-path builder."""

import numpy as np
import pytest

from PAOFLOW.transport.grid.kpath import _parse_path_labels, build_surface_kpath


class _FakeDataController:
    """Minimal stand-in exposing the arrays/attributes the builder reads."""

    def __init__(self, ibrav=None, band_path=None, high_sym_points=None):
        cell = np.eye(3)
        self._arrays = {
            'a_vectors': cell,
            'b_vectors': cell,
            'high_sym_points': high_sym_points if high_sym_points is not None else {},
        }
        self._attributes = {'alat': 1.0, 'ibrav': ibrav, 'band_path': band_path}

    def data_dicts(self):
        return self._arrays, self._attributes


@pytest.mark.unit
def test_parse_path_labels_accumulates_segment_counts():
    """Tick indices are the running sum of the per-segment point counts."""
    ticks, labels = _parse_path_labels('gG 10\nX 5\nM 0\n')

    assert labels == ['gG', 'X', 'M']
    np.testing.assert_array_equal(ticks, [0, 10, 15])


@pytest.mark.unit
@pytest.mark.parametrize(('direction', 'axis'), [('x', 0), ('y', 1), ('z', 2)])
def test_surface_kpath_projects_out_transport_axis(direction, axis):
    """The surface-normal component is removed so phases stay in-plane."""
    kpath = build_surface_kpath(
        _FakeDataController(ibrav=1),
        transport_direction=direction,
        band_path='gG-X-M-gG',
        nk_path=30,
    )

    np.testing.assert_allclose(kpath.vkpt_par3D[:, axis], 0.0)
    assert kpath.vkpt_par3D.shape[1] == 3
    assert kpath.nkpts == kpath.vkpt_par3D.shape[0]


@pytest.mark.unit
def test_surface_kpath_uses_unit_weights():
    """The spectral function is reported raw, so weights must not be normalized."""
    kpath = build_surface_kpath(
        _FakeDataController(ibrav=1),
        transport_direction='z',
        band_path='gG-X',
        nk_path=20,
    )

    np.testing.assert_allclose(kpath.wk_par, 1.0)
    assert kpath.wk_par.shape == (kpath.nkpts,)


@pytest.mark.unit
def test_surface_kpath_distance_is_monotonic():
    """The plotting abscissa must increase along the path and start at zero."""
    kpath = build_surface_kpath(
        _FakeDataController(ibrav=1),
        transport_direction='z',
        band_path='gG-X-M',
        nk_path=40,
    )

    assert kpath.kdist[0] == 0.0
    assert np.all(np.diff(kpath.kdist) >= -1e-12)
    assert kpath.kdist[-1] > 0.0
    assert kpath.kdist.shape == (kpath.nkpts,)


@pytest.mark.unit
def test_surface_kpath_labels_align_with_ticks():
    """Every tick index addresses a real k-point and pairs with a label."""
    kpath = build_surface_kpath(
        _FakeDataController(ibrav=1),
        transport_direction='z',
        band_path='gG-X-M',
        nk_path=40,
    )

    assert len(kpath.ticks) == len(kpath.labels)
    assert kpath.labels[0] == 'gG'
    assert np.all(kpath.ticks >= 0)
    assert np.all(kpath.ticks < kpath.nkpts)


@pytest.mark.unit
def test_surface_kpath_nk_path_is_approximately_honored():
    """The two-pass dk rescale should land near the requested point count."""
    target = 100
    kpath = build_surface_kpath(
        _FakeDataController(ibrav=1),
        transport_direction='z',
        band_path='gG-X',
        nk_path=target,
    )

    assert abs(kpath.nkpts - target) <= 0.1 * target


@pytest.mark.unit
def test_surface_kpath_requires_lattice_information():
    """Without ibrav or an explicit path the builder cannot proceed."""
    with pytest.raises(ValueError, match='surface k-path'):
        build_surface_kpath(
            _FakeDataController(ibrav=None),
            transport_direction='z',
        )


@pytest.mark.unit
def test_surface_kpath_rejects_ibrav_zero_without_explicit_path():
    """ibrav=0 has no tabulated high-symmetry points."""
    with pytest.raises(ValueError, match='ibrav=0'):
        build_surface_kpath(
            _FakeDataController(ibrav=0),
            transport_direction='z',
            band_path='gG-X',
        )
