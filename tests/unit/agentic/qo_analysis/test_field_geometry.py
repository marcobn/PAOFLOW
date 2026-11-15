import numpy as np
import pytest

from PAOFLOW.agentic.qo_analysis.field_geometry import (
    conventional_cell,
    direction_report,
    direction_to_field_angles,
    field_angles_to_direction,
)

pytestmark = pytest.mark.unit

NBSN2_PRIMITIVE = [[0.5, 0.0, 1.690651], [0.5, 0.867802, 0.0], [0.0, 0.867802, 1.690651]]
NBSN2_CENTRING = [[1, 1, -1], [-1, 1, 1], [1, -1, 1]]


def test_cubic_conventions_coincide():
    cube = np.eye(3)
    assert direction_to_field_angles(cube, [1, 1, 0], 'reciprocal') == pytest.approx((45.0, 90.0))
    assert direction_to_field_angles(cube, [1, 1, 0], 'real') == pytest.approx((45.0, 90.0))


def test_orthorhombic_110_differs_between_conventions():
    conventional = conventional_cell(np.array(NBSN2_PRIMITIVE), NBSN2_CENTRING)
    assert np.allclose(np.linalg.norm(conventional, axis=1), [1.0, 1.735604, 3.381302], atol=1e-5)
    assert direction_to_field_angles(conventional, [1, 1, 0], 'reciprocal')[0] == pytest.approx(
        29.949, abs=1e-3
    )
    assert direction_to_field_angles(conventional, [1, 1, 0], 'real')[0] == pytest.approx(
        60.051, abs=1e-3
    )


@pytest.mark.parametrize('convention', ['reciprocal', 'real'])
def test_angles_to_indices_round_trip(convention):
    conventional = conventional_cell(np.array(NBSN2_PRIMITIVE), NBSN2_CENTRING)
    indices = field_angles_to_direction(conventional, 29.95, 45.0, convention)
    assert direction_to_field_angles(conventional, indices, convention) == pytest.approx(
        (29.95, 45.0), abs=1e-6
    )


def test_nbsn2_45_degrees_is_not_11_2p4():
    conventional = conventional_cell(np.array(NBSN2_PRIMITIVE), NBSN2_CENTRING)
    assert field_angles_to_direction(conventional, 29.95, 45.0)[2] == pytest.approx(3.902, abs=1e-3)
    assert direction_to_field_angles(conventional, [1, 1, 2.4])[1] == pytest.approx(58.41, abs=0.01)


def test_report_flags_mismatch():
    conventional = conventional_cell(np.array(NBSN2_PRIMITIVE), NBSN2_CENTRING)
    report = direction_report(
        conventional, [dict(label='[110]', hkl=[1, 1, 0], azimuthal=60.05, polar=90.0)]
    )
    assert 'MISMATCH' in report
