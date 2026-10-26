"""The window of a sparse pass, derived from the property ranges.

The arithmetic of :mod:`PAOFLOW.sparse.window` on synthetic ranges, and the
range each property declares from its dense arguments.  No DFT data: the
properties are built on a ``restart=True`` driver.
"""

import numpy as np
import pytest

from PAOFLOW.PAOFLOW import PAOFLOW
from PAOFLOW.sparse import window as W


def _range(emin, emax, kind=W.NONE, margin=0.0, tmax=None, method='p'):
    return W.Range(method, emin, emax, kind, margin, tmax)


def test_thermal_margin_is_ten_kT():
    assert W.kt_margin(300.0) == pytest.approx(10 * 8.617333e-5 * 300.0)
    assert W.kt_margin(0.0) == 0.0


def test_smearing_margin_is_four_widths():
    assert W.smearing_margin(0.2) == pytest.approx(0.8)


def test_probe_count_is_padded():
    assert W.padded_count(8) == 16
    assert W.padded_count(1000) == 1020


def test_bottom_window_takes_the_highest_range_top():
    ranges = [_range(-12.0, 2.2, W.SMEARING, 0.5), _range(-2.0, 2.0, W.THERMAL, 0.26, 300.0)]
    assert W.derive_bottom(ranges, ptop=6.96) == (pytest.approx(2.7), False)


def test_bottom_window_is_capped_at_the_projectable_top():
    assert W.derive_bottom([_range(-1.0, 6.9, W.SMEARING, 0.5)], ptop=6.96) == (6.96, True)
    # no range: the whole well-projected range
    assert W.derive_bottom([], ptop=6.96) == (6.96, True)
    assert W.derive_bottom([], ptop=None) == (None, True)


def test_interior_window_wraps_every_range():
    ranges = [_range(-2.0, 2.0, W.SMEARING, 0.5), _range(-1.2, 1.7, W.THERMAL, 0.26, 300.0)]
    elo, ehi = W.derive_interior(ranges, ptop=6.96)
    assert (elo, ehi) == (pytest.approx(-2.5), pytest.approx(2.5))
    assert W.derive_interior([_range(-1.0, 8.0)], ptop=6.96) == (-1.0, 6.96)


def test_interior_window_needs_a_range_below_the_top():
    with pytest.raises(ValueError, match='no property of this pass has one'):
        W.derive_interior([], ptop=6.96)
    with pytest.raises(ValueError, match='above the well-projected'):
        W.derive_interior([_range(7.5, 9.0)], ptop=6.96)


def test_pass_window_reports_its_margins():
    window = W.PassWindow(
        'interior',
        -2.5,
        2.5,
        None,
        False,
        (_range(-2.0, 2.0, W.SMEARING, 0.5), _range(-1.0, 1.0, W.THERMAL, 0.1, 100.0)),
        'interior',
    )
    assert window.interior == (-2.5, 2.5)
    assert window.margin(W.SMEARING) == 0.5
    assert window.margin(W.THERMAL, 100.0) == 0.1
    # a temperature no range of the pass used: computed on the spot
    assert window.margin(W.THERMAL, 600.0) == pytest.approx(W.kt_margin(600.0))
    assert window.margin(W.NONE) == 0.0


@pytest.fixture
def engine(tmp_path):
    p = PAOFLOW(workpath=str(tmp_path), outputdir='r', restart=True, sparse=True)
    attr = p.data_controller.data_attributes
    attr.setdefault('nspin', 1)
    return p.sparse


def test_properties_declare_their_ranges(engine):
    from PAOFLOW.sparse.properties import _load

    registry = _load()
    cases = [
        (registry['dos'](engine, emin=-12.0, emax=2.2), (-12.0, 2.2, W.SMEARING, None)),
        (
            registry['conductivity'](engine, emin=-2.0, emax=2.0, cond_tensor=[[0, 0]]),
            (-2.0, 2.0, W.SMEARING, None),
        ),
        (
            registry['transport'](engine, tmin=100, tmax=600, emin=-1.0, emax=1.5),
            (-1.0, 1.5, W.THERMAL, 600),
        ),
        (
            registry['doping'](engine, tmax=300, emin=-1.0, emax=1.0),
            (-1.0, 1.0, W.THERMAL, 300),
        ),
        (
            registry['fermi_surface'](engine, fermi_up=0.5, fermi_dw=-0.5),
            (-0.5, 0.5, W.NONE, None),
        ),
        (
            registry['rashba_edelstein'](engine, emin=-1.0, emax=1.0, intra_band=True),
            (-1.0, 1.0, W.SMEARING, None),
        ),
        (
            registry['rashba_edelstein'](engine, emin=-1.0, emax=1.0),
            (-1.0, 1.0, W.NONE, None),
        ),
        (
            registry['anomalous_Hall'](engine, emin=-0.5, emax=0.5, a_tensor=[[0, 1]]),
            (-0.5, 0.5, W.NONE, None),
        ),
    ]
    for prop, expected in cases:
        assert prop.energy_range() == expected, prop.method
    dielectric = registry['dielectric_tensor'](engine, emin=0.0, emax=6.0, d_tensor='diag')
    emin, emax, kind, _ = dielectric.energy_range()
    assert emin == -np.inf and (emax, kind) == (6.0, W.NONE)
    assert registry['effective_mass'](engine).energy_range() is None


def test_a_clipped_range_is_written_back(engine):
    from PAOFLOW.sparse.properties import _load

    registry = _load()
    dos = registry['dos'](engine, emin=-12.0, emax=2.2)
    dos.apply_range(-2.0, 2.0)
    assert dos.energy_range()[:2] == (-2.0, 2.0)
    transport = registry['transport'](engine, emin=-2.0, emax=2.0)
    transport.apply_range(-1.0, 1.0)
    assert (transport.kwargs['emin'], transport.kwargs['emax']) == (-1.0, 1.0)
