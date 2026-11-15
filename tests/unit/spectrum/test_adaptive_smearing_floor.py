"""Floor and cap of the adaptive (Yates) smearing widths.

The Yates width ``afac |v| dk`` vanishes at band extrema and is tiny on flat bands. A state
narrower than the spacing of the energy grid drops out of the adaptive DOS (or turns it into
NaN when the width is exactly zero). The floor of ``do_adaptive_smearing`` restores the count.
"""

import numpy as np
import pytest
from scipy.integrate import trapezoid

from PAOFLOW.spectrum.do_adaptive_smearing import do_adaptive_smearing
from PAOFLOW.spectrum.do_dos import do_dos_adaptive


class _DC:
    def __init__(self, arrays, attributes):
        self._a = arrays
        self._t = attributes

    def data_dicts(self):
        return self._a, self._t

    def write_file_row_col(self, *args, **kwargs):
        pass

    def broadcast_single_array(self, *args, **kwargs):
        pass


def _model(nk=12):
    """Three bands on a ring of k-points: flat, cosine, and a parabola with its minimum on the grid."""
    k = np.linspace(-np.pi, np.pi, nk, endpoint=False)
    energies = np.stack([np.full(nk, -3.0), -np.cos(k), 2.0 + 0.5 * k**2], axis=1)
    velocities = np.stack([np.zeros(nk), np.sin(k), k], axis=1)
    nawf = 3
    pksp = np.zeros((nk, 3, nawf, nawf, 1), dtype=complex)
    for n in range(nawf):
        pksp[:, 0, n, n, 0] = velocities[:, n]
    arrays = {'E_k': energies[:, :, None], 'pksp': pksp}
    attributes = {
        'nawf': nawf,
        'nspin': 1,
        'nkpnts': nk,
        'omega': 100.0,
        'bnd': nawf,
        'smearing': 'gauss',
        'verbose': False,
    }
    return _DC(arrays, attributes), velocities


def _raw_widths(dc, velocities, afac=0.7):
    _, attr = dc.data_dicts()
    dk = (8.0 * np.pi**3 / attr['omega'] / attr['nkpnts']) ** (1.0 / 3.0)
    return afac * dk * np.abs(velocities)


def test_floor_and_cap_bound_the_widths():
    dc, velocities = _model()
    raw = _raw_widths(dc, velocities)
    do_adaptive_smearing(dc, 'gauss', None, min_width=0.05, max_width=0.3)
    widths = dc.data_dicts()[0]['deltakp'][:, :, 0]
    assert widths.min() == pytest.approx(0.05)
    assert widths.max() == pytest.approx(0.3)
    inside = (raw > 0.05) & (raw < 0.3)
    assert inside.any()
    assert np.allclose(widths[inside], raw[inside])
    pair = dc.data_dicts()[0]['deltakp2'][:, :, :, 0]
    assert pair.min() >= 0.05 - 1e-15 and pair.max() <= 0.3 + 1e-15
    assert np.allclose(np.diagonal(pair, axis1=1, axis2=2), 0.05)


@pytest.mark.parametrize('min_width', [None, 0.0])
def test_no_floor_keeps_raw_widths(min_width):
    dc, velocities = _model()
    do_adaptive_smearing(dc, 'gauss', None, min_width=min_width)
    widths = dc.data_dicts()[0]['deltakp'][:, :, 0]
    assert np.allclose(widths, _raw_widths(dc, velocities))
    assert np.all(widths[:, 0] == 0.0)


def test_cap_alone():
    dc, velocities = _model()
    raw = _raw_widths(dc, velocities)
    do_adaptive_smearing(dc, 'gauss', None, max_width=0.3)
    widths = dc.data_dicts()[0]['deltakp'][:, :, 0]
    assert np.allclose(widths, np.minimum(raw, 0.3))


def test_invalid_bounds_raise():
    dc, _ = _model()
    with pytest.raises(ValueError):
        do_adaptive_smearing(dc, 'gauss', None, min_width=-1e-3)
    with pytest.raises(ValueError):
        do_adaptive_smearing(dc, 'gauss', None, min_width=0.1, max_width=0.05)
    with pytest.raises(ValueError):
        do_adaptive_smearing(dc, 'gauss', None, max_width=-0.1)


def _adaptive_count(dc, emin=-6.0, emax=12.0, ne=1801):
    """Integral of the adaptive DOS; ``do_dos_adaptive`` weights every state by 1/nk.

    The window holds every Gaussian with its tails (widths are capped at 0.5 eV below).
    """
    ene = np.linspace(emin, emax, ne)
    with np.errstate(divide='ignore', invalid='ignore'):
        do_dos_adaptive(dc, emin, emax, ne)
    dos = dc.data_dicts()[0]['dosdk']
    return dos, trapezoid(dos, ene)


def test_floored_dos_counts_every_state():
    dc, _ = _model()
    nawf = dc.data_dicts()[1]['nawf']
    do_adaptive_smearing(dc, 'gauss', None, min_width=0.05, max_width=0.5)
    dos, count = _adaptive_count(dc)
    assert np.isfinite(dos).all()
    assert count == pytest.approx(nawf, abs=1e-6)


def test_unfloored_dos_loses_the_flat_band():
    dc, _ = _model()
    nawf = dc.data_dicts()[1]['nawf']
    do_adaptive_smearing(dc, 'gauss', None, max_width=0.5)
    dos, count = _adaptive_count(dc)
    assert (not np.isfinite(dos).all()) or count < nawf - 0.5
