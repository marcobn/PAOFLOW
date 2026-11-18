"""The linear-response routes keep their numbers through the Kubo-Bastin campaign (14 November 2026).

Every case of ``_kubo_cases.CASES`` is re-run and each output file compared with the values frozen before the
campaign (``data/kubo_legacy_reference.npz``). Deliberate changes (bug fixes) are listed in ``CHANGED`` with the
reason; everything else must agree to 1e-10 relative to the largest value of the file. On 18 November 2026 the
constants moved to CODATA 2018 (``utils/constants.py``): the 20 entries that carry e^2/hbar were refreshed then
(relative change 1.69e-5, one 1.19e-5); nothing else changed.
"""

from __future__ import annotations

import numpy as np
import pytest

from . import _kubo_cases as kc

pytestmark = pytest.mark.unit

#: Files whose legacy values were wrong and are fixed on purpose: (case, file name prefix) -> reason.
CHANGED = {
    (
        'kane_mele/spin_Hall',
        'SCD',
    ): 'AC grid was linspace(0, shift) with shift = inf for models (NaN)',
    ('rashba/spin_Hall', 'SCD'): 'AC grid was linspace(0, shift) with shift = inf for models (NaN)',
    (
        'rashba/anomalous_Hall',
        'MCD',
    ): 'AC grid was linspace(0, shift) with shift = inf for models (NaN)',
    (
        'weyl/anomalous_Hall',
        'MCD',
    ): 'AC grid was linspace(0, shift) with shift = inf for models (NaN)',
}


@pytest.fixture(scope='module')
def reference():
    with np.load(kc.REFERENCE) as data:
        return {key: data[key] for key in data.files}


@pytest.mark.parametrize('case', sorted(kc.CASES))
def test_route_keeps_its_numbers(case, reference, tmp_path):
    outputs = kc.run_case(case, str(tmp_path))
    expected = {
        key.split('|', 1)[1]: value
        for key, value in reference.items()
        if key.startswith(case + '|')
    }
    assert expected, 'no reference for %s' % case
    for fname, want in expected.items():
        if any(case == c and fname.startswith(p) for c, p in CHANGED):
            continue
        assert fname in outputs, '%s: %s not written' % (case, fname)
        got = outputs[fname]
        assert got.shape == want.shape, (case, fname, got.shape, want.shape)
        scale = max(np.nanmax(np.abs(want)), 1e-300)
        assert np.allclose(got, want, rtol=1e-10, atol=1e-10 * scale, equal_nan=True), (
            case,
            fname,
            np.nanmax(np.abs(got - want)) / scale,
        )
