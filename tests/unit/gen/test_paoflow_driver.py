"""Unit tests for the 2D-aware band-path generation in paoflow_driver."""

from __future__ import annotations

from PAOFLOW.gen import paoflow_driver as d

_COMMON = {
    'savedir': 'pwscf.save',
    'prefix': 'pwscf',
    'upfs': ['C.upf'],
    'basisdir': 'BASIS_PS',
    'outputdir': 'output',
}


def _run_cfg(**kw):
    cfg = dict(_COMMON)
    cfg.update(
        {
            'ibrav': 4,
            'is_2d': True,
            'properties': ['bands'],
            'std_basis': 'standard',
            'npool': 1,
            'smearing': 'gauss',
            'spin_orbit': False,
            'nk': 400,
            'emin': -8.0,
            'emax': 4.0,
            'ne': 1000,
            'do_pdos': False,
            'interpolate': False,
            'nfft': 0,
        }
    )
    cfg.update(kw)
    return cfg


def test_band_path_2d_known_lattices():
    assert d._band_path_2d(4)[0] == 'gG-M-K-gG'
    assert d._band_path_2d(6)[0] == 'gG-X-M-gG'
    assert d._band_path_2d(8)[0] == 'gG-X-S-Y-gG'
    # in-plane points must all lie in the kz = 0 plane
    for ibrav in (4, 6, 8, 9, 12):
        _, high_sym = d._band_path_2d(ibrav)
        assert all(pt[2] == 0.0 for pt in high_sym.values())


def test_band_path_2d_unknown_returns_none():
    assert d._band_path_2d(3) == (None, None)


def test_run_script_2d_emits_inplane_path():
    text = d.build_run_script(_run_cfg(ibrav=4, is_2d=True))
    assert "BAND_PATH = 'gG-M-K-gG'" in text
    assert "'K': (0.3333333333333333, 0.3333333333333333, 0.0)" in text
    assert 'p.bands(ibrav=IBRAV, nk=NK, band_path=BAND_PATH,' in text
    assert 'in-plane band path only' in text


def test_run_script_3d_uses_default_path():
    text = d.build_run_script(_run_cfg(ibrav=2, is_2d=False))
    assert 'BAND_PATH' not in text
    assert "p.bands(ibrav=IBRAV, nk=NK, fname='bands')" in text


def test_run_script_2d_unknown_ibrav_todo():
    text = d.build_run_script(_run_cfg(ibrav=3, is_2d=True))
    assert 'No built-in 2D path for ibrav=3' in text
    assert 'BAND_PATH = None' in text
    # the bands call still references the (None) constants so the user can fill in
    assert 'band_path=BAND_PATH' in text


def test_wants_explicit_band_path():
    assert d._wants_explicit_band_path({'ibrav': 0}) is True
    assert d._wants_explicit_band_path({'ibrav': 4, 'is_2d': True}) is True
    assert d._wants_explicit_band_path({'ibrav': 4, 'is_2d': False}) is False
    assert d._wants_explicit_band_path({'ibrav': 4}) is False
