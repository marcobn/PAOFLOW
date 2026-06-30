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


# --------------------------------------------------------------------------- #
# Raman workflow generation
# --------------------------------------------------------------------------- #
def _raman_cfg(**kw):
    cfg = dict(_COMMON)
    cfg.update(
        {
            'prefix': 'Si2',
            'savedir': 'Si2.save',
            'supercell': 2,
            'displacement': 0.01,
            'mesh': 16,
            'units': 'cm-1',
            'pp_dir': 'Si2.save',
            'hubbard_file': None,
            'mpi_qe': 'mpirun -np 16',
            'qe_path': '~/Local/Programs/qe-7.4.1/bin/',
            'raman': True,
            'raman_delta': 0.05,
            'raman_nbnd': 21,
            'raman_npool': 4,
            'raman_smearing': 'gauss',
            'raman_nfft': 24,
            'raman_e_static': 0.05,
            'raman_temperature': 300.0,
            'raman_gamma': 4.0,
            'raman_laser_nm': None,
            'raman_pthr': 0.95,
            'raman_configuration': 'extended',
        }
    )
    cfg.update(kw)
    return cfg


def test_build_raman_script_phases_and_calls():
    text = d.build_raman_script(_raman_cfg())
    # three-phase driver
    assert "choices=['generate', 'run', 'analyse', 'all']" in text
    assert 'def generate():' in text
    assert 'def run_cells():' in text
    assert 'def analyse():' in text
    # generate vs. analyse both call raman_spectrum with the right generate flag
    assert 'generate=True,' in text
    assert 'generate=False,' in text
    assert 'p.raman_spectrum(' in text
    # analyse passes the optical-pipeline settings
    assert 'basispath=BASISPATH,' in text
    assert 'p.finish_execution()' in text


def test_build_raman_script_constants():
    text = d.build_raman_script(_raman_cfg())
    assert 'SUPERCELL_MATRIX = 2' in text
    assert 'DELTA = 0.05' in text
    assert 'NBND = 21' in text
    assert 'NFFT = (24, 24, 24)' in text
    assert "PREFIX = 'Si2'" in text
    assert "MPI_QE = 'mpirun -np 16'" in text
    assert 'LASER_NM = None' in text


def test_build_raman_script_nbnd_default_and_no_nfft():
    text = d.build_raman_script(_raman_cfg(raman_nbnd=0, raman_nfft=0))
    assert 'NBND = None' in text
    assert 'NFFT = None' in text


def test_build_raman_script_laser_value():
    text = d.build_raman_script(_raman_cfg(raman_laser_nm=532.0))
    assert 'LASER_NM = 532.0' in text


def test_build_raman_plot_script():
    text = d.build_raman_plot_script(_raman_cfg())
    assert 'plot_raman_spectrum(' in text
    assert '_raman_spectrum.dat' in text
    assert '_raman_modes.dat' in text
    assert '--xmin' in text and '--xmax' in text and '--save' in text
