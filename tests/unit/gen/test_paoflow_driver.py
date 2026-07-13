"""Unit tests for the 2D-aware band-path generation in paoflow_driver."""

from __future__ import annotations

import pytest

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
    assert '_raman_spectrum.dat' in text
    assert '_raman_modes.dat' in text
    assert '--xmin' in text and '--xmax' in text and '--save' in text
    # Plots are drawn directly with matplotlib so multiple curves share one axes.
    assert 'import matplotlib.pyplot as plt' in text
    assert 'ax.plot(freq' in text


def test_build_raman_plot_script_overlays_all_spectra():
    text = d.build_raman_plot_script(_raman_cfg())
    # The plot script globs for every spectrum and overlays them on a single
    # axes (with a legend) so 'all' / excitation-profile outputs share one plot.
    assert 'import glob' in text
    assert "glob.glob(os.path.join(OUTPUTDIR, FNAME + '*_raman_spectrum.dat'))" in text
    assert 'fig, ax = plt.subplots()' in text
    assert 'for spectrum in spectra:' in text
    assert 'ax.legend(' in text
    assert '--normalize' in text and '--sticks' in text


def test_build_raman_plot_script_excitation_profile():
    text = d.build_raman_plot_script(_raman_cfg())
    # --excitation plots mode intensity vs laser energy (eV) from the per-laser
    # *_raman_modes.dat files (static channel skipped, only active modes drawn).
    assert '--excitation' in text
    assert 'def plot_excitation(args):' in text
    assert 'def _laser_ev(label):' in text
    assert 'EV_NM = 1239.841984' in text
    assert "glob.glob(os.path.join(OUTPUTDIR, FNAME + '*_raman_modes.dat'))" in text
    assert "ax.set_xlabel('Laser energy (eV)'" in text
    assert 'if args.excitation:' in text


def test_build_raman_script_method_static_default():
    text = d.build_raman_script(_raman_cfg())
    assert "METHOD = 'static'" in text
    assert 'method=METHOD,' in text
    assert 'lifetime=LIFETIME,' in text


def test_build_raman_script_method_resonance_list():
    text = d.build_raman_script(
        _raman_cfg(raman_method='resonance', raman_laser_nm=[488.0, 532.0], raman_lifetime=0.2)
    )
    assert "METHOD = 'resonance'" in text
    assert 'LASER_NM = [488.0, 532.0]' in text
    assert 'LIFETIME = 0.2' in text
    assert 'resonance Raman workflow' in text


def test_build_raman_script_method_all_title():
    text = d.build_raman_script(_raman_cfg(raman_method='all', raman_laser_nm=[532.0]))
    assert "METHOD = 'all'" in text
    assert 'static + resonance Raman workflow' in text


def test_parse_laser_list_comma_and_expression():
    assert d.parse_laser_list('488, 514.5, 532') == [488.0, 514.5, 532.0]
    assert d.parse_laser_list('532') == [532.0]
    profile = d.parse_laser_list('[n for n in range(450, 650, 5)]')
    assert profile[0] == 450.0
    assert profile[-1] == 645.0
    assert len(profile) == 40


def test_parse_laser_list_rejects_unsafe_input():
    with pytest.raises(ValueError):
        d.parse_laser_list('__import__("os").system("echo hi")')
    # Blank input yields an empty list (the caller decides what to do).
    assert d.parse_laser_list('   ') == []


# --------------------------------------------------------------------------- #
# Phonon workflow generation
# --------------------------------------------------------------------------- #
def _phonon_cfg(**kw):
    cfg = dict(_COMMON)
    cfg.update(
        {
            'prefix': 'Mg1O1',
            'savedir': 'Mg1O1.save',
            'ibrav': 2,
            'supercell': 2,
            'displacement': 0.06,
            'mesh': [12, 12, 12],
            'units': 'cm-1',
            'do_thermal': True,
            'pp_dir': 'HERE',
            'hubbard_file': None,
            'born': True,
            'born_method': 'dfpt',
            'vibdielectric': True,
            'vibdielectric_gamma': 4.0,
            'vibdielectric_emissivity': True,
            'vibdielectric_emis_temp': [300.0],
            'mpi_qe': 'mpirun -np 4',
            'qe_path': '',
        }
    )
    cfg.update(kw)
    return cfg


def test_build_phonon_script_vibrational_dielectric_wired():
    text = d.build_phonon_script(_phonon_cfg())
    # Constants block exposes the toggle, damping and output sub-directory.
    assert 'VIBDIELECTRIC = True' in text
    assert 'VIBDIELECTRIC_GAMMA = 4.0' in text
    assert "VIBDIELECTRIC_DIR = 'vibdielectric'" in text
    # The analyse phase calls vibrational_dielectric, gated on NAC + the toggle.
    assert 'if nac and VIBDIELECTRIC:' in text
    assert 'p.vibrational_dielectric(' in text
    assert 'gamma=VIBDIELECTRIC_GAMMA,' in text
    assert 'outdir=VIBDIELECTRIC_DIR,' in text


def test_build_phonon_script_emissivity_wired():
    text = d.build_phonon_script(_phonon_cfg())
    # Emissivity toggle + temperature constants and call arguments.
    assert 'VIBDIELECTRIC_EMISSIVITY = True' in text
    assert 'VIBDIELECTRIC_EMIS_TEMP = [300.0]' in text
    assert 'emissivity=VIBDIELECTRIC_EMISSIVITY,' in text
    assert 'emis_temperature=VIBDIELECTRIC_EMIS_TEMP,' in text


def test_build_phonon_script_emissivity_disabled():
    text = d.build_phonon_script(_phonon_cfg(vibdielectric_emissivity=False))
    assert 'VIBDIELECTRIC_EMISSIVITY = False' in text
    # The argument is still threaded so the toggle alone controls it.
    assert 'emissivity=VIBDIELECTRIC_EMISSIVITY,' in text


def test_build_phonon_script_vibrational_dielectric_disabled():
    text = d.build_phonon_script(_phonon_cfg(vibdielectric=False))
    assert 'VIBDIELECTRIC = False' in text
    # The call is still emitted but guarded so it never runs when disabled.
    assert 'if nac and VIBDIELECTRIC:' in text


def test_build_phonon_plot_script_includes_reststrahlen():
    text = d.build_phonon_plot_script(_phonon_cfg())
    assert "os.path.join(OUTPUTDIR, 'vibdielectric')" in text
    assert "os.path.isfile(os.path.join(vibdir, 'epsr_xx.dat'))" in text
    assert 'pplt.plot_optical(' in text
    # The reststrahlen (phonon) emissivity is plotted when present.
    assert "os.path.isfile(os.path.join(vibdir, 'emish_xx.dat'))" in text
    assert "['emish']" in text


# --------------------------------------------------------------------------- #
# Electron-phonon (PAO route) workflow generation
# --------------------------------------------------------------------------- #
def _elphon_cfg(**kw):
    cfg = dict(_COMMON)
    cfg.update(
        {
            'prefix': 'lead',
            'savedir': 'lead.save',
            'source': 'ahc',
            'coupling_dir': 'ahc_dir',
            'kgrid': [9, 9, 9],
            'qgrid': [3, 3, 3],
            'nbnd': 22,
            'masses_amu': [207.2],
            'nelec': 14,
            'nk_dense': 18,
            'sigma_ry': 0.02,
            'mu_star': 0.10,
            'pthr': 0.90,
            'q_weights': [],
        }
    )
    cfg.update(kw)
    return cfg


def test_build_elphon_script_compiles_and_wires_ahc():
    text = d.build_elphon_script(_elphon_cfg())
    compile(text, 'main.elphon.py', 'exec')  # must be valid Python
    assert "SOURCE = 'ahc'" in text
    assert 'KGRID = (9, 9, 9)' in text
    assert 'QGRID = (3, 3, 3)' in text
    assert 'MASSES_AMU = [207.2]' in text
    assert 'NELEC = 14' in text
    assert "COUPLING_DIR = os.path.join(HERE, 'ahc_dir')" in text
    # the analyse phase calls the AO driver, and no tokens are left unsubstituted
    assert 'eliashberg_from_qe_coupling(' in text
    assert 'source=SOURCE,' in text
    for tok in ('__SAVEDIR__', '__SOURCE__', '__KGRID__', '__MASSES__', '__QWEIGHTS__'):
        assert tok not in text


def test_build_elphon_script_elphmat_source():
    text = d.build_elphon_script(
        _elphon_cfg(source='elphmat', coupling_dir='elph_dir', q_weights=[1, 8, 6, 12])
    )
    assert "SOURCE = 'elphmat'" in text
    assert "COUPLING_DIR = os.path.join(HERE, 'elph_dir')" in text
    assert 'Q_WEIGHTS = [1.0, 8.0, 6.0, 12.0]' in text
