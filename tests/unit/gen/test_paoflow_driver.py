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


def test_run_script_orbital_hall_main_and_plot():
    """orbital_Hall is a selectable property: the run script calls it with the energy window and the
    sigma^{L_z}_{xy} element, the plot script gets a menu entry reading the ohcEf files."""
    cfg = _run_cfg(ibrav=2, is_2d=False, properties=['bands', 'spin_Hall', 'orbital_Hall'])
    main = d.build_run_script(cfg)
    compile(main, 'main.py', 'exec')
    assert 'p.orbital_Hall(emin=EMIN, emax=EMAX, o_tensor=[[0, 1, 2]])' in main
    assert 'EMIN, EMAX, NE = ' in main
    plot = d.build_plot_script(cfg)
    compile(plot, 'plot.py', 'exec')
    assert 'def plot_orbital_hall' in plot and "_many('*.ohcEf*.dat')" in plot
    assert 'pplt.plot_ohc(' in plot and 'Orbital Hall conductivity' in plot
    assert ('orbital_Hall', 'Orbital Hall conductivity') in d.PROPERTY_MENU
    assert 'orbital_Hall' in d.ENERGY_WINDOW_PROPERTIES


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
    assert "MPI_QE = mpi_prefix('mpirun -np 16')" in text  # PAOFLOW_MPI_QE first
    assert 'from PAOFLOW.utils.qe_launch import mpi_prefix, qe_command' in text
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
# Electron-phonon workflow (paoflow-gen elphon writes a system folder, see
# test_elphon_system.py); only the EPW a2f reader used by its drivers is tested here.
# --------------------------------------------------------------------------- #


def test_read_epw_a2f_helper_parses_epw_format(tmp_path):
    a2f = tmp_path / 'pb.a2f'
    a2f.write_text(
        ' w[meV] a2f and integrated 2*a2f/w for   10 smearing values\n'
        '   0.5000000   0.1000000   0.0100000\n'
        '   1.0000000   0.2000000   0.0500000\n'
        ' Integrated el-ph coupling\n  #            1.1514582\n'
    )
    from PAOFLOW.elphon.qe_elph_io import read_epw_a2f

    w, a2f_values, lam = read_epw_a2f(str(a2f))
    assert list(w) == [0.5, 1.0] and list(a2f_values) == [0.1, 0.2] and lam[-1] == 0.05


def test_build_run_script_nonlinear_optics_block():
    cfg = _run_cfg(
        ibrav=2, is_2d=False, properties=['bands', 'nonlinear'], nlo_nk=24, nlo_emax=3.5,
        nlo_eta=0.04, nlo_scissor=0.86, nlo_components=['xyz'],
    )  # fmt: skip
    text = d.build_run_script(cfg)
    compile(text, 'main.py', 'exec')
    assert 'def run_nonlinear():' in text and '    run_nonlinear()' in text
    assert 'NLO_NK = (24, 24, 24)' in text and 'NLO_SCISSOR = 0.86' in text
    assert "NLO_COMPONENTS = ['xyz']" in text
    # the validated defaults: extended basis of the run's pseudopotentials, position dipoles
    assert 'NLO_DIPOLES = True' in text and 'NLO_SHG_TERMS = False' in text
    block = text[text.index('def run_nonlinear():') :].split('def main')[0]
    assert "configuration='extended'" in block
    assert 'p.nonlinear_optics(' in block and 'shg_terms=NLO_SHG_TERMS' in block
    assert 'coupling=(NLO_COUPLING if NLO_MOLECULE else None)' in block
    # the non-linear run has its own PAOFLOW instance and never interpolates (HRs must survive)
    assert 'p.interpolated_hamiltonian(' not in block
    assert 'p.coupled_optical_response(' not in block
    assert (
        'nonlinear',
        'Non-linear optics (shift / injection currents, SHG) on the PAO Hamiltonian',
    ) in d.PROPERTY_MENU
    plot = d.build_plot_script(cfg)
    compile(plot, 'plot.py', 'exec')
    assert 'def plot_shg' in plot and "pplt.plot_nonlinear(OUTPUTDIR, 'shg'" in plot
    assert 'plot_shift_current' in plot and 'plot_injection_current' in plot
    assert 'plot_shg_terms' not in plot and 'plot_coupled' not in plot


def test_build_run_script_nonlinear_terms_coupled_molecule_and_2d():
    cfg = _run_cfg(
        ibrav=2, is_2d=False, properties=['nonlinear'], nlo_nk=24, nlo_shg_terms=True,
        nlo_coupled=True, coupled_nk=12, coupled_kernels=('none', 'hartree'),
        coupled_shg_energies=[0.1, 1.0], nlo_responses=('shg',),
    )  # fmt: skip
    text = d.build_run_script(cfg)
    compile(text, 'main.py', 'exec')
    assert 'NLO_SHG_TERMS = True' in text and "NLO_RESPONSES = ('shg',)" in text
    assert 'COUPLED_NK = (12, 12, 12)' in text and "COUPLED_KERNELS = ('none', 'hartree')" in text
    assert 'COUPLED_SHG_ENERGIES = [0.1, 1.0]' in text and 'p.coupled_optical_response(' in text
    plot = d.build_plot_script(cfg)
    compile(plot, 'plot.py', 'exec')
    for name in ('plot_shg_terms', 'plot_coupled_dielectric', 'plot_coupled_shg'):
        assert 'def %s():' % name in plot
    # a molecule: Gamma point, every component, the coupled response of the molecule, beta plots
    mol = _run_cfg(
        ibrav=1, is_2d=False, properties=['nonlinear'], nlo_molecule=True, nlo_coupling='hartree',
    )  # fmt: skip
    text = d.build_run_script(mol)
    compile(text, 'main.py', 'exec')
    assert 'NLO_NK = (1, 1, 1)' in text and 'NLO_MOLECULE = True' in text
    assert "NLO_COUPLING = 'hartree'" in text and "NLO_COMPONENTS = 'all'" in text
    assert 'def plot_molecular_beta():' in d.build_plot_script(mol)
    # a 2D material: sheet units in main.py and in the plots
    sheet = _run_cfg(ibrav=4, is_2d=True, properties=['nonlinear'])
    assert 'NLO_TWOD = True' in d.build_run_script(sheet)
    assert 'two_d=True' in d.build_plot_script(sheet)
    # projwfc projections: no internal basis, hence no dipoles
    projwfc = _run_cfg(ibrav=2, is_2d=False, properties=['nonlinear'], projector='projwfc')
    assert 'NLO_DIPOLES = False' in d.build_run_script(projwfc)


def test_collect_nonlinear_prompts(monkeypatch):
    answers = iter(['n', '30', '0.0', '5.0', '501', '0.1', '0.86', 'xyz', 'shg', 'y', 'y', 'n', 'y',
                    '12', 'none,hartree,hartree+alda', '5.5', '56', '0.1,1.0', '0.1'])  # fmt: skip
    monkeypatch.setattr('builtins.input', lambda *_: next(answers))
    cfg = d.collect_nonlinear({'is_2d': False, 'projector': 'internal'})
    assert cfg['nlo_nk'] == 30 and cfg['nlo_ne'] == 501 and cfg['nlo_components'] == 'xyz'
    assert cfg['nlo_responses'] == ('shg',) and cfg['nlo_dipoles'] and cfg['nlo_shg_terms']
    assert cfg['nlo_coupled'] and cfg['coupled_kernels'] == ('none', 'hartree', 'hartree+alda')
    assert cfg['coupled_shg_energies'] == [0.1, 1.0] and cfg['coupled_ne'] == 56
    assert cfg['nlo_metal'] is False and 'nlo_temperature' not in cfg


def test_build_run_script_metal_block():
    cfg = _run_cfg(
        ibrav=2, is_2d=False, properties=['nonlinear'], nlo_metal=True, nlo_temperature=500.0,
        nlo_tau=15.0, nlo_coherence_damping=2.0, nlo_coupled=True, coupled_kernels=('none', 'hartree'),
        coupled_shg_energies=[],
    )  # fmt: skip
    text = d.build_run_script(cfg)
    assert 'NLO_FERMI_SURFACE = True' in text and 'NLO_TEMPERATURE = 500.0' in text
    assert 'NLO_TAU = 15.0' in text and 'NLO_COHERENCE_DAMPING = 2.0' in text
    block = text[text.index('def run_nonlinear():') :].split('def main')[0]
    assert 'fermi_surface=NLO_FERMI_SURFACE' in block
    assert block.count('temperature=NLO_TEMPERATURE, relaxation_time=NLO_TAU') == 2
    assert block.count('coherence_damping=NLO_COHERENCE_DAMPING') == 1  # not the coupled response
    insulator = d.build_run_script(_run_cfg(ibrav=2, is_2d=False, properties=['nonlinear']))
    assert "NLO_FERMI_SURFACE = 'auto'" in insulator and 'NLO_TEMPERATURE = 0.0' in insulator
    assert 'NLO_COHERENCE_DAMPING = 1.0' in insulator
    plot = d.build_plot_script(cfg)
    assert 'def plot_photocurrent_channels' in plot and 'def plot_shg_terms' in plot


def test_plot_script_hall_coefficient_reads_smeared_three_column_file(tmp_path, monkeypatch):
    """The Hall entry finds hall_trace_<smearing>_<spin>.dat (the name do_transport writes when a
    smearing is set) and plots column 2 (R_H) against column 1 (E), one curve per temperature."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cfg = _run_cfg(ibrav=2, is_2d=False, properties=['transport'])
    plot = d.build_plot_script(cfg)
    assert "_spin_channels('hall_trace*_0.dat')" in plot
    rows = [
        (300.0, -0.5, 1.0e-9),
        (300.0, 0.0, 2.0e-9),
        (600.0, -0.5, 3.0e-9),
        (600.0, 0.0, 4.0e-9),
    ]
    (tmp_path / 'hall_trace_gauss_0.dat').write_text(
        ''.join('%8.2f % .5f % 9.5e \n' % row for row in rows)
    )
    namespace = {'__name__': 'generated_plot', '__file__': str(tmp_path / 'plot.py')}
    exec(compile(plot, 'plot.py', 'exec'), namespace)
    namespace['OUTPUTDIR'] = str(tmp_path)
    monkeypatch.setattr(plt, 'show', lambda *a, **k: None)
    namespace['plot_hall_coefficient']()
    lines = plt.gcf().axes[0].get_lines()
    assert [line.get_label() for line in lines] == ['300 K', '600 K']
    assert list(lines[1].get_xdata()) == [-0.5, 0.0]
    assert list(lines[1].get_ydata()) == [3.0e-9, 4.0e-9]
    plt.close('all')
