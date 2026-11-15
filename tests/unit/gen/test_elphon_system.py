"""Unit tests of ``paoflow-gen elphon`` (system folder generator) and its driver templates."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.gen import elphon_system as es
from PAOFLOW.gen import paoflow_driver as pd

PB_SCF = """&CONTROL
  calculation     = 'scf'
  prefix          = 'lead',
  pseudo_dir      = './',
  outdir          = './tmp'
  tprnfor         = .true.
/
&SYSTEM
  ibrav           = 2,
  celldm(1)       = 9.5,
  nat             = 1,
  ntyp            = 1,
  ecutwfc         = 60.0
  nosym           = .true.
/
&ELECTRONS
  conv_thr        = 1.0d-10
/
ATOMIC_SPECIES
  Pb 207.2 Pb.upf
ATOMIC_POSITIONS {crystal}
  Pb 0.00 0.00 0.00
K_POINTS {automatic}
  8 8 8  0 0 0
"""

EXAMPLES = Path(__file__).resolve().parents[3] / 'examples' / 'elphon'


def _cfg(**kw):
    cfg = dict(
        name='Lead',
        prefix='lead',
        kgrid=[6, 6, 6],
        qgrid=[3, 3, 3],
        nbnd=16,
        nelec=14,
        masses_amu=[207.2],
        coupling_source='dfpt',
        irreducible=True,
        metal=True,
        pw_template_text=PB_SCF,
        pseudo_dir='../',
        workflows=['phonons', 'coupling', 'superconductivity', 'optics', 'transport', 'arpes'],
        superconductivity=dict(
            nk=12,
            nq=6,
            sigma_ev=0.05,
            mu_star=0.1,
            transport=True,
            fs_coupling=False,
            me_temps=[1.0, 3.0, 5.0],
            me_linear_temps=(1.0, 5.0, 3),
            wscut=0.1,
        ),  # fmt: skip
        optics=dict(
            nk=12,
            nq=6,
            omega_ev=(0.05, 3.0, 0.05),
            temps_k=[300.0],
            dielectric=dict(nfft=24, intraband=True),
        ),  # fmt: skip
        transport=dict(
            temps_k=[100.0, 300.0], rho0_uohm_cm=0.0, b_ee_per_ev=0.0, thickness_um=float('inf')
        ),  # fmt: skip
        arpes=dict(
            photon_ev=21.2,
            work_function_ev=4.25,
            direction=(0.5, 0.0, 0.5),
            window=(-0.1, 0.02, 41),
            self_energy=['isotropic'],
            temps_k=[10.0],
            segment_ev=0.12,
        ),  # fmt: skip
        phonons=dict(units='cm-1', mesh=(8, 8, 8), sigma=3.0),
        coupling=dict(nk=12, direction=(1.0, 0.0, 0.0)),
    )
    cfg.update(kw)
    return cfg


def _load(path):
    namespace = {'__file__': str(path)}
    exec(compile(open(path).read(), str(path), 'exec'), namespace)
    return namespace


def test_render_system_py_executes_to_an_elphon_system(tmp_path):
    text = es.render_system_py(_cfg())
    path = tmp_path / 'system.py'
    path.write_text(text)
    ns = _load(path)
    system = ns['SYSTEM']
    assert system.prefix == 'lead' and system.coarse_grid == (6, 6, 6) and system.qgrid == (3, 3, 3)
    assert system.root == str(tmp_path) and system.save_path.endswith('qe/nscf/lead.save')
    assert system.masses_amu == [207.2] and system.nelec == 14 and system.cache_dir == 'cache'
    assert ns['NAME'] == 'Lead'
    assert ns['SUPERCONDUCTIVITY']['nk'] == 12 and ns['TRANSPORT']['thickness_um'] == float('inf')
    assert ns['OPTICS']['dielectric'] == {'nfft': 24, 'intraband': True}
    assert ns['PHONONS'] == {'units': 'cm-1', 'mesh': (8, 8, 8), 'sigma': 3.0}


def test_qe_inputs_dfpt_irreducible_wedge_and_epsil():
    files = es.qe_inputs(_cfg(metal=False))
    assert sorted(files) == [
        'qe/nscf/nscf.in',
        'qe/phonon/ph.in',
        'qe/phonon/scf.in',
        'qe/run_qe.sh',
    ]
    assert 'K_POINTS automatic\n  6 6 6 0 0 0' in files['qe/nscf/nscf.in']
    assert (
        "prefix          = 'lead'" in files['qe/nscf/nscf.in']
        and 'nbnd            = 16' in files['qe/nscf/nscf.in']
    )
    assert 'nosym' not in files['qe/phonon/scf.in']
    assert (
        'epsil     = .true.' in files['qe/phonon/ph.in']
        and "fildvscf  = 'dvscf'" in files['qe/phonon/ph.in']
    )
    assert (
        'nq1 = 3,' in files['qe/phonon/ph.in'] and 'tr2_ph    = 1.0d-16' in files['qe/phonon/ph.in']
    )
    assert 'epw' not in files['qe/run_qe.sh'] and 'cd ../nscf' in files['qe/run_qe.sh']


def test_qe_inputs_metal_has_no_epsil_and_epw_source_adds_epw_files():
    files = es.qe_inputs(_cfg(metal=True, coupling_source='epw', exclude_bands=[1, 2, 3, 4, 5]))
    assert 'epsil' not in files['qe/phonon/ph.in']
    assert 'K_POINTS crystal\n216\n' in files['qe/nscf/nscf.in']
    assert "bands_skipped = 'exclude_bands = 1:5'" in files['qe/nscf/epw.in']
    assert 'nbndsub     = 11' in files['qe/nscf/epw.in']
    assert (
        "write_placeholder_ukk('lead.ukk', nbnd=16, nk_total=6 * 6 * 6, exclude_bands=[1, 2, 3, 4, 5], nelec=14)"
        in files['qe/nscf/write_ukk.py']
    )
    assert 'write_ukk.py' in files['qe/run_qe.sh'] and 'epw.x' in files['qe/run_qe.sh']


def test_qe_inputs_without_template_write_the_kpoints_card_only():
    files = es.qe_inputs(_cfg(pw_template_text=None, irreducible=False))
    assert 'qe/nscf/nscf.kpoints' in files and 'qe/phonon/scf.in' not in files
    assert files['qe/nscf/nscf.kpoints'].startswith('K_POINTS crystal\n216\n')


def test_qe_inputs_optics_nscf():
    files = es.qe_inputs(_cfg(optics_nscf=True, optics_kgrid=(4, 4, 4), optics_nbnd=40))
    assert 'K_POINTS automatic\n  4 4 4 0 0 0' in files['qe/optics/nscf.in']
    assert 'nbnd            = 40' in files['qe/optics/nscf.in']
    assert 'cd ../optics' in files['qe/run_qe.sh']


def test_workflow_files_metal_vs_semiconductor_transport():
    metal = es.workflow_files(_cfg(metal=True, workflows=['transport']))
    semi = es.workflow_files(_cfg(metal=False, workflows=['transport']))
    assert metal == ['qe/kmesh.py', 'transport/resistivity.py', 'transport/plot_resistivity.py']
    assert semi == ['qe/kmesh.py', 'transport/mobility.py', 'transport/plot_mobility.py']
    assert es.workflow_files(_cfg(metal=True, workflows=['optics'])) == [
        'qe/kmesh.py',
        'optics/dielectric.py',
    ]


def test_write_system_folder_copies_the_templates_and_every_driver_parses_help(
    tmp_path, monkeypatch
):
    written = es.write_system_folder(_cfg(), str(tmp_path))
    names = sorted(os.path.relpath(p, tmp_path) for p in written)
    assert 'system.py' in names and 'qe/run_qe.sh' in names and 'qe/kmesh.py' in names
    for workflow, files in es.WORKFLOWS.items():
        if workflow in ('transport', 'optics'):  # the metal configuration gets the -metal sets
            continue
        for rel in files:
            assert rel in names, rel
            assert (tmp_path / rel).read_bytes() == (Path(es.TEMPLATE_DIR) / rel).read_bytes()
    assert (tmp_path / 'cache').is_dir() and (tmp_path / 'figures').is_dir()
    assert os.access(tmp_path / 'qe' / 'run_qe.sh', os.X_OK)
    # every driver loads system.py and parses --help without touching QE data
    drivers = [p for p in written if p.endswith('.py') and 'qe/' not in p and p.endswith('.py')]
    drivers = [p for p in drivers if os.path.basename(p) != 'system.py']
    assert len(drivers) == 11
    for path in drivers:
        monkeypatch.setattr(sys, 'argv', [path, '--help'])
        with pytest.raises(SystemExit) as info:
            runpy.run_path(path, run_name='__main__')
        assert info.value.code == 0, path
    # second call refuses to overwrite without force
    again = es.write_system_folder(_cfg(), str(tmp_path))
    assert again == []
    assert len(es.write_system_folder(_cfg(), str(tmp_path), force=True)) == len(written)


def test_examples_are_copies_of_the_templates():
    if not EXAMPLES.is_dir():
        pytest.skip('examples/elphon not present')
    checked = 0
    for system in sorted(EXAMPLES.iterdir()):
        if not system.is_dir() or system.name.startswith('_'):
            continue
        for rel in list(sum(es.WORKFLOWS.values(), ())) + list(es.QE_HELPERS):
            example = system / rel
            if example.is_file():
                template = Path(es.TEMPLATE_DIR) / rel
                assert example.read_bytes() == template.read_bytes(), '%s differs from %s' % (
                    example,
                    template,
                )
                checked += 1
    assert checked > 0


def _answers(monkeypatch, answers):
    """Feed the generator prompts from a list of answers (blank = default)."""
    queue = list(answers)

    def fake_input(prompt):
        return queue.pop(0) if queue else ''

    monkeypatch.setattr(pd, '_input', fake_input)


def test_collect_elphon_system_prompts_and_reprompts_incommensurate_q(tmp_path, monkeypatch):
    (tmp_path / 'scf.in').write_text(PB_SCF)
    _answers(
        monkeypatch,
        [
            '',  # pw.x input (detected scf.in)
            '',  # prefix (lead)
            'Pb',  # name
            '',  # pseudo_dir
            '',  # coupling source dfpt
            '6',  # k-grid
            '4',  # q-grid: does not divide 6 -> re-prompt
            '3',  # q-grid
            '',  # irreducible wedge
            '16',  # nbnd
            '',  # masses (207.2 from the template)
            '14',  # nelec
            'y',  # metal
            '',  # basis dir
            '',  # pthr
            'superconductivity, arpes',  # workflows
            '6',  # nq
            '12',  # nk
            '',  # sigma
            '',  # mu*
            'n',  # transport Eliashberg
            'y',  # fs coupling
            '',  # fsthick
            '1, 9, 3',  # ME temps
            '',  # photon
            '',  # work function
            '',  # direction
            '',  # label
        ],
    )

    class Args:
        prefix = None
        out = str(tmp_path / 'pb')

    cfg = es.collect_elphon_system(Args(), str(tmp_path))
    assert cfg['prefix'] == 'lead' and cfg['name'] == 'Pb' and cfg['metal'] is True
    assert cfg['kgrid'] == [6, 6, 6] and cfg['qgrid'] == [3, 3, 3] and cfg['masses_amu'] == [207.2]
    assert cfg['workflows'] == ['superconductivity', 'arpes']
    assert cfg['superconductivity']['me_temps'] == [1.0, 5.0, 9.0]
    assert cfg['superconductivity']['fs_coupling'] is True
    assert cfg['arpes']['self_energy'] == ['isotropic', 'anisotropic']
    assert cfg['pseudo_dir'] == '../../../'  # the template's directory, from <out>/qe/phonon
    written = es.write_system_folder(cfg, cfg['out'])
    assert any(p.endswith('superconductivity/me_aniso.py') for p in written)
    assert any(p.endswith('qe/phonon/ph.in') for p in written)


def _system_folder(tmp_path, **kw):
    folder = tmp_path / 'sys'
    es.write_system_folder(_cfg(**kw), str(folder))
    return folder


def test_me_template_runs_on_an_eliashberg_npz(tmp_path, monkeypatch):
    from PAOFLOW.elphon.eph_kq import THZ_TO_EV, eliashberg_from_modes

    folder = _system_folder(tmp_path)
    out = folder / 'superconductivity' / 'output'
    out.mkdir()
    lam_qv, om = np.array([[1.0]]), np.array([[0.008 / THZ_TO_EV]])
    np.savez(out / 'eliashberg.npz', **eliashberg_from_modes(lam_qv, om), omega_qv_thz=om)
    script = str(folder / 'superconductivity' / 'me.py')
    monkeypatch.setattr(
        sys, 'argv', [script, '--temps', '1', '9', '3', '--linear-temps', '1', '9', '3']
    )
    runpy.run_path(script, run_name='__main__')
    me_dir = out / 'me'
    assert (me_dir / 'lead.imag_iso_001.00').exists() and (me_dir / 'lead.acon_iso_001.00').exists()
    gap = np.loadtxt(me_dir / 'gap_vs_T.dat')
    assert gap.shape == (3, 5) and gap[0, 1] > 0.0 and gap[-1, 1] == 0.0
    assert np.load(me_dir / 'migdal_eliashberg.npz')['Tc_linear'] > 1.0


def test_me_aniso_template_runs_on_a_fs_coupling(tmp_path, monkeypatch):
    from PAOFLOW.elphon import migdal_eliashberg as me
    from PAOFLOW.elphon.eph_kq import THZ_TO_EV
    from PAOFLOW.elphon.fermi_surface_coupling import write_fs_coupling

    folder = _system_folder(tmp_path)
    omega, a2F = me.a2f_from_modes([[1.0]], [[0.01 / THZ_TO_EV]], None, 2.0e-4, 500)
    lam_density = 2.0 * a2F * omega[0] / omega
    shares = np.array([[1.2, 0.2], [0.2, 0.4]])
    out = folder / 'superconductivity' / 'output'
    out.mkdir()
    write_fs_coupling(
        str(out / 'fs_coupling.npz'),
        {
            'coupling': shares[:, :, None] * lam_density, 'freq_ev': omega,
            'weight': np.array([0.3, 0.7]), 'band': np.array([3, 4]),
            'k_cryst': np.zeros((2, 3)), 'energy_ev': np.array([0.01, -0.01]),
            'bg': np.eye(3), 'nk_dense': 4, 'nq_dense': 4, 'fsthick_ev': 0.2,
        },
    )  # fmt: skip
    script = str(folder / 'superconductivity' / 'me_aniso.py')
    monkeypatch.setattr(sys, 'argv', [script, '--temps', '1', '30'])
    runpy.run_path(script, run_name='__main__')
    gap = np.loadtxt(out / 'me_aniso' / 'gap_vs_T_aniso.dat')
    assert gap.shape == (2, 4) and gap[0, 3] > gap[0, 2] > 0.0 and gap[1, 3] == 0.0


def test_resistivity_template_runs_on_a_transport_eliashberg_npz(tmp_path, monkeypatch):
    folder = _system_folder(tmp_path)
    out = folder / 'superconductivity' / 'output'
    out.mkdir()
    omega = np.linspace(0.0005, 0.02, 40)
    a2f = 0.5 * np.exp(-(((omega - 0.008) / 0.002) ** 2))
    np.savez(out / 'eliashberg.npz', omega=omega, a2F=a2f, a2F_tr=0.8 * a2f, lambda_tr=1.0,
             omega_plasma_ev=9.0, **{'lambda': 1.2})  # fmt: skip
    script = str(folder / 'transport' / 'resistivity.py')
    monkeypatch.setattr(sys, 'argv', [script, '--temps', '100', '300', '--rho0', '0.5',
                                      '--omega-max', '0.5', '--omega-step', '0.01'])  # fmt: skip
    runpy.run_path(script, run_name='__main__')
    rows = np.loadtxt(folder / 'transport' / 'output' / 'resistivity.dat')
    assert rows.shape == (2, 4) and rows[1, 1] > rows[0, 1] > 0.5
    assert (folder / 'transport' / 'output' / 'emissivity.npz').exists()


def test_kmesh_template_rewrites_the_nscf_card_and_system_py(tmp_path, monkeypatch):
    folder = _system_folder(tmp_path)
    script = str(folder / 'qe' / 'kmesh.py')
    monkeypatch.setattr(sys, 'argv', [script, '9', '9', '9'])
    runpy.run_path(script, run_name='__main__')
    assert 'K_POINTS automatic\n  9 9 9 0 0 0' in (folder / 'qe' / 'nscf' / 'nscf.in').read_text()
    assert 'coarse_grid=(9, 9, 9)' in (folder / 'system.py').read_text()
    monkeypatch.setattr(sys, 'argv', [script, '8', '8', '8'])  # 3 does not divide 8
    with pytest.raises(SystemExit):
        runpy.run_path(script, run_name='__main__')
