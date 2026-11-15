"""Smoke test of the example workflow drivers on the Si DFPT asset (6^3 k, 3^3 q, small dense grids).

Builds a system folder with the drivers of ``PAOFLOW/gen/elphon_templates`` (the files of
``examples/elphon/<System>``) whose ``system.py`` points at the asset, and runs every workflow
in-process (``runpy`` under ``sys.argv``), the way a user runs ``python <workflow>/<driver>.py``.
The dense grids are tiny, so the numbers are not converged; the test checks that each driver
runs end to end on the shared cache and writes its output files.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.gen import elphon_system as es

pytestmark = pytest.mark.integration
BASIS = str(Path(__file__).resolve().parents[3] / 'BASIS')


@pytest.fixture(scope='module')
def system_folder(elphon_assets, tmp_path_factory):
    """A system folder on the Si asset: every workflow, grids of the asset, PAOFLOW's BASIS."""
    root = tmp_path_factory.mktemp('si_workflows')
    cfg = dict(
        name='Si', prefix='si', kgrid=[6, 6, 6], qgrid=[3, 3, 3], masses_amu=[28.0855], nelec=8,
        nbnd=12, pthr=0.95, coupling_source='dfpt', metal=False, basisdir=BASIS,
        workflows=['phonons', 'coupling', 'superconductivity', 'optics', 'transport', 'arpes'],
        superconductivity=dict(nk=6, nq=3, sigma_ev=0.05, mu_star=0.1, me_temps=[1.0, 5.0],
                               me_linear_temps=(1.0, 5.0, 2)),
        optics=dict(nk=6, nq=3, omega_ev=(0.5, 3.0, 0.5), temps_k=[300.0], degauss_ev=0.05,
                    dielectric=dict(nfft=4)),
        transport=dict(nk=6, nq=6, temps_k=[300.0], window_ev=0.3, smearing_ev=0.02),
        arpes=dict(photon_ev=21.2, work_function_ev=4.8, direction=(0.5, 0.0, 0.5), direction_label='X',
                   window=(-13.0, 1.5, 61), resolution=(0.05, 0.0), self_energy=['fan-migdal'],
                   temps_k=[300.0], segment_ev=None, npts=4, nq=3, eta_ev=0.05, intensity_power=0.3),
        phonons=dict(units='cm-1', mesh=(4, 4, 4), sigma=3.0),
        coupling=dict(nk=6, direction=(1.0, 0.0, 0.0)),
    )  # fmt: skip
    es.write_system_folder(cfg, str(root))
    # the asset holds the pp.py layout (save/ with dvscf, dyn and patterns) and the full-grid nscf
    system_py = root / 'system.py'
    text = system_py.read_text().replace(
        "    qe_dir='qe',",
        "    qe_dir='qe',\n    phonon_dir=%r,\n    save_dir=%r,"
        % (os.path.join(elphon_assets, 'save'), os.path.join(elphon_assets, 'si.save')),
    )
    system_py.write_text(text)
    return root


def _run(folder, workflow, driver, *args, monkeypatch):
    script = str(folder / workflow / driver)
    monkeypatch.setattr(sys, 'argv', [script, *args])
    runpy.run_path(script, run_name='__main__')
    return folder / workflow / 'output'


def test_system_py_loads(system_folder):
    from PAOFLOW.elphon.workflows import load_system

    module = load_system(str(system_folder / 'optics'))
    module.SYSTEM.check_inputs()
    assert module.SYSTEM.coupling_dir.endswith('save') and module.OPTICS['nk'] == 6


def test_superconductivity_workflow(system_folder, monkeypatch):
    out = _run(
        system_folder,
        'superconductivity',
        'eliashberg.py',
        '--nk',
        '6',
        '--nq',
        '3',
        monkeypatch=monkeypatch,
    )
    data = np.load(out / 'eliashberg.npz')
    assert np.isfinite(data['lambda']) and (out / 'alpha2F.dat').exists()
    assert list((system_folder / 'cache').glob('*.npy')), 'the coarse vertex is cached'
    _run(system_folder, 'superconductivity', 'me.py', '--no-linear', monkeypatch=monkeypatch)
    assert (out / 'me' / 'gap_vs_T.dat').exists()


def test_optics_workflow(system_folder, monkeypatch):
    out = _run(
        system_folder, 'optics', 'absorption.py', '--nk', '6', '--nq', '3', monkeypatch=monkeypatch
    )
    data = np.load(out / 'absorption.npz', allow_pickle=True)
    assert (
        float(data['indirect_gap_dft_ev']) > 0.3 and (out / 'epsilon2_indabs_300.0K.dat').exists()
    )
    monkeypatch.setattr('matplotlib.pyplot.show', lambda *a, **k: None, raising=False)
    _run(system_folder, 'optics', 'plot_absorption.py', monkeypatch=monkeypatch)
    assert (out / 'absorption.png').exists()


def test_transport_workflow(system_folder, monkeypatch):
    out = _run(
        system_folder, 'transport', 'mobility.py', '--nk', '6', '--nq', '6', monkeypatch=monkeypatch
    )
    rows = np.loadtxt(out / 'mobility.dat', ndmin=2)
    assert (
        rows.shape[0] == 1 and np.all(np.isfinite(rows)) and (out / 'scattering_rates.npz').exists()
    )


def test_arpes_workflow(system_folder, monkeypatch):
    out = _run(
        system_folder,
        'arpes',
        'arpes.py',
        '--nq',
        '3',
        '--npts',
        '4',
        '--temps',
        '300',
        '--no-plot',
        monkeypatch=monkeypatch,
    )
    result = np.load(out / 'self_energy_fan_migdal.npz')
    assert result['sigma_ev'].shape[1] == 4 and np.all(result['sigma_ev'].imag <= 1e-9)
    assert (out / 'arpes_bare.npz').exists() and (out / 'arpes_fan_migdal_300K.npz').exists()


@pytest.mark.slow
def test_coupling_workflow(system_folder, monkeypatch):
    # three minutes on four cores: the plain and the model variant each build the coarse
    # vertex from the dvscf files (the long-range physics is covered by test_long_range_sic.py)
    # the asset's ph.x run has epsil: Si is non-polar (|Z*| < 0.01), so the model and the plain
    # curves coincide for the LO mode and differ by the residual dipole term (< 1 %) elsewhere
    out = _run(
        system_folder,
        'coupling',
        'long_range_check.py',
        '--nk',
        '6',
        '--no-plot',
        monkeypatch=monkeypatch,
    )
    rows = np.loadtxt(out / 'g_path.dat', ndmin=2)
    assert rows.shape == (3, 8) and np.all(np.isfinite(rows))
    assert np.allclose(rows[:, 1], rows[:, 4], rtol=1e-4)
    assert np.allclose(rows[:, 1:4], rows[:, 4:7], rtol=1e-2)


def test_phonons_workflow_needs_the_dyn0_layout(system_folder, monkeypatch):
    # the pp.py save/ layout has no <prefix>.dyn0: the q2r + matdyn driver reports it
    with pytest.raises((FileNotFoundError, OSError)):
        _run(system_folder, 'phonons', 'phonons.py', '--no-plot', monkeypatch=monkeypatch)
