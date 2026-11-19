"""Unit tests of the ``paoflow-gen`` SSCHA workflow (``main.sscha.py``).

The generated driver is loaded as a module and run end to end with its three launchers on a harmonic model: the
PAOFLOW object is the gapped-graphene model carrying the delta-TaN species, the pw.x command is a stand-in that
writes pw.x-format outputs with the forces -Phi u of the converged SSCHA matrices of the test data (so the first
population converges), and ``sbatch`` is a script that records its arguments.
"""

from __future__ import annotations

import importlib.util
import os
import stat
import sys
import types

import numpy as np
import pytest

from PAOFLOW.gen import sscha_driver as sd

pytestmark = pytest.mark.unit

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'phonon', 'data', 'sscha_tan'
)
SSCHA = os.path.join(DATA, 'sscha_T2800', 'tan.dyn')
BOHR_ANGSTROM = 0.529177210903

FAKE_PW = r"""import os, sys
import numpy as np
BOHR = 0.529177210903
phi = np.load(sys.argv[1]); reference = np.load(sys.argv[2])
text = open(sys.argv[sys.argv.index('-in') + 1]).read()
prefix = text.split("prefix = '")[1].split("'")[0]
os.makedirs('tmp_' + prefix, exist_ok=True)  # pw.x scratch: the driver must remove it
cell = np.array([[float(x) for x in r.split()] for r in text.split('CELL_PARAMETERS angstrom\n')[1].splitlines()[:3]]) / BOHR
rows = text.split('ATOMIC_POSITIONS angstrom\n')[1].split('K_POINTS')[0].strip().splitlines()
symbols = [r.split()[0] for r in rows]
positions = np.array([[float(x) for x in r.split()[1:]] for r in rows]) / BOHR
u = (positions - reference).reshape(-1)
force = -(phi @ u)
out = ['     Program PWSCF v.7.5 starts', '     celldm(1)=   1.000000  celldm(2)=   0.000000',
       '     number of atoms/cell      = %12d' % len(symbols), '     number of atomic types    =            2',
       '     crystal axes: (cart. coord. in units of alat)']
out += ['               a(%d) = ( %12.8f %12.8f %12.8f )' % (i + 1, *c) for i, c in enumerate(cell)]
out += ['', '     site n.     atom                  positions (alat units)']
out += ['     %4d           %-3s tau(%4d) = ( %14.9f %14.9f %14.9f  )' % (i + 1, s, i + 1, *p)
        for i, (s, p) in enumerate(zip(symbols, positions))]
out += ['', '!    total energy              = %18.8f Ry' % (0.5 * u @ phi @ u), '',
        '     Forces acting on atoms (cartesian axes, Ry/au):', '']
out += ['     atom %4d type  1   force = %14.8f %14.8f %14.8f' % (i + 1, *f) for i, f in enumerate(force.reshape(-1, 3))]
out += ['', '     Total force = %12.6f' % np.linalg.norm(force), '', '   JOB DONE.']
print('\n'.join(out))
"""

FAKE_SBATCH = """#!/bin/sh
echo "$@" >> "$SBATCH_LOG"
echo "Welcome to the cluster"
echo 4242
"""


def _cfg(**overrides) -> dict:
    cfg = dict(
        savedir='tan.save',
        prefix='tan',
        outputdir='output',
        pp_dir='/pseudo',
        temperatures=[2800.0],
        supercell=2,
        kgrid=[2, 2, 2],
        electronic_temperature='ionic',
        population_sizes=[20],
        hessian_size=20,
        harmonic_dyn=SSCHA,
        launcher='local',
        mpi_qe='',
        qe_pw='pw.x',
    )
    cfg.update(overrides)
    return cfg


def test_build_sscha_script_constants_and_syntax():
    text = sd.build_sscha_script(
        _cfg(supercell=[2, 2, 3], electronic_temperature=1500.0, harmonic_dyn='ph/x.dyn')
    )
    compile(text, 'main.sscha.py', 'exec')
    for expected in (
        'TEMPERATURES = [2800.0]',
        'SUPERCELL = (2, 2, 3)',
        'KGRID = (2, 2, 2)',
        'ELECTRONIC_TEMPERATURE = 1500.0',
        "HARMONIC_DYN = os.path.join(HERE, 'ph/x.dyn')",
        'HARMONIC_QGRID = (2, 2, 3)',
        "PP_DIR = '/pseudo'",
        "LAUNCHER = 'local'",
        'pao.sscha_setup(',
        'pao.sscha_status()',
        "pao.sscha_force_constants('hessian'",
    ):
        assert expected in text, expected
    assert '__' not in text.split('if __name__')[0].replace('__file__', '').replace('__init__', '')


def test_collect_sscha_prompts_local_and_slurm(monkeypatch, tmp_path):
    args = types.SimpleNamespace(savedir=None, prefix=None)
    answers = iter(['tan.save', 'tan', 'HERE', '1000, 2000', '2', 'qe/tan.dyn', '6', '6 6 6', 'ionic',
                    '100, 200', '400', 'local', 'pw.x -nk 4', 'mpirun -np 8'])  # fmt: skip
    monkeypatch.setattr('builtins.input', lambda *_: next(answers))
    cfg = sd.collect_sscha(args, str(tmp_path))
    assert cfg['temperatures'] == [1000.0, 2000.0] and cfg['harmonic_qgrid'] == [6, 6, 6]
    assert cfg['kgrid'] == [6, 6, 6] and cfg['electronic_temperature'] == 'ionic'
    assert cfg['population_sizes'] == [100, 200] and cfg['qe_ph'] == 'ph.x -nk 4'
    answers = iter(['tan.save', 'tan', '/pp', '300', '3', 'h/tan.dyn', '3', '', 'fixed', '500', '50', '100',
                    'slurm', 'pw.x', 'srun', 'genoa', '1', '64', '01:00:00', 'proj', '8', '00:30:00', '10',
                    'source ~/env.sh', 'python3'])  # fmt: skip
    monkeypatch.setattr('builtins.input', lambda *_: next(answers))
    cfg = sd.collect_sscha(args, str(tmp_path))
    assert cfg['kgrid'] is None and cfg['electronic_temperature'] == 500.0
    assert cfg['slurm_directives'] == [
        '#SBATCH -p genoa',
        '#SBATCH -N 1',
        '#SBATCH -n 64',
        '#SBATCH -t 01:00:00',
        '#SBATCH -A proj',
    ]
    assert cfg['slurm_step_directives'][2] == '#SBATCH -n 8' and cfg['env_setup'] == [
        'source ~/env.sh'
    ]
    compile(sd.build_sscha_script(cfg), 'main.sscha.py', 'exec')


@pytest.fixture()
def driver(tmp_path, monkeypatch):
    """The generated driver as a module, on the harmonic model, working in ``tmp_path``."""
    pytest.importorskip('sscha')
    from PAOFLOW import PAOFLOW
    from PAOFLOW.phonon.sscha_bridge import require_sscha

    truth = require_sscha()[0].Phonons(SSCHA, 3)
    supercell = truth.GenerateSupercellDyn(truth.GetSupercell())
    np.save(tmp_path / 'phi.npy', np.real(supercell.dynmats[0]))
    np.save(tmp_path / 'ref.npy', np.asarray(supercell.structure.coords) / BOHR_ANGSTROM)
    (tmp_path / 'fake_pw.py').write_text(FAKE_PW)

    path = tmp_path / 'main.sscha.py'
    path.write_text(sd.build_sscha_script(_cfg()))
    spec = importlib.util.spec_from_file_location('main_sscha', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def model_paoflow(**_):
        pao = PAOFLOW.PAOFLOW(
            model={'label': 'graphene', 't': 2.7, 'delta': 1.0},
            outputdir=str(tmp_path / 'model'),
            verbose=False,
        )
        arry, attr = pao.data_controller.data_dicts()
        arry['species'] = [('N', 'N.upf'), ('Ta', 'Ta.upf')]
        attr.update(insulator=False, qe_smearing='mp', qe_degauss=0.02, ecutwfc=84.0, nspin=1)
        return pao

    module.PAOFLOW = types.SimpleNamespace(PAOFLOW=model_paoflow)
    module.QE_PW = '%s %s %s %s' % (
        sys.executable,
        tmp_path / 'fake_pw.py',
        tmp_path / 'phi.npy',
        tmp_path / 'ref.npy',
    )
    np.random.seed(3)
    return module


def _run_fake_pw(module, inputs):
    """What a scheduler would do with the inputs."""
    module.run_local(inputs)


def test_driver_local_launcher_runs_to_the_force_constants(driver, capsys):
    driver.run([2800.0])
    status = driver._paoflow(2800.0).sscha_status()
    assert status['stage'] == 'done' and status['stable'] and status['converged']
    assert (
        status['exported'] == ['auxiliary', 'hessian']
        and status['electronic_temperature'] == 2800.0
    )
    folder = driver.sscha_dir(2800.0)
    assert os.path.isfile(os.path.join(folder, 'force_constants_hessian', 'tan.dyn0'))
    assert os.path.isfile(os.path.join(folder, 'tan_sscha_auxiliary.fc'))
    assert not [d for d in os.listdir(os.path.join(folder, 'pop1')) if d.startswith('tmp_')]
    assert "smearing = 'fd'" in open(os.path.join(folder, 'pop1', 'cfg_0000.in')).read()
    out = capsys.readouterr().out
    assert 'free-energy Hessian: stable' in out and 'next: done' in out
    driver.run([2800.0])  # nothing left to do
    assert 'next: done' in capsys.readouterr().out


def test_driver_slurm_launcher_chains_arrays_and_steps(driver, tmp_path, monkeypatch, capsys):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text(FAKE_SBATCH)
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', '%s:%s' % (bin_dir, os.environ['PATH']))
    monkeypatch.setenv('SBATCH_LOG', str(tmp_path / 'sbatch.log'))
    driver.LAUNCHER, driver.MPI_QE = 'slurm', 'srun'
    driver.SLURM_DIRECTIVES, driver.SLURM_STEP_DIRECTIVES = ['#SBATCH -n 64'], ['#SBATCH -n 4']
    driver.ENV_SETUP = ['source ~/env.sh']

    driver.run([2800.0])  # setup, population 1, its array and the step job
    calls = open(tmp_path / 'sbatch.log').read().splitlines()
    assert len(calls) == 2 and '--array=0-19%40' in calls[0] and 'pw_array.sh' in calls[0]
    assert '--dependency=afterany:4242' in calls[1] and calls[1].endswith('step.sh')
    pop1 = os.path.join(driver.sscha_dir(2800.0), 'pop1')
    array = open(os.path.join(pop1, 'pw_array.sh')).read()
    assert (
        '#SBATCH -n 64' in array
        and 'source ~/env.sh' in array
        and 'srun' in array
        and 'rm -rf tmp_*_$idx' in array
    )
    assert 'step --temperature 2800 --chain' in open(os.path.join(pop1, 'step.sh')).read()

    driver.MPI_QE = ''
    _run_fake_pw(
        driver, sorted(os.path.join(pop1, f) for f in os.listdir(pop1) if f.endswith('.in'))[:-1]
    )
    driver.step(
        [2800.0], chain=True
    )  # the step job: 19 of 20 finished -> minimise, Hessian population, submit
    calls = open(tmp_path / 'sbatch.log').read().splitlines()
    assert len(calls) == 4 and 'pw_array.sh' in calls[2] and '--array=0-19%40' in calls[2]
    status = driver._paoflow(2800.0).sscha_status()
    assert status['stage'] == 'forces' and status['population'] == 2 and status['converged']

    driver.step(
        [2800.0], chain=True
    )  # the array came back empty: stop instead of resubmitting forever
    assert len(open(tmp_path / 'sbatch.log').read().splitlines()) == 4
    assert 'call `run` to resubmit' in capsys.readouterr().out
    driver.run([2800.0])  # resubmits the missing configurations
    assert len(open(tmp_path / 'sbatch.log').read().splitlines()) == 6


def test_driver_manual_launcher_lists_the_inputs(driver, capsys):
    driver.LAUNCHER = 'manual'
    driver.run([2800.0])
    out = capsys.readouterr().out
    assert 'run pw.x on these 20 inputs' in out and 'output/sscha_T2800/pop1/cfg_0019.in' in out
    for population in (1, 2):
        folder = os.path.join(driver.sscha_dir(2800.0), 'pop%d' % population)
        _run_fake_pw(
            driver, sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.in'))
        )
        driver.run([2800.0])
    assert driver._paoflow(2800.0).sscha_status()['stage'] == 'done'


def test_array_spec_compresses_ranges(driver):
    assert driver._array_spec([0, 1, 2, 3]) == '0-3'
    assert driver._array_spec([7, 3, 8, 9, 12]) == '3,7-9,12'
