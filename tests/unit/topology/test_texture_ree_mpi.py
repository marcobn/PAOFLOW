"""Regression checks for local textures consumed by MPI REE calculations."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.unit
_LAUNCHER = shutil.which('mpiexec') or shutil.which('mpirun')
_DRIVER = Path(__file__).with_name('texture_ree_mpi_driver.py')
_SOURCE = Path(__file__).resolve().parents[3] / 'src'


def _run_driver(ranks: int, destination: Path, environment: dict[str, str]) -> None:
    destination.mkdir()
    result = subprocess.run(
        [_LAUNCHER, '-n', str(ranks), sys.executable, str(_DRIVER), str(destination)],
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f'Texture/REE driver failed with {ranks} MPI ranks.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )


@pytest.mark.skipif(_LAUNCHER is None, reason='MPI launcher is unavailable')
def test_texture_ree_agrees_with_uneven_mpi_partition(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(
        filter(None, [str(_SOURCE), environment.get('PYTHONPATH')])
    )
    environment['OMP_NUM_THREADS'] = environment['OPENBLAS_NUM_THREADS'] = '1'
    version = subprocess.run(
        [_LAUNCHER, '--version'],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if (
        'Open MPI' in version.stdout + version.stderr
        or 'OpenRTE' in version.stdout + version.stderr
    ):
        environment['OMPI_MCA_rmaps_base_oversubscribe'] = '1'
        if getattr(os, 'geteuid', lambda: -1)() == 0:
            environment['OMPI_ALLOW_RUN_AS_ROOT'] = '1'
            environment['OMPI_ALLOW_RUN_AS_ROOT_CONFIRM'] = '1'
    serial, parallel = tmp_path / 'one-rank', tmp_path / 'two-ranks'
    _run_driver(1, serial, environment)
    _run_driver(2, parallel, environment)
    assert json.loads((serial / 'counts.json').read_text()) == [5]
    assert json.loads((parallel / 'counts.json').read_text()) == [2, 3]
    tables = sorted(path.name for path in serial.glob('*.dat'))
    archives = sorted(path.name for path in serial.glob('*.npz'))
    assert len(tables) == 22 and len(archives) == 4
    assert tables == sorted(path.name for path in parallel.glob('*.dat'))
    assert archives == sorted(path.name for path in parallel.glob('*.npz'))
    for name in tables:
        np.testing.assert_allclose(
            np.loadtxt(parallel / name),
            np.loadtxt(serial / name),
            rtol=1e-10,
            atol=1e-18,
            err_msg=name,
        )
    for name in archives:
        with (
            np.load(serial / name, allow_pickle=False) as expected,
            np.load(parallel / name, allow_pickle=False) as actual,
        ):
            assert actual.files == expected.files
            for key in expected.files:
                np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-15)
