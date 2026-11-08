"""run_main: an error on one MPI rank ends the job instead of leaving the other ranks waiting."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from PAOFLOW.utils.mpi_guard import run_main

pytestmark = pytest.mark.unit


def test_single_process_behaves_like_main():
    assert run_main(lambda: 42) == 42
    with pytest.raises(ValueError, match='boom'):
        run_main(lambda: (_ for _ in ()).throw(ValueError('boom')))
    for code in (None, 0, 3, 'message'):
        with pytest.raises(SystemExit) as raised:
            run_main(lambda code=code: sys.exit(code))
        assert raised.value.code == code


SCRIPT = textwrap.dedent(
    """
    import sys
    from mpi4py import MPI
    from PAOFLOW.utils.mpi_guard import run_main

    def main():
        world = MPI.COMM_WORLD
        if world.Get_rank() == 1:
            if sys.argv[1] == 'raise':
                raise ValueError('only rank 1 fails')
            sys.exit('rank 1 gives up')
        world.Barrier()  # rank 0 waits here for ever without the guard

    run_main(main)
    """
)


@pytest.mark.parametrize(
    'how, expected',
    [('raise', 'ValueError: only rank 1 fails'), ('exit', 'stopped: rank 1 gives up')],
)
def test_error_on_one_rank_aborts_the_job(tmp_path, how, expected):
    launcher = shutil.which('mpirun') or shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('no mpirun')
    script = tmp_path / 'driver.py'
    script.write_text(SCRIPT)
    env = dict(os.environ, OMP_NUM_THREADS='1')
    started = time.time()
    try:
        result = subprocess.run(
            [launcher, '-np', '2', sys.executable, str(script), how],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
            cwd=tmp_path,
        )
    except subprocess.TimeoutExpired:
        pytest.fail('the job hung: rank 0 still waits at the barrier')
    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert expected in output and 'aborting every MPI rank' in output
    assert '[rank 1 of 2]' in output
    assert time.time() - started < 90
