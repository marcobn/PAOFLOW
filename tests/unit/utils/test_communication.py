"""Collective helpers of utils.communication: rows and rounds, slices, the scatter_full layout.

The MPI test shrinks the limits (64-byte rounds, 7-element slices) so that the round and slice
paths run with small arrays, and checks every result bit for bit against NumPy references; it also
compares the position dipoles computed directly in the scatter_full layout with the former route
(each rank a contiguous slice, gathered on rank 0 and scattered again), to round-off.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from PAOFLOW.utils import communication as cm

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SI_ONCV_DIR = REPO_ROOT / 'examples' / 'qe_examples' / 'example16' / 'Si_ONCV'


def test_steps_cover_the_buffer_in_pieces():
    assert cm._steps(0, 5) == [(0, 0)]
    assert cm._steps(12, 5) == [(0, 5), (5, 10), (10, 12)]
    assert cm._steps(10, 5) == [(0, 5), (5, 10)]


def test_row_rounds_respect_the_byte_cap(monkeypatch):
    monkeypatch.setattr(cm, 'MAX_MESSAGE_BYTES', 1000)
    counts = np.array([7, 6, 0])
    per_round, nrounds = cm._row_rounds(counts, row_bytes=48, nranks=3)
    assert per_round * 48 * 3 <= 1000 and per_round >= 1
    assert nrounds * per_round >= 7 and (nrounds - 1) * per_round < 7
    assert cm._row_rounds(np.zeros(3, dtype=int), 48, 3)[1] == 0


@pytest.mark.parametrize('nsize', [0, 1, 5, 37, 100])
@pytest.mark.parametrize('npool', [1, 2, 3])
@pytest.mark.parametrize('nranks', [1, 2, 3, 7])
def test_scatter_full_indices_partition_the_axis(nsize, npool, nranks):
    parts = [cm.scatter_full_indices(nsize, npool, nranks, r) for r in range(nranks)]
    joined = np.concatenate(parts)
    assert sorted(joined.tolist()) == list(range(nsize))
    for r, part in enumerate(parts):  # as many rows as scatter_full allocates on rank r
        ts, te = cm.load_balancing(nranks, r, nsize % nranks)
        assert len(part) == nsize // nranks + (te - ts)


def test_single_rank_helpers_with_tiny_limits(monkeypatch):
    monkeypatch.setattr(cm, 'MAX_MESSAGE_BYTES', 64)
    monkeypatch.setattr(cm, 'MAX_MPI_COUNT', 7)
    rng = np.random.default_rng(1)
    arr = rng.standard_normal((5, 3, 2)) + 1j * rng.standard_normal((5, 3, 2))
    local = cm.scatter_array(arr)
    assert np.array_equal(local, arr)
    back = np.zeros_like(arr)
    cm.gather_array(back, local)
    assert np.array_equal(back, arr)
    ints = np.arange(23, dtype=np.int64).reshape(23, 1)
    assert np.array_equal(cm.reduce_full(ints), ints)
    fortran = np.asfortranarray(np.arange(30.0).reshape(5, 6))
    assert np.array_equal(cm.allreduce_inplace(fortran.copy(order='F')), fortran)
    assert np.array_equal(cm.bcast_array(fortran.copy(order='F')), fortran)
    with pytest.raises(ValueError):
        cm.bcast_array(np.arange(20.0).reshape(4, 5)[:, ::2])


MPI_SCRIPT = textwrap.dedent(
    """
    import sys
    import numpy as np
    from mpi4py import MPI
    from PAOFLOW.utils import communication as cm
    from PAOFLOW.utils.mpi_guard import run_main

    SI_DIR = sys.argv[1]


    def check(name, ok):
        if not ok:
            raise AssertionError('%s differs on rank %d' % (name, cm.rank))


    def main():
        rank, size = cm.rank, cm.size
        rng = np.random.default_rng(7)
        arr = rng.standard_normal((37, 3, 4)) + 1j * rng.standard_normal((37, 3, 4))

        cm.MAX_MESSAGE_BYTES = 64  # many rounds: one row (12 complex) is 192 bytes
        local = cm.scatter_array(arr if rank == 0 else None)
        sizes = cm.load_sizes(size, arr.shape[0], 1)
        check('scatter_array', np.array_equal(local, arr[sizes[rank, 1]:sizes[rank, 1] + sizes[rank, 2]]))
        full = np.zeros_like(arr) if rank == 0 else None
        cm.gather_array(full, local)
        if rank == 0:
            check('gather_array', np.array_equal(full, arr))

        for npool in (1, 2):
            mine = cm.scatter_full(arr if rank == 0 else None, npool)
            check('scatter_full', np.array_equal(mine, arr[cm.scatter_full_indices(37, npool)]))
            back = cm.gather_full(mine, npool)
            if rank == 0:
                check('gather_full', np.array_equal(back, arr))

        small = arr[:, :2, :2].copy()  # 4 elements per row
        cm.MAX_MPI_COUNT = 10  # staged rounds: more rows (37) than the count limit
        local = cm.scatter_array(small if rank == 0 else None)
        check('scatter_array staged', np.array_equal(local, small[sizes[rank, 1]:sizes[rank, 1] + sizes[rank, 2]]))
        full = np.zeros_like(small) if rank == 0 else None
        cm.gather_array(full, local)
        if rank == 0:
            check('gather_array staged', np.array_equal(full, small))
        cm.MAX_MPI_COUNT = 2**31 - 1

        scalars = np.zeros(size, dtype=np.int64) if rank == 0 else None
        cm.gather_array(scalars, np.array(rank + 10, dtype=np.int64))
        if rank == 0:
            check('gather_array 0-d', np.array_equal(scalars, np.arange(size) + 10))

        cm.MAX_MPI_COUNT = 7  # sliced reductions and broadcasts
        ints = (np.arange(23, dtype=np.int64) + 100 * rank).reshape(23, 1)
        total = cm.reduce_full(ints)
        expected = sum((np.arange(23, dtype=np.int64) + 100 * r).reshape(23, 1) for r in range(size))
        if rank == 0:
            check('reduce_full', np.array_equal(total, expected))
        fortran = np.asfortranarray((np.arange(30, dtype=np.int64) * (rank + 1)).reshape(5, 6))
        cm.allreduce_inplace(fortran)
        check('allreduce_inplace', np.array_equal(fortran, np.arange(30).reshape(5, 6) * sum(range(1, size + 1))))
        data = np.arange(41.0) if rank == 0 else np.zeros(41)
        cm.bcast_array(data)
        check('bcast_array', np.array_equal(data, np.arange(41.0)))
        cm.MAX_MPI_COUNT = 2**31 - 1
        cm.MAX_MESSAGE_BYTES = 2**30

        if SI_DIR != 'none':  # position dipoles: direct scatter_full layout = former gather + scatter
            from PAOFLOW.hamiltonian.position_dipoles import position_dipoles_on_grid

            alat = 10.2
            arrays = {
                'species': [('Si', 'Si_ONCV_PBE_sr.UPF')], 'atoms': ['Si', 'Si'],
                'tau': np.array([[0.0, 0.0, 0.0], [0.25 * alat] * 3]),
                'a_vectors': 0.5 * np.array([[-1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [-1.0, 1.0, 0.0]]),
            }
            attributes = {'fpath': SI_DIR, 'alat': alat, 'dftSO': False}

            class DC:
                def data_dicts(self):
                    return arrays, attributes

            kgrid = np.random.default_rng(3).uniform(-0.5, 0.5, size=(3, 11))
            build = lambda kg: position_dipoles_on_grid(DC(), kg, pao_tol=1e-2, n_q=300)
            npool = 2
            new = np.ascontiguousarray(build(kgrid[:, cm.scatter_full_indices(11, npool)]))
            ks, ke = cm.load_balancing(size, rank, 11)
            part = np.ascontiguousarray(build(kgrid[:, ks:ke]))
            full = np.zeros((11,) + part.shape[1:], dtype=part.dtype) if rank == 0 else None
            cm.gather_array(full, part)
            old = cm.scatter_full(np.ascontiguousarray(full) if rank == 0 else None, npool)
            # equal to round-off, not bit for bit: the vectorised dipole arithmetic depends on how
            # many k-points are computed together (one k alone differs from the batch by ~1e-14),
            # so the former route also varied at this level with the number of ranks
            check('position dipoles', np.allclose(new, old, rtol=0.0, atol=1e-12))

        print('OK rank %d of %d' % (rank, size), flush=True)


    run_main(main)
    """
)


@pytest.mark.parametrize('nranks', [2, 3])
def test_helpers_under_mpi_with_tiny_limits(tmp_path, nranks):
    launcher = shutil.which('mpirun') or shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('no mpirun')
    script = tmp_path / 'driver.py'
    script.write_text(MPI_SCRIPT)
    si_dir = str(SI_ONCV_DIR) if (SI_ONCV_DIR / 'Si_ONCV_PBE_sr.UPF').exists() else 'none'
    env = dict(os.environ, OMP_NUM_THREADS='1')
    try:
        result = subprocess.run(
            [launcher, '-np', str(nranks), sys.executable, str(script), si_dir],
            capture_output=True,
            text=True,
            check=False,
            timeout=600,
            env=env,
            cwd=tmp_path,
        )
    except subprocess.TimeoutExpired:
        pytest.fail('the MPI job hung')
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-3000:]
    assert all(f'OK rank {r} of {nranks}' in output for r in range(nranks)), output[-3000:]


LARGE_SCRIPT = textwrap.dedent(
    """
    import numpy as np
    from PAOFLOW.utils import communication as cm
    from PAOFLOW.utils.mpi_guard import run_main


    def main():
        rows = 2_200_000_000 // cm.size  # 2.2e9 int8 elements in total: beyond a C int
        local = np.full((rows, 1), cm.rank + 1, dtype=np.int8)
        full = np.zeros((rows * cm.size, 1), dtype=np.int8) if cm.rank == 0 else None
        cm.gather_array(full, local)
        if cm.rank == 0:
            assert full.size > 2**31 - 1
            assert all(int(full[r * rows, 0]) == r + 1 and int(full[(r + 1) * rows - 1, 0]) == r + 1
                       for r in range(cm.size))
        back = cm.scatter_array(full if cm.rank == 0 else None)
        assert back.shape == (rows, 1) and int(back[0, 0]) == cm.rank + 1 and int(back[-1, 0]) == cm.rank + 1
        print('OK large rank %d' % cm.rank, flush=True)


    run_main(main)
    """
)


@pytest.mark.slow
def test_gather_beyond_the_int_count_under_mpi(tmp_path):
    """2.2e9 int8 elements (about 4.4 GB in total): fails with MPI_ERR_ARG on Open MPI 5.0 before."""
    if os.environ.get('PAOFLOW_LARGE_MPI_TEST') != '1':
        pytest.skip('set PAOFLOW_LARGE_MPI_TEST=1 (needs about 5 GB of memory)')
    launcher = shutil.which('mpirun') or shutil.which('mpiexec')
    if launcher is None:
        pytest.skip('no mpirun')
    script = tmp_path / 'large.py'
    script.write_text(LARGE_SCRIPT)
    result = subprocess.run(
        [launcher, '-np', '2', sys.executable, str(script)],
        capture_output=True, text=True, check=False, timeout=1200, cwd=tmp_path,
        env=dict(os.environ, OMP_NUM_THREADS='1'),
    )  # fmt: skip
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-3000:]
    assert 'OK large rank 0' in output and 'OK large rank 1' in output
