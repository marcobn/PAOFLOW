"""Per-stage timing and memory record of one benchmark case.

The record is a JSON file rewritten after every stage, so a job killed by
the scheduler (out of memory, walltime) still leaves everything up to the
stage that died, with ``status`` stuck at ``'running'``.  ``finalize``
then fills in the scheduler's verdict.

Peak memory per stage comes from ``VmHWM`` in ``/proc/self/status``, reset
before each stage through ``/proc/self/clear_refs`` (Linux >= 4.0).  Where
that reset is not permitted, the lifetime peak ``ru_maxrss`` is recorded
instead and ``peak_rss_scope`` says so.
"""

import json
import os
import platform
import socket
import sys
import time
import traceback
from contextlib import contextmanager

import numpy as np

GB = 1024.0**3


def _proc_status_kb(field):
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith(field + ':'):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def _reset_peak_rss():
    """Reset ``VmHWM`` to the current RSS; False where not permitted."""
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def _maxrss_bytes():
    import resource

    # kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def meminfo_bytes():
    """``MemTotal`` and ``MemAvailable`` of the node, in bytes."""
    out = {}
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                key, value = line.split(':', 1)
                if key in ('MemTotal', 'MemAvailable'):
                    out[key] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return out


def array_bytes(arrays, min_bytes=1 << 20):
    """``{name: nbytes}`` of the ndarrays in a DataController dict that are
    at least ``min_bytes`` on this rank."""
    out = {}
    for name, value in arrays.items():
        if isinstance(value, np.ndarray) and value.nbytes >= min_bytes:
            out[name] = int(value.nbytes)
    return out


def environment(comm):
    """Hardware, threading and software versions of the run."""
    import scipy

    try:
        from PAOFLOW.sparse.engine import _node_local_ranks

        local = _node_local_ranks(comm)
    except Exception:
        local = None
    env = {
        'host': socket.gethostname(),
        'platform': platform.platform(),
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'mpi_ranks': comm.Get_size(),
        'ranks_per_node': local,
        'cpu_count': os.cpu_count(),
        'meminfo': meminfo_bytes(),
        'threads': {
            k: os.environ.get(k)
            for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
        },
        'slurm': {
            k: os.environ.get(k)
            for k in (
                'SLURM_JOB_ID',
                'SLURM_ARRAY_JOB_ID',
                'SLURM_ARRAY_TASK_ID',
                'SLURM_JOB_NODELIST',
                'SLURM_CPUS_PER_TASK',
            )
        },
        'paoflow_commit': _git_commit(),
    }
    try:
        from threadpoolctl import threadpool_info

        env['blas'] = [
            {k: i.get(k) for k in ('internal_api', 'num_threads', 'version')}
            for i in threadpool_info()
        ]
    except Exception:
        env['blas'] = None
    return env


def _git_commit():
    import subprocess

    try:
        import PAOFLOW

        src = os.path.dirname(os.path.abspath(PAOFLOW.__file__))
        out = subprocess.run(
            ['git', '-C', src, 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=10
        )
        dirty = subprocess.run(
            ['git', '-C', src, 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except Exception:
        return None


class Recorder:
    """Collects the record of one case and writes it on rank 0.

    Parameters
    ----------
    path : str
        The ``record.json`` to (re)write.
    case : dict
        The case, stored verbatim.
    comm : mpi4py communicator
    """

    def __init__(self, path, case, comm):
        self.path = path
        self.comm = comm
        self.rank = comm.Get_rank()
        self.data = {
            'case': case,
            'status': 'running',
            'started': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'env': environment(comm),
            'stages': [],
            'modules': [],
            'hamiltonian': {},
            'solver': {},
            'mesh': {},
            'outputs': {},
        }
        self.flush()

    # ------------------------------------------------------------------
    def flush(self):
        """Write the record atomically (rank 0 only)."""
        if self.rank != 0:
            return
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=1, default=_jsonable)
        os.replace(tmp, self.path)

    def set(self, section, **values):
        """Merge ``values`` into one top-level section and flush."""
        self.data.setdefault(section, {}).update(values)
        self.flush()

    def _gather_memory(self, arrays=None):
        reset_ok = getattr(self, '_reset_ok', False)
        peak = _proc_status_kb('VmHWM')
        peak = peak * 1024 if (peak is not None and reset_ok) else _maxrss_bytes()
        rss = _proc_status_kb('VmRSS')
        rss = rss * 1024 if rss is not None else None
        local_arrays = array_bytes(arrays) if arrays is not None else {}
        allv = self.comm.gather((peak, rss, sum(local_arrays.values()), local_arrays), root=0)
        if self.rank != 0:
            return None
        peaks = [a[0] for a in allv]
        rsss = [a[1] for a in allv if a[1] is not None]
        arr_tot = [a[2] for a in allv]
        return {
            'peak_rss_max': max(peaks),
            'peak_rss_sum': sum(peaks),
            'rss_max': max(rsss) if rsss else None,
            'peak_rss_scope': 'stage' if reset_ok else 'lifetime',
            'arrays_max': max(arr_tot),
            'arrays_sum': sum(arr_tot),
            'arrays_rank0': allv[0][3],
        }

    @contextmanager
    def stage(self, name, arrays=None, **extra):
        """Time one stage between barriers and record its peak memory.

        ``arrays`` (a DataController ``data_arrays`` dict) is sized after the
        stage, so the record shows which arrays the stage left behind.
        """
        self.comm.Barrier()
        self._reset_ok = _reset_peak_rss()
        self._reset_ok = all(self.comm.allgather(self._reset_ok))
        entry = {'name': name, 'status': 'running', **extra}
        if self.rank == 0:
            self.data['stages'].append(entry)
            self.flush()
        t0 = time.perf_counter()
        try:
            yield entry
        except BaseException:
            entry['status'] = 'failed'
            entry['seconds'] = time.perf_counter() - t0
            self.flush()
            raise
        self.comm.Barrier()
        entry['seconds'] = time.perf_counter() - t0
        entry['status'] = 'ok'
        mem = self._gather_memory(arrays)
        if self.rank == 0:
            entry.update(mem)
        self.flush()

    def record_module_time(self, name, seconds):
        """One line of PAOFLOW's own ``report_module_time`` output."""
        if self.rank == 0:
            stage = self.data['stages'][-1]['name'] if self.data['stages'] else None
            self.data['modules'].append({'name': name, 'seconds': seconds, 'stage': stage})

    def fail(self, exc):
        """Record a Python exception; every rank writes its own error file,
        so an error raised on one rank only is not lost."""
        info = {
            'rank': self.rank,
            'type': type(exc).__name__,
            'message': str(exc)[:4000],
            'traceback': traceback.format_exc()[-8000:],
        }
        err_path = os.path.join(os.path.dirname(self.path), 'error_rank%d.json' % self.rank)
        with open(err_path, 'w') as f:
            json.dump(info, f, indent=1)
        if self.rank == 0:
            self.data['status'] = 'python_error'
            self.data['error'] = info
            self.data['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
            self.flush()

    def done(self):
        self.data['status'] = 'ok'
        self.data['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        self.flush()


def _jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)
