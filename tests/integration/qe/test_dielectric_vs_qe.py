"""Phase 1 harness for the dielectric-tensor benchmark against QE epsilon.x.

This file is a **placeholder** that wires up the test infrastructure
(metric extraction, environment-variable discovery, baseline JSON file)
without yet committing to a specific set of benchmark inputs in the test
asset tarball.  See ``TODOs/nonlocal_velocity_correction.md`` for the
broader plan.

How to enable the actual benchmark
----------------------------------
Set ``PAOFLOW_DIELECTRIC_BENCHMARK_DIR`` to a directory that contains
one subdirectory per material, each holding:

* ``main.py``               — standalone PAOFLOW driver script that
  writes ``output/{epsi,epsr,eels}_{xx,yy,zz}.dat``.
* The QE save dir, UPF, wavefunctions, etc. used by ``main.py`` (paths
  inside ``main.py`` are relative to the subdirectory).
* QE reference files ``epsi_<prefix>.dat``, ``epsr_<prefix>.dat``,
  ``eels_<prefix>.dat`` produced by ``epsilon.x`` on the same grid.

Each subdirectory's name is matched against the keys of ``_BENCHMARKS``
below to pick metric windows and acceptance tolerances.

Phase 1 deliverable
-------------------
* Metric extractor (``dielectric_metrics.py``) — done.
* Metric-extractor unit tests
  (``tests/unit/integration/test_dielectric_metrics.py``) — done.
* This placeholder + baseline JSON — done.

Phase 4 deliverable
-------------------
* Promote ``PAOFLOW_DIELECTRIC_BENCHMARK_DIR`` from opt-in env var to
  a regular asset tarball (mirroring the existing QE asset machinery).
* Tighten the tolerances in ``_BENCHMARKS`` once the non-local
  pseudopotential velocity correction is in place.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from .dielectric_metrics import (
    metrics_from_paoflow_output,
    metrics_from_qe_outputs,
)

HERE = Path(__file__).resolve().parent
BASELINE_PATH = HERE / 'dielectric_baseline.json'

ENV_VAR = 'PAOFLOW_DIELECTRIC_BENCHMARK_DIR'


@dataclass(frozen=True)
class _Benchmark:
    subdir: str  # directory name inside $PAOFLOW_DIELECTRIC_BENCHMARK_DIR
    qe_prefix: str  # e.g. 'aluminum' → epsi_aluminum.dat
    eps2_peak_window: tuple[float, float]
    eels_peak_window: tuple[float, float]


# Three benchmarks span the relevant physics regimes for the non-local
# pseudopotential velocity correction:
#   * Si  — insulator, sp electrons only           → correction should be small
#   * Al  — simple metal, sp electrons only        → Drude piece unchanged
#   * Cu  — d-electron metal                       → 2x underprediction today
_BENCHMARKS: list[_Benchmark] = [
    _Benchmark(
        subdir='example15',
        qe_prefix='silicon',
        eps2_peak_window=(3.0, 4.0),
        eels_peak_window=(3.0, 4.0),
    ),
    _Benchmark(
        subdir='example17_Al_epsilon',
        qe_prefix='aluminum',
        eps2_peak_window=(0.5, 3.0),
        eels_peak_window=(6.0, 9.5),
    ),
    _Benchmark(
        subdir='example17_Cu_epsilon',
        qe_prefix='copper',
        eps2_peak_window=(1.5, 4.5),
        eels_peak_window=(5.0, 9.5),
    ),
]


def _resolve_benchmark_root() -> Optional[Path]:
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        return None
    return path


def _load_baseline() -> dict:
    if not BASELINE_PATH.exists():
        return {}
    return json.loads(BASELINE_PATH.read_text())


def _run_main_py(bench_dir: Path) -> None:
    """Run ``main.py`` inside ``bench_dir`` in-place.

    We deliberately do *not* sandbox-copy: the example ``main.py`` scripts
    use relative paths like ``../../../BASIS/`` that would break under a
    copytree.  The only side effect is a refreshed ``output/`` directory
    inside the benchmark dir, which the example owns anyway.
    """
    env = os.environ.copy()
    env.setdefault('MPLBACKEND', 'Agg')
    subprocess.run([sys.executable, 'main.py'], cwd=bench_dir, check=True, env=env)


@pytest.mark.integration
@pytest.mark.parametrize('bench', _BENCHMARKS, ids=lambda b: b.subdir)
def test_dielectric_vs_qe(bench: _Benchmark) -> None:
    """Run the PAOFLOW dielectric calculation and compare metrics to QE.

    Skipped by default — set ``PAOFLOW_DIELECTRIC_BENCHMARK_DIR`` to an
    appropriately-laid-out directory to enable.  See module docstring.
    """
    bench_root = _resolve_benchmark_root()
    if bench_root is None:
        pytest.skip(
            f'Set {ENV_VAR}=<path-to-benchmark-root> to enable the '
            'PAOFLOW-vs-QE dielectric benchmarks.  See module docstring.'
        )

    bench_dir = bench_root / bench.subdir
    if not bench_dir.is_dir():
        pytest.skip(f'Benchmark subdirectory missing: {bench_dir}')

    main_py = bench_dir / 'main.py'
    if not main_py.exists():
        pytest.skip(f'main.py missing from {bench_dir}')

    qe_files = {
        'epsr': bench_dir / f'epsr_{bench.qe_prefix}.dat',
        'epsi': bench_dir / f'epsi_{bench.qe_prefix}.dat',
        'eels': bench_dir / f'eels_{bench.qe_prefix}.dat',
    }
    missing = [str(p) for p in qe_files.values() if not p.exists()]
    if missing:
        pytest.skip(f'QE reference file(s) missing for {bench.subdir}: ' + ', '.join(missing))

    _run_main_py(bench_dir)
    outdir = bench_dir / 'output'
    assert outdir.is_dir(), f'PAOFLOW did not produce {outdir}'

    pao_metrics = metrics_from_paoflow_output(
        outdir,
        eps2_peak_window=bench.eps2_peak_window,
        eels_peak_window=bench.eels_peak_window,
    )

    qe_metrics = metrics_from_qe_outputs(
        epsr_path=qe_files['epsr'],
        epsi_path=qe_files['epsi'],
        eels_path=qe_files['eels'],
        eps2_peak_window=bench.eps2_peak_window,
        eels_peak_window=bench.eels_peak_window,
    )

    baseline_all = _load_baseline()
    record = {
        'paoflow': pao_metrics.asdict(),
        'qe': qe_metrics.asdict(),
    }
    print(f'\n[dielectric-vs-QE] {bench.subdir} measured: {json.dumps(record, indent=2)}')

    baseline = baseline_all.get(bench.subdir)
    if baseline is None:
        pytest.skip(
            f'No baseline recorded for {bench.subdir} in {BASELINE_PATH.name}; '
            f'measured values were printed above.  Add them to the baseline JSON '
            f'to enable regression checking.'
        )

    # Phase-1 contract: PAOFLOW metrics must not drift vs the recorded
    # baseline.  Tolerances are intentionally loose (they pin "the code
    # produces approximately what it produced before") and are tightened
    # in Phase 4 once the non-local correction lands.
    tol_rel = baseline.get('paoflow_rel_tol', 0.05)
    tol_abs = baseline.get('paoflow_abs_tol', 0.1)
    for key, expected in baseline['paoflow'].items():
        measured = pao_metrics.asdict()[key]
        assert measured == pytest.approx(expected, rel=tol_rel, abs=tol_abs), (
            f'{bench.subdir}.{key}: measured={measured} vs baseline={expected} '
            f'(rel_tol={tol_rel}, abs_tol={tol_abs})'
        )
