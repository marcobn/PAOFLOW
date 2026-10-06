# Sparse-scaling benchmark

This benchmark grows bulk Si by cell doubling and runs `dos` and Boltzmann `transport` through
each PAOFLOW pipeline until that pipeline fails. Every run records time, memory and sparsity, and
a notebook plots how they scale. It tells us:

- the largest system each pipeline handles, and what stops it (memory, time, or a refusal);
- where the pipelines cross over;
- measured values for the solver policy constants `dense_n_max` and `dense_ratio`, which are
  defaults today and not measurements;
- the size at which the KPM/Sternheimer formulations planned in `handoff.md` become necessary.

## Why doubled Si

`doubling_Hamiltonian` is the one way both pipelines grow a system. The sparse engine builds its
bond list from the base-cell H(R), and from then on it grows only by doubling, without ever
densifying. A doubled cell is still the same crystal, so the per-cell DoS and conductivity must
match the primitive cell. That gives an exact reference at every size, including sizes where no
dense run fits.

The cell size doubles with each step. The base cell of example01 has `nawf = 18`, so `d = 8`
gives `nawf = 4608` and `d = 11` gives `nawf = 36864`. The k-mesh is halved along every doubled
axis, which keeps the physical k-density, and therefore the per-cell result, fixed.

## Pipelines

| name | what runs | expected limit |
|---|---|---|
| `dense` | the dense pipeline: `pao_eigh`, `gradient_and_momenta`, `adaptive_smearing`, `dos`, `transport` | memory of H(R) and H(k), O(nawf² nk) |
| `sparse` | sparse, from the bottom of the spectrum (`energy_window`), with `dense_n_max` lifted | per-k LAPACK scratch, O(nawf²) per k, or time |
| `sparse_interior` | sparse, `interior_window` around E_F, `bond_order=36` | ARPACK shift-invert cost and fill-in |
| `sparse_interior_bo8` | the same with `bond_order=8` | the truly sparse regime (see below) |
| `solver_sweep` | LAPACK against ARPACK on single H(k), for nev/n from 1% to 50% | calibrates `dense_ratio` and the per-k memory |

The dense pipeline can only interpolate *up* from the 12³ H(R) grid of the QE run. At larger
`d` its mesh is clamped to that grid, so its k-point count is higher than in the sparse runs.
The record flags these runs (`mesh.clamped`), and the per-k comparison in the notebook accounts
for it.

Si PAO hoppings decay slowly. With `bond_order=36` (rcut 32 Bohr), one row of H(k) holds about
8500 nonzeros, so H(k) is effectively dense until `nawf` is around 10⁴. With `bond_order=8` it
holds about 890. The two interior ladders separate the cost of the cutoff from the cost of the
algorithm.

## Running it

1. Edit [config.py](config.py), the only file that needs changing:
   - the Slurm account, partition and `setup` lines (module loads, virtualenv);
   - the resources per pipeline;
   - the maximum doubling per pipeline.

   Nothing is passed on the command line.
2. Submit:

   ```bash
   cd benchmarks/sparse_scaling
   python submit.py
   ```

   This writes `runs/cases.json`, one job array per pipeline (one task per size), and a
   finalize job that waits for all of them (`afterany`). Every size is submitted at once,
   because a failing size is a data point, not a reason to stop. Set
   `CONFIG['slurm']['dry_run'] = True` to write the scripts without submitting them.
3. When the jobs finish, the finalize job runs `finalize.py`, which adds the sacct state and
   MaxRSS to every record and marks jobs killed for OOM or timeout. It then runs `collect.py`,
   which writes `runs/results.csv` and `runs/sweep.csv`. You can also run both by hand.
4. Open [analysis.ipynb](analysis.ipynb).

### Resources

- **Ranks per node.** The sparse bond list is replicated on every rank, so more ranks per node
  means *less* memory per rank. Prefer few ranks with many threads: the threads go to BLAS and
  LAPACK through `OMP_NUM_THREADS` and `MKL_NUM_THREADS`, which the job scripts set from
  `--cpus-per-task`.
- **Pools.** `npool` should equal `ntasks`, which gives one pool per rank.
- **Node memory.** Use `--mem=0` (the whole node) with `--exclusive`, so a job fails on the
  benchmark's own memory and not on a request that is too small.
- **Walltime.** Each pipeline's `time` is a cap, and a case that hits it becomes `timeout` in
  the results. Calibrate it from a first pass.

## Files

| file | role |
|---|---|
| `config.py` | every setting |
| `ladder.py` | `build_cases(config)`: the sizes, doubling patterns and meshes |
| `run_case.py` | `run_case(case, outdir)`, which runs one case. Its `__main__` is the Slurm array task (it reads `SLURM_ARRAY_TASK_ID`) |
| `recorder.py` | per-stage timing between MPI barriers, per-stage peak RSS (`VmHWM`, reset each stage), array sizes and environment; rewrites `record.json` after every stage, so a killed job still leaves a partial record |
| `submit.py` | writes `cases.json` and the job scripts, then calls `sbatch` |
| `finalize.py` | adds the sacct state and MaxRSS to each record |
| `collect.py` | builds `results.csv` and `sweep.csv`, including the per-cell accuracy against dense d=0 and against the pipeline's own d=0 |
| `analysis.ipynb` | the figures and the summary tables |

Each case directory holds `record.json`, the PAOFLOW `output/` (DoS, conductivity, Seebeck, and
`sparse.log` for sparse runs), and `error_rank<r>.json` if a rank raised an exception.

## Known issue: velocities on doubled cells

As of 2026-10-06, `doubling_Hamiltonian` zeroes the intra-cell offset `Dnm` for orbital pairs in
different replicas. This affects both the dense `doubling_attr_arry` and the sparse `double_axis`,
which copies it. On the base cell, `Dnm` has commutator form and drops out of the band
velocities. After doubling it no longer does, and the velocities are wrong by about 35%.

On an equivalent doubled cell, σ_xx changes by about 50% and becomes anisotropic, and the
adaptive DoS changes by 1.5%. Until this is fixed, the d > 0 accuracy columns measure this bug.
The timing and memory columns are unaffected.

## Not covered yet

- **On-site disorder.** `CONFIG['disorder_meV']` adds random on-site energies to the sparse bond
  list after doubling. They break the exact folding degeneracies, which are ARPACK's worst case.
  The price is the exact per-cell reference, so it is off by default.
- **Twisted bilayer graphene and other EDTB systems.** These go in through
  `PAOFLOW(model=...)`, which builds the full-cell dense H(R), so they cannot start above the
  dense limit. They need a direct EDTB bond list → `SparseHamiltonian` bridge first.
