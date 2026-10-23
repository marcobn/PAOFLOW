# Sparse-scaling benchmark

This benchmark grows bulk Si by cell doubling, up to `d = 9` (`nawf = 9216`), and runs every
property that makes sense for Si through each sparse PAOFLOW pipeline. Every run records time,
memory and sparsity, and two notebooks plot how they scale and whether the results per primitive
cell stay the same. It tells us:

- the largest system each pipeline handles, and what stops it (memory, time, or a refusal);
- whether every property per primitive cell is unchanged by the doubling, and how far it is from
  the dense pipeline;
- measured values for the solver policy constants `dense_n_max` and `dense_ratio`, which are
  defaults today and not measurements;
- the size at which the KPM/Sternheimer formulations planned in `handoff.md` become necessary.

## Why doubled Si

The sparse engine builds its bond list from the base-cell H(R), and from then on it grows only by
`doubling_Hamiltonian`, without ever densifying. A doubled cell is still the same crystal, so
every property per primitive cell must match the primitive cell. That gives an exact reference at
every size, including sizes where no dense run fits.

The cell size doubles with each step. The base cell of example01 has `nawf = 18`, so `d = 8`
gives `nawf = 4608` and `d = 9` gives `nawf = 9216`. The k-mesh is halved along every doubled
axis, which keeps the physical k-density, and therefore the per-cell result, fixed.

## The reference

The `reference` pipeline is one case: the dense PAOFLOW pipeline on the primitive cell, with no
`doubling_Hamiltonian` call, on the full `base_nfft` mesh (48³), computing every property of every
sparse pipeline. Each sparse case is compared with it per primitive cell, and with the same
pipeline one size down:

- the deviation **from the reference** includes what the sparse settings change on purpose. The
  bond cutoff (`bond_order=36`) shifts the bands slightly (2–7% on the curves at the smoke-test
  mesh, most near the gap), and an interior window leaves out states outside it. Neither depends
  on `d`;
- the deviation **from the previous size** isolates what doubling changes, which should be zero.

## Properties

The properties live in `CONFIG['properties']` with their arguments, in the order they run.
[observables.py](observables.py) turns each one's output files into curves per primitive cell:

| property | curves | per-cell normalization |
|---|---|---|
| `dos` (with PDOS) | `dos`, `pdos_s`, `pdos_p`, `pdos_d` | / N |
| `transport` | `sigma_*`, `seebeck_*`, `kappa_*`, `pf_*` (xx, yy, zz); `hall_trace` with `do_hall` | none |
| `conductivity` | `cond_xx`, `cond_xy` (zero by symmetry, measured on the `cond_xx` scale) | / N |
| `doping` | `doping_mu`, μ against T at a p-type 10¹⁸ cm⁻³ | none |
| `dielectric_tensor` | `epsi_xx/yy/zz` (ε″ only, cut 0.5 eV below the window top) | none |
| `dielectric_tensor`, plots only | `epsr`, `eels`, `refl`, `nref`, `kref`, `alpha`, `sigmar`, `sigmai` (`optics=True`) | none |
| `effective_mass` | `inv_mass_dos`: Σ g(E − E_nk) / m_dos,nk over all states, per k-point | / (N nk) |

Left out on purpose: the Hall family (anomalous, spin and orbital Hall), `linear_response`,
`spin_texture` and `rashba_edelstein` require spin-orbit coupling, which example01 does not have.
`bands`, `fermi_surface`, `ipr` and the other per-k or per-path outputs are folded by the doubling
and cannot be compared per cell.

## Pipelines

| name | what runs | expected limit |
|---|---|---|
| `reference` | dense, primitive cell, no doubling, every property below | d = 0 only |
| `sparse` | sparse, from the bottom of the spectrum (`energy_window`), with `dense_n_max` lifted: `doping`, `conductivity`, `dielectric_tensor`, `transport`, `dos` | per-k LAPACK scratch, O(nawf²) per k, or time |
| `sparse_curvature` | the same window, but the full spectrum at every k-point for the band curvature: `transport(do_hall=True)`, `effective_mass` | a full (nawf, nawf) eigh and a few nawf³ products per k; on 4 × 4 threads d = 8 needs roughly 30 h and d = 9 several days (estimates from one d = 7 k-point) |
| `sparse_interior` | sparse, `interior_window` around E_F, `bond_order=36`; `doping` and `dielectric_tensor` are skipped (they need the states below the window) | ARPACK shift-invert cost and fill-in |
| `sparse_interior_bo8` | the same with `bond_order=8` | the truly sparse regime (see below) |
| `solver_sweep` | LAPACK against ARPACK on single H(k), for nev/n from 1% to 50%, timed on rank 0 with the threads of one rank (the other ranks wait) | calibrates `dense_ratio` and the per-k memory |

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
4. Open [observables.ipynb](observables.ipynb) for plots of every property of every sparse
   pipeline against the reference, and [analysis.ipynb](analysis.ipynb) for time, memory, the
   limits and the deviation statistics. `observables.ipynb` reads the output files directly, so it also works while the ladder
   is still running.

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
| `observables.py` | reads a case's output files into curves per primitive cell, and the deviation between two curves |
| `collect.py` | builds `results.csv` and `sweep.csv`, including every curve's deviation from the reference and from the previous size |
| `observables.ipynb` | plots of each property (no PDOS), every size of every sparse pipeline against the reference |
| `analysis.ipynb` | time, memory, sparsity, solver crossover and limits |

Each case directory holds `record.json`, the PAOFLOW `output/` (every property's files, and
`sparse.log` for sparse runs), and `error_rank<r>.json` if a rank raised an exception. A doubled
cell writes one PDOS file per orbital, `orb_<m>_pdosdk_0.dat` (9216 at d = 9), and
`observables.py` caches their sum by orbital type in `pdos_by_l.npz`.

## Known issue: velocities on doubled cells (dense only)

The dense `doubling_attr_arry` zeroes the intra-cell offset `Dnm` for orbital pairs in different
replicas. On the base cell, `Dnm` has commutator form and drops out of the band velocities.
After doubling it no longer does, and the velocities are wrong by about 35%. On an equivalent
doubled cell, σ_xx changes by about 50% and picks up anisotropy and off-diagonal components that a
cubic crystal cannot have, while the adaptive DoS changes by about 1.5%.

The sparse pipeline was fixed on 2026-10-22, and the dense one was left as it is:

- `double_axis` keeps `Dnm` as the position difference of the doubled cell for every pair;
- `_band_diagonal_velocities` keeps each state's three velocity components together inside a
  degenerate group whenever the group's velocity blocks commute. Folding makes such groups common,
  and resolving each direction separately mispairs the components.

With both fixes, sparse d = 1 and 2 match d = 0 per cell to about 1e-5 (DoS, σ, S, κ). Sparse
d = 0 still matches dense to 1e-9 in the DoS and σ_xx. Until the dense doubling is fixed, the d > 0
accuracy columns of dense cases measure this bug, and so do those of sparse runs made before the
fix. The timing and memory columns are unaffected.

## Known issues found by the smoke test

A run of the whole harness at a small mesh (16³ base, d ≤ 2) found these. They are not fixed.

- **Interband quantities on doubled cells (sparse).** The per-k kernels shared with the dense
  pipeline (`momentum_k`, `perturb_split`, `band_curvature_k_ij`) rotate a degenerate group
  separately for each direction, so in the groups that folding creates "state n" can be a
  different state in each component. The velocities no longer have this problem, but these do:
  - ε″, through the adaptive interband widths: about 4e-3 per size. With a fixed width it is 1e-4;
  - `cond_xx`, through the pairing of products with widths: about 5e-4 per size;
  - the band curvature: `hall_trace` 3e-2 and `inv_mass_dos` 0.2 per size.
- **Interior-window transport near the window edges.** The transport scan keeps 0.26 eV (10 kT)
  from each edge. The adaptive smearing widths reach about 0.2 eV, so states just outside the
  window are missing: σ is 20% low 0.3 eV inside the bottom edge and 34% low 0.4 eV inside the
  top. It is the same at every `d`.
- **Doping.** μ at a p-type 10¹⁸ cm⁻³ comes out at about 0.76 eV, in the conduction band, in the
  dense reference as well as the sparse runs. The DoS integral does not seem to include the spin
  degeneracy (it gives 4 electrons per cell, not 8). Sparse and dense agree, so the comparison is
  still valid.

## Not covered yet

- **On-site disorder.** `CONFIG['disorder_meV']` adds random on-site energies to the sparse bond
  list after doubling. They break the exact folding degeneracies, which are ARPACK's worst case.
  The price is the exact per-cell reference, so it is off by default.
- **Twisted bilayer graphene and other EDTB systems.** These go in through
  `PAOFLOW(model=...)`, which builds the full-cell dense H(R), so they cannot start above the
  dense limit. They need a direct EDTB bond list → `SparseHamiltonian` bridge first.
