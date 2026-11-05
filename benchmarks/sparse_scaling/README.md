# Sparse-scaling benchmark

This benchmark grows crystals by cell doubling and runs their properties through each sparse
PAOFLOW pipeline, size by size. Every run records time, memory and sparsity, and the notebooks
plot how they scale and whether the results per primitive cell stay the same. It tells us:

- the largest system each pipeline handles, and what stops it (memory, time, or a refusal);
- whether every property per primitive cell is unchanged by the doubling, and how far it is from
  the dense pipeline;
- measured values for the solver policy constants `dense_n_max` and `dense_ratio`, which are
  defaults today and not measurements;
- the size at which the KPM/Sternheimer formulations planned in `handoff.md` become necessary.

## Systems

Each system has its own folder under [systems/](systems/), with its settings (`system.py`), its
notebook of curves (`observables.ipynb`) and its runs (`runs/`). The code that builds, runs and
reads the cases is shared, in this folder.

| system | data | `nawf` per cell | largest d (`nawf`) | what only it tests |
|---|---|---|---|---|
| [Si](systems/si/) | example01 | 18 | 8 (4608) | doping and transport of a cubic semiconductor, the interior-window ladders, the solver sweep |
| [Pt](systems/pt/) | example05 | 18 | 4 (288) | `spin_Hall` (with `do_ac`), `orbital_Hall`, `linear_response('shc')`; transport of a metal, and an interior window through E_F |
| [Fe](systems/fe/) | example04 | 26 | 4 (416) | `anomalous_Hall` (with `do_ac`) and `linear_response('ahc')`, both non-zero; transport of a magnetic metal |
| [Te](systems/te/) | example11 | 54 | 4 (864) | `rashba_edelstein` (spin and orbital), `linear_response('ree')`, `spin_texture` (d = 0 only); transport of a trigonal semiconductor |

Pt, Fe and Te have spin-orbit coupling from the DFT run, which Si does not, so between them the
four systems run every mesh property of PAOFLOW that can be compared per cell. The `system.py` of
each explains its energy ranges, cutoff and symmetry.

The Pt, Fe and Te save directories (`systems/pt/pt.save`, `systems/fe/fe.save`,
`systems/te/Te-L.save`, 37 MB) are extracted from the QE test assets
(`qe-test_assets_integration-assets-v1.tar.gz`, members `example05/pt.save`, `example04/fe.save`
and `example11/Te-L.save`; see `tests/integration/qe/README.md`). They are not under version
control (`.gitignore`), so copy them along with the benchmark to the cluster. `submit.py` refuses
to submit a system whose save directory is missing.

Pt, Fe and Te run to d = 4 for now, a first pass that measures how the cost grows per size. Each
size costs the full-spectrum pipeline (`sparse_curvature`) about 2.5 times the previous one at
large d (Si took 1.7 h at `nawf = 2304`), so d = 7 for Pt and Fe, and d = 6 for Te (whose base
cell is three times larger), are the most a 24 h walltime would allow; raise `max_doublings` once
the first pass shows the growth.

## Why doubled crystals

The sparse engine builds its bond list from the base-cell H(R), and from then on it grows only by
`doubling_Hamiltonian`, without ever densifying. A doubled cell is still the same crystal, so
every property per primitive cell must match the primitive cell. That gives an exact reference at
every size, including sizes where no dense run fits.

The cell size doubles with each step: Si's base cell has `nawf = 18`, so `d = 8` gives
`nawf = 4608`. The k-mesh is halved along every doubled axis, which keeps the physical k-density,
and therefore the per-cell result, fixed. The spin and orbital operators `Sj`/`Lj` are built at
the base cell (the `operators` stage) and doubled with it.

## The reference

The `reference` pipeline of each system is one case: the dense PAOFLOW pipeline on the primitive
cell, with no `doubling_Hamiltonian` call, on the full `base_nfft` mesh, computing every property
of every sparse pipeline. Each sparse case is compared with it per primitive cell, and with the
same pipeline one size down:

- the deviation **from the reference** includes what the sparse settings change on purpose. Si's
  bond cutoff shifts the bands slightly (2–7% on its curves at the smoke-test mesh, most near the
  gap), the sparse pass leaves out the placeholder states at the PAO shift that the dense one
  keeps, which every sum over pairs of states sees (the Kubo conductivity, the band curvature),
  and an interior window leaves out states outside it. None of these depends on `d`;
- the deviation **from the previous size** isolates what doubling changes, which should be zero.

## Properties

A system's properties live in `SYSTEM['properties']` with their arguments, in the order they run.
A key `'<method>:<tag>'` calls `<method>`, so `linear_response` runs once per response.
[observables.py](observables.py) turns each one's output files into curves per primitive cell:

| property | curves | per-cell normalization |
|---|---|---|
| `dos` (with PDOS) | `dos`, `pdos_s`, `pdos_p`, `pdos_d` (PDOS without spin-orbit coupling only, see below) | / N |
| `transport` | `sigma_*`, `seebeck_*`, `kappa_*`, `pf_*` (xx, yy, zz, and xy, xz, yz, which symmetry zeroes, on the xx scale) | none |
| `transport(do_hall=True)` | `hall_trace`; `hall_xyz`, `hall_xzy`, `hall_yzx`; `nernst_xyz`, `nernst_yxz`; `hall_zero`, `nernst_zero` (RMS of the components symmetry zeroes) | none |
| `conductivity` | `cond_xx`, `cond_xy` (zero by symmetry, measured on the `cond_xx` scale), `cond_zz` (Te) | / N |
| `doping` | `doping_mu`, μ against T at a p-type doping | none |
| `dielectric_tensor` | `epsi_xx/yy/zz` (ε″ only, cut 0.5 eV below the window top) | none |
| `dielectric_tensor`, plots only | `epsr`, `eels`, `refl`, `nref`, `kref`, `alpha`, `sigmar`, `sigmai` (`optics=True`) | none |
| `effective_mass` | `inv_mass_dos`: Σ g(E − E_nk) / m_dos,nk over all states, per k-point; `inv_mass_trace`: the same with tr m⁻¹_nk / 3 | / (N nk) |
| `linear_response` (`'cond'`, `'ahc'`, `'shc'`) | `lr_cond_*`, `lr_ahc_*`, `lr_shc_even_*` | none |
| `linear_response('ree')` | `lr_ree_even_*` | / N |
| `spin_Hall`, `orbital_Hall`, `anomalous_Hall` | `berry_shc_*`, `berry_ohc_*`, `berry_ahc_*`; with `do_ac`, `scd_re/im_*`, `mcd_re/im_*` | none |
| `rashba_edelstein` | `ree_spin_*`, `ree_orbital_*` (the accumulation over the current, a ratio) | none |
| `spin_texture` | `spin_texture`: the RMS of \|⟨S⟩\| over the mesh, per band, d = 0 only | none |
| `bands` | `bands`: the band path, the bands one after another, d = 0 only (`reference` and `sparse`); its deviation is the RMS band difference in eV | none |

Components that symmetry zeroes are measured on the scale of a symmetry-allowed partner. The
defaults, which hold in all four systems, are in `observables.DEFAULT_ZEROS`; a system adds its
own in `SYSTEM['zeros']` (σ^z_xz of the spin Hall conductivity of Pt, say) and removes the ones
that do not vanish for it in `SYSTEM['nonzero']` (the anomalous Hall σ_xy of Fe).

Left out on purpose: `fermi_surface`, `ipr`, `orbital_texture` on the mesh and the other per-k
or per-path outputs are folded by the doubling and cannot be compared per cell. The band path
and the spin texture are compared at d = 0 only, where they show what the sparse settings
change. `jdos` reads the DFT eigenvalues, not the PAO Hamiltonian. The
notebooks list what each system leaves out.

## Pipelines

| name | systems | what runs | expected limit |
|---|---|---|---|
| `reference` | all | dense, primitive cell, no doubling, every property of the system | d = 0 only |
| `sparse` | all | sparse, from the bottom of the spectrum (window from the property ranges), with `dense_n_max` lifted: the band-diagonal properties | per-k LAPACK scratch, O(nawf²) per k, or time |
| `sparse_curvature` | all | the same window, but the full spectrum at every k-point, for the band curvature (`transport(do_hall=True)`, `effective_mass`) and every property that sums over pairs of states (`linear_response`, the Hall conductivities) | a full (nawf, nawf) eigh and a few nawf³ products per k; on 4 × 4 threads Si's d = 8 needed roughly 30 h before `band_curvature_k` made the curvature 5–8× cheaper (estimate from one d = 7 k-point) |
| `sparse_interior` | Si, Pt, Fe | sparse, `interior_window` around E_F; `doping` and `dielectric_tensor` are skipped (they need the states below the window) | ARPACK shift-invert cost and fill-in |
| `sparse_interior_bo8` | Si | the same with `bond_order=8` | the truly sparse regime (see below) |
| `solver_sweep` | Si | LAPACK against ARPACK on single H(k), for nev/n from 1% to 50%, timed on rank 0 with the threads of one rank (the other ranks wait) | calibrates `dense_ratio` and the per-k memory |

Si PAO hoppings decay slowly. With `bond_order=36` (rcut 32 Bohr), one row of H(k) holds about
8500 nonzeros, so H(k) is effectively dense until `nawf` is around 10⁴. With `bond_order=8` it
holds about 890. The two interior ladders separate the cost of the cutoff from the cost of the
algorithm; they measure the algorithm, so only Si runs them.

Pt, Fe and Te keep every H(R) element (`hopping_threshold=0`). Their QE k-grids are coarse
(12³, and 6 × 6 × 4 for Te), so the largest shell cutoff they represent is 22–26 Bohr, and it moves
the bands near E_F by 47 (Pt), 140 (Fe) and 110 meV (Te) on the band path. That alone took the
anomalous Hall conductivity of Fe 44% away from the reference. Without a cutoff, sparse d = 0
matches the dense reference: the bands to 3–7 meV near E_F (35 meV for Te, the off-grid
interpolation on its coarse grid), and the anomalous Hall conductivity to 2×10⁻³. H(k) is dense
at these sizes anyway. The full bond list is 2.6 times larger: Fe's doubling peaks at about 8 GB
per rank at d = 4, but 63 GB at d = 7, where its sparse pipelines would need 2 ranks per job.

## Running it

1. Edit [config.py](config.py) for what the systems share:
   - `CONFIG['systems']`, the systems to submit (Si already ran, so it is not in the list:
     adding it back reruns it and overwrites its runs);
   - the Slurm account, partition and `setup` lines (module loads, virtualenv);
   - the default resources of a job.

   Edit `systems/<name>/system.py` for one system: its properties, its pipelines, their maximum
   doubling and any resources that differ from the defaults (`'slurm'` of a pipeline). Nothing
   is passed on the command line.
2. Submit:

   ```bash
   cd benchmarks/sparse_scaling
   python submit.py
   ```

   This writes `systems/<name>/runs/cases.json` and one job array per pipeline of each system
   (one task per size), and a finalize job that waits for all of them (`afterany`). Every size
   is submitted at once, because a failing size is a data point, not a reason to stop. Set
   `CONFIG['slurm']['dry_run'] = True` to write the scripts without submitting them.
3. When the jobs finish, the finalize job runs `finalize.py` and `collect.py` on each system.
   `finalize.py` adds the sacct state and MaxRSS to every record and marks jobs killed for OOM
   or timeout. `collect.py` writes `results.csv` and `sweep.csv` into the system's `runs/`. Run
   by hand, both go through every system that has runs.
4. Open `systems/<name>/observables.ipynb` for plots of every property of every sparse pipeline
   against the reference, and [analysis.ipynb](analysis.ipynb) (set `SYSTEM` in its first cell)
   for time, memory, the limits and the deviation statistics. The observables notebooks read the
   output files directly, so they also work while a ladder is still running.

### Resources

- **Ranks per node.** The sparse bond list is replicated on every rank, so more ranks per node
  means *less* memory per rank. Prefer few ranks with many threads: the threads go to BLAS and
  LAPACK through `OMP_NUM_THREADS` and `MKL_NUM_THREADS`, which the job scripts set from
  `--cpus-per-task`.
- **Pools.** `npool` should equal `ntasks`, which gives one pool per rank.
- **Node memory.** The default request is `--mem=2500` (MB), because a larger one waits in
  tigu's queue. tigu does not enforce it: Si's `sparse_curvature` asked for 2500 and used 35 GB.
  On a cluster that enforces `--mem`, use `--mem=0` (the whole node) with `--exclusive`, so a job
  fails on the benchmark's own memory and not on a request that is too small.
- **CPUs.** Each job takes 12 CPUs by default (4 ranks of 3 threads).
- **Walltime.** Each pipeline's `time` is a cap, and a case that hits it becomes `timeout` in
  the results. Calibrate it from a first pass.

## Files

| file | role |
|---|---|
| `config.py` | the settings the systems share, and `load_system(name)` |
| `systems/<name>/system.py` | one system: data, mesh, properties, pipelines, symmetry zeros |
| `systems/<name>/observables.ipynb` | plots of each property (no PDOS), every size of every sparse pipeline against the reference |
| `systems/<name>/runs/` | `cases.json`, the job scripts and logs, and one directory per case |
| `ladder.py` | `build_cases(config, name, system)`: the sizes, doubling patterns and meshes |
| `run_case.py` | `run_case(case, outdir)`, which runs one case. Its `__main__` is the Slurm array task (it reads `SLURM_ARRAY_TASK_ID` and `BENCH_OUTPUT_ROOT`) |
| `recorder.py` | per-stage timing between MPI barriers, per-stage peak RSS (`VmHWM`, reset each stage), array sizes and environment; rewrites `record.json` after every stage, so a killed job still leaves a partial record |
| `submit.py` | writes `cases.json` and the job scripts of each system, then calls `sbatch` |
| `finalize.py` | adds the sacct state and MaxRSS to each record |
| `observables.py` | reads a case's output files into curves per primitive cell, and the deviation between two curves |
| `collect.py` | builds `results.csv` and `sweep.csv`, including every curve's deviation from the reference and from the previous size |
| `plots.py` | the figures of the observables notebooks |
| `analysis.ipynb` | time, memory, sparsity, solver crossover and limits, for one system |
| `runs/` | the finalize job and its log, which span the systems |

Each case directory holds `record.json`, the PAOFLOW `output/` (every property's files, and
`sparse.log` for sparse runs), and `error_rank<r>.json` if a rank raised an exception. A doubled
cell writes one PDOS file per orbital, `orb_<m>_pdosdk_0.dat` (4608 at d = 8), and
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

## Known issues found by the smoke test (Si)

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

## Found by the spin-orbit smoke test (Pt, Fe, Te)

A run at a 12³ mesh (12 × 12 × 8 for Te), d ≤ 1, on 2026-10-26. The per-system notebooks have
the numbers. Fixed:

- **Sparse `bands()` crashed on a default path** (`KeyError: 'write_QE_bands'`): the engine did
  not set the flag the path writer reads, nor take the dense `write_QE_path` argument. It does
  now.

Not fixed; they are in shared or dense code:

- **SOC PDOS files overwrite each other.** `_build_orbital_prefixes` gives both spinor copies of
  an orbital the same prefix (`Pt1_6pz` twice), so the undoubled cell writes the second over the
  first, in the dense and the sparse pipeline. Doubled cells write `orb_<m>` and keep them all.
  `observables.py` skips the PDOS when prefixes repeat.
- **`doping` double counts spinor bands.** `calc_N`/`solve_for_mu` weight the DoS by
  `dosweight=2.0` always, so with spin-orbit coupling μ lands deep in the valence band (Te: in the
  4d shell, −36 eV). Te does not run `doping`.
- **`linear_response` reads `a_tensor` only when `s_tensor` is given** (dense and sparse). Without
  it the tensor comes from `arrays['a_tensor']`, which is all nine components by default, or what
  an earlier `anomalous_Hall` left there. Every system passes both.

Seen in the sparse engine, not yet explained:

- **The band-diagonal Rashba-Edelstein response of Te moves by 7–27% per doubling**, while σ, S
  and `cond` stay within 10⁻¹¹. A likely cause, not checked state by state: doubling along x makes
  k and −k exactly degenerate on the planes k_x = ±1/4, and `texture_k` takes ⟨S⟩ from the raw
  eigenvectors while the velocities are resolved inside each degenerate group, so ⟨S⟩ and v
  belong to different states there.
- **The band curvature of Fe drifts per doubling** (Hall 4–8%, Nernst 6–8%, masses 5–11% at
  d = 1), as Si's does; Pt's does not (10⁻¹²). At d = 0 the curvature family also differs from the
  dense reference in every system (Hall 1–15%, Nernst 2–12%, masses 0.1–0.7), with the exact
  H(R) and everything else matching to 10⁻³ or better.
- **Te's symmetry zeros are not small** at the smoke mesh, in dense and sparse alike: REE xy is
  14–20% of xx, and the Hall and Nernst components with a repeated index are of the order of the
  allowed ones. Point group 32 zeroes them; check them at the ladder's mesh.

## Not covered yet

- **On-site disorder.** `CONFIG['disorder_meV']` adds random on-site energies to the sparse bond
  list after doubling. They break the exact folding degeneracies, which are ARPACK's worst case.
  The price is the exact per-cell reference, so it is off by default.
- **Twisted bilayer graphene and other EDTB systems.** These go in through
  `PAOFLOW(model=...)`, which builds the full-cell dense H(R), so they cannot start above the
  dense limit. They need a direct EDTB bond list → `SparseHamiltonian` bridge first.
