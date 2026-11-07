# SSCHA in PAOFLOW: design and implementation (7 November 2026)

Status: M1-M3 implemented on `develop-marco` (uncommitted), unit-tested on a harmonic model; M4 (Slurm orchestration
in `agentic/`) and M5 (TaN validation through the API) open. python-sscha 1.6.2 and CellConstructor 1.6.2 are
installed in the `work` environment (Mac Studio) and in `~/venvs/sscha` on Snellius (build recipes in section 7). The
TaN campaign (`production/XTa_project/snellius_workflow/sscha/`) is the prototype this code was distilled from.

Decisions (user, 7 Nov 2026): separate methods; pw.x inputs from PAOFLOW's writer; electron-phonon included in the
first version; optional extra named `sscha`.

## 1. Goal

Anharmonic, temperature-dependent phonons as a phonon source of PAOFLOW, the way phonopy provides the harmonic
finite-displacement ones: the stochastic self-consistent harmonic approximation (SSCHA; Monacelli, Errea, Calandra,
Mauri, J. Phys.: Condens. Matter 33, 363001 (2021)) through python-sscha as an optional external engine.

Uses, in order of priority:

1. Lattices that are dynamically unstable in the harmonic approximation but stable at temperature (δ-TaN: X
   −264 cm⁻¹ harmonic, +95 cm⁻¹ in the SSCHA at 2800 K), and their stability temperature (free-energy Hessian).
2. Temperature-dependent force constants for the special-displacement (thermal) cells: the ZG cell is a special
   configuration of the SSCHA Gaussian, so the auxiliary force constants at T are its consistent input.
3. Electron-phonon coupling with anharmonic phonons: harmonic DFPT dvscf with SSCHA frequencies and polarisations
   (α²F(T), λ(T), resistivity, emissivity at high T).
4. Free energies of competing phases (later; not in the first version).

## 2. What the TaN prototype established

| step | prototype (`sscha_run.py`, `sscha_to_fc.py`, `sscha_interp_check.py`) | lesson |
|---|---|---|
| start | harmonic ph.x matrices of the q of the supercell grid (6³ run → Γ, L, X for a 2³ cell), `Symmetrize`, `ForcePositiveDefinite` | the q selection and the positive-definite start must be automatic |
| configurations | `Ensemble.generate(N)`, pw.x inputs written by ASE | the force runs must carry the parent run's occupations (Fermi–Dirac at k_BT for hot electrons) and settings: write them with PAOFLOW's own supercell writer (`phonon/io.py`, `occupation_lines`), not ASE |
| forces | pw.x array jobs, outputs loaded with `load_from_calculator_output` (ASE parser) | file ordering is irrelevant (each output carries its structure); a PAOFLOW force reader (`phonon/io.harvest_qe_forces`) would remove the ASE parsing dependency |
| minimisation | `SSCHA_Minimizer` (min_step_dyn 0.05, Kong–Liu 0.5, meaningful factor 1e-3); 4 populations of 100–200 configurations at 2800 and 4500 K | defaults that worked; population sizes grow when the Kong–Liu ratio drops |
| stability | `get_free_energy_hessian(include_v4=False)` from a final population of 800 | the physical phonons; the auxiliary ones are positive by construction and cannot show an instability |
| other q | `Phonons.Interpolate(coarse, fine, support_dyn_coarse=harmonic, support_dyn_fine=harmonic)`: the anharmonic correction on the supercell grid added to the harmonic matrices of the ph.x grid | stability must be judged on the fine grid (216 q for TaN), not only on Γ, X, L |
| export | `save_qe` (no `dyn0`), q2r.x → `.fc`, matdyn check | write the `dyn0` index ourselves; PAOFLOW's `qe_force_constants.real_space_force_constants` / `write_qe_force_constants` can replace q2r.x |
| downstream | `zg.py --write --force-constants <fc>`; workflow options `skip_phonon_check`, `force_constants` | the thermal cells already accept any force constants |

## 3. Public workflow (as implemented)

```python
from PAOFLOW import PAOFLOW

pao = PAOFLOW(savedir='qe/phonon/tan.save')            # scf of the primitive cell (occupations at k_B T)
pao.sscha_setup(temperature=2800.0,
                supercell_matrix=2,
                harmonic_dynamical_matrices='qe/phonon/tan.dyn',   # ph.x run on a grid containing the supercell q
                kgrid=(6, 6, 6),
                population_sizes=[100, 100, 200, 200, 300],
                hessian_size=800,
                sscha_dir='sscha_T2800')

while True:
    population = pao.sscha_population()      # dict: population, inputs (cfg_XXXX.in), hessian
    # ... pw.x on every input, output cfg_XXXX.out next to it (any launcher: Slurm array, mpirun loop)
    if population['hessian']:
        break
    pao.sscha_minimize(forces='qe')           # reads the outputs, minimises, writes pop<p>/dyn*; reports convergence

pao.sscha_hessian()                           # physical phonons from the final population; stability verdict
pao.sscha_force_constants('auxiliary')        # thermal-ensemble force constants on the ph.x grid (QE .fc)
pao.sscha_force_constants('hessian')          # physical phonons on the ph.x grid (dispersions, e-ph)

pao.special_displacement([2800.0], force_constants_file='sscha')   # ZG cells with the auxiliary force constants
pao.finish_execution()
```

The run is a state machine on disk (`<sscha_dir>/state.json`: generated, minimized, converged, hessian population),
so the driver can be re-entered after every batch of pw.x jobs: `sscha_setup` with the same temperature and supercell
continues the run, and a population that waits for its forces is returned again by `sscha_population`, never
regenerated. Execution of the force runs stays outside PAOFLOW, as for `phonons()` and `electron_phonon()`. The Slurm
orchestration (self-chaining populations, monitoring) belongs to the agentic layer (M4).

Electron-phonon with SSCHA phonons (harmonic DFPT dvscf, anharmonic frequencies and polarisations at the dense q):

```python
system = ElphonSystem(..., phonon_source='sscha',
                      sscha_dynamical_matrices='sscha_T2800/force_constants_hessian/tan.dyn')
```

`ElphonSystem.coupling_arguments` then passes `phonon_at_q` (built by `phonon_interp_from_dyn` from those matrices, on
the same q-grid as the coupling) to `carrier_scattering_dense_q`, `phonon_assisted_absorption_dense_q`,
`fan_migdal_self_energy` and `eliashberg_dense_q`, so α²F(T), λ(T), transport, optics and self-energies follow the
temperature. The `g(R_e, R_p)` cache needs no new key: it is stored in the Cartesian displacement basis and built
without the phonons, which enter only at the dense q.

## 4. Backend (as implemented)

`src/PAOFLOW/phonon/sscha_bridge.py`: procedural functions called by the orchestrator; CellConstructor and sscha are
imported lazily (`require_sscha`, `ImportError` naming the `sscha` extra).

| function | role |
|---|---|
| `dynamical_matrix_files(prefix)` | the `<prefix>1 ... <prefix>N` files of a ph.x / CellConstructor set |
| `supercell_qpoint_files(harmonic_prefix, supercell)` | the irreducible q of the ph.x grid commensurate with the supercell (Γ first) |
| `starting_matrices(harmonic_prefix, supercell, start_dir, positive_definite=True)` | CellConstructor `Phonons` of those q (copied to `start/`), symmetrised, imaginary modes made positive; harmonic and starting frequencies |
| `configuration_input(dc, structure, prefix, pp_dir, kgrid, conv_thr, mixing_beta)` | pw.x input of one configuration through `phonon/io._namelists` (occupations of the parent save, `tprnfor`), species, `CELL_PARAMETERS`/`ATOMIC_POSITIONS` in Å, `K_POINTS` |
| `generate_population(...)` | `Ensemble.generate` and one `cfg_XXXX.in` per configuration |
| `completed_outputs(population_dir)` | outputs with `JOB DONE` and the total force |
| `load_population(matrices, T, population_dir, min_fraction=0.95)` | ensemble from the outputs (python-sscha `load_from_calculator_output`, ASE parser); error below 95 % finished |
| `minimise(ensemble, min_step_dyn, kong_liu_ratio, meaningful_factor)` | `SSCHA_Minimizer`; matrices, free energy and error, convergence, auxiliary frequencies |
| `free_energy_hessian(ensemble, include_v4=False)` | physical phonons |
| `lowest_frequencies(matrices)` | lowest mode per q (Γ translations skipped) |
| `interpolate_correction(sscha, harmonic_coarse, harmonic_fine, supercell, fine_grid)` | `Phonons.Interpolate` with both supports: anharmonic correction of the supercell grid added to the harmonic fine matrices |
| `write_force_constants(matrices, prefix, grid, fc_path)` | `save_qe` + `dyn0`, then PAOFLOW's `read_dynamical_matrices` / `real_space_force_constants` / `write_qe_force_constants` (no q2r.x) |

`phonon/io._namelists` gained `conv_thr` and `mixing_beta` (defaults unchanged: 1e-8, 0.7).

State: `attr['sscha']` holds the settings of `sscha_setup` (temperature, supercell, paths, k-grid, population sizes,
minimiser settings, `nqirr`, `converged`, `stable`); `attr['sscha_force_constants_file']` (auxiliary `.fc`, read by
`special_displacement(force_constants_file='sscha')`) and `attr['sscha_dynamical_matrices']` (`{'auxiliary',
'hessian'}` → path prefix) are set by `sscha_force_constants`. Ensembles, matrices of every population and summaries
stay on disk. All CellConstructor work runs on rank 0; results are broadcast.

Not done: a PAOFLOW pw.x force reader (`harvest_qe_forces`) to drop the ASE parsing; polar crystals with SSCHA
matrices (the long-range model of the DFPT run is passed to the interpolator, but SSCHA supercell matrices have not
been checked against the non-analytic term); stress and cell relaxation (fixed cell only); free energies of
competing phases.

## 5. Dependencies and packaging

- Optional extra in `pyproject.toml`: `sscha = ["python-sscha>=1.6", "cellconstructor>=1.6", "ase>=3.22"]`
  (phonopy stays in `phonon`). `require_sscha` raises an `ImportError` naming the extra.
- Fortran extensions: build notes in the docs (section 7).
- No new runtime configuration files: every setting is an argument of `sscha_setup`.

## 6. Tests (as implemented)

`tests/unit/phonon/test_sscha_bridge.py` (unit mark; the CellConstructor tests skip without the extra; 10 tests, ~8 s).
Fixtures `tests/unit/phonon/data/sscha_tan/` (52 KB): δ-TaN 2³ harmonic matrices with Fermi-Dirac electrons at
2800 K (X −223 cm⁻¹) and the converged SSCHA auxiliary matrices at 2800 K.

1. q selection: Γ, L, X for the 2³ cell, Γ for 1³; the prefix must end in `.dyn`.
2. Starting matrices positive definite (−223.4 cm⁻¹ → positive).
3. Configurations: pw.x input with the parent occupations (`fd`, degauss), nat 16, Å cards, k-grid, `conv_thr`,
   `mixing_beta`, positions equal to the ensemble's.
4. Harmonic consistency: ASE calculator with F = −Φ u; the minimisation returns Φ and the free-energy Hessian equals Φ.
5. Interpolation exact on the 8 q of the supercell grid (matched modulo G) and equal to the harmonic fine matrices
   when SSCHA = harmonic.
6. Export: the written force constants give the X frequencies 100.3 / 103.6 cm⁻¹ through PAOFLOW's interpolator.
7. Orchestrator end to end: a model `PAOFLOW` object, the `sscha_*` methods, and a pw.x stand-in writing pw.x-format
   outputs with harmonic forces of the SSCHA fixture taken as the exact potential; starting from the unstable
   harmonic matrices the run converges (population 4 of 100 configurations) to that potential, a pending population
   is not regenerated, a missing-forces population raises, `sscha_setup` resumes, the Hessian is stable and equal to
   Φ, the exported force constants reproduce it. This also covers the force parsing (units, sign, atom order).
8. `ElphonSystem(phonon_source='sscha')`: the interpolator from the fixture gives the X frequencies; validation errors
   for an unknown source and for missing matrices; `'dfpt'` gives no interpolator.

Open: the slow integration test with a quartic model potential (convergence and an unstable verdict without DFT);
validation of the API on the TaN 2800 / 4500 K production results (M5).

## 7. Installation notes (what worked)

Mac Studio (`work`, Python 3.13, numpy on Accelerate): `/usr/local/bin` holds x86 `pkg-config` and `cmake` that
cannot run on Apple silicon and break meson. Recipe:

```bash
E=/Users/marco/anaconda3/envs/work
$E/bin/python -m pip install pkgconf          # arm64 pkg-config wheel
# blas.pc, lapack.pc, openblas.pc pointing at the env's OpenBLAS (-lopenblas) in $E/share/sscha_build/pkgconfig
export PATH=$E/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin PKG_CONFIG=$E/bin/pkgconf \
       PKG_CONFIG_PATH=$E/share/sscha_build/pkgconfig:$E/lib/pkgconfig \
       LDFLAGS="-L$E/lib -Wl,-rpath,$E/lib" SDKROOT=$(xcrun --show-sdk-path) FC=/opt/homebrew/bin/gfortran
$E/bin/python -m pip install cellconstructor python-sscha     # only these two were added (dry run checked)
```

Snellius (foss-2025a, `~/venvs/sscha` with system site packages): libblas.so / liblapack.so symlinks to OpenBLAS and a
blas.pc in `~/venvs/sscha/lib/blasfix`, exported by `~/sscha_env.sh`. Lonestar6 and aflowlib: not installed yet.

## 8. Milestones (agentic time)

| | content | status |
|---|---|---|
| M1 | `sscha_bridge.py` with unit tests | done (7 Nov 2026) |
| M2 | orchestrator methods, `special_displacement(force_constants_file='sscha')`, docstrings | done; a docs page and an example still to write |
| M3 | electron-phonon with SSCHA phonons (`ElphonSystem(phonon_source='sscha')`) and the `phonon_at_q` keyword of the dense drivers | done |
| M4 | port of the Slurm orchestration (self-chaining populations, monitors) to `agentic/` | open (1-2 h) |
| M5 | validation: TaN at 2800 / 4500 K through the API; the stability-temperature scan | open (compute hours) |
