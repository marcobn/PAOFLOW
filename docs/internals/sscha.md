# Anharmonic phonons at temperature (SSCHA)

PAOFLOW computes temperature-dependent anharmonic phonons with the stochastic self-consistent harmonic
approximation (SSCHA; Monacelli, Errea, Calandra and Mauri, *J. Phys.: Condens. Matter* **33**, 363001 (2021)),
with python-sscha and CellConstructor as the engine, the way phonopy provides the harmonic finite-displacement
phonons. The interface writes the pw.x inputs of the configurations with PAOFLOW's own supercell writer, reads the
forces back, minimises the free energy, computes the free-energy Hessian and exports force constants that the rest
of PAOFLOW consumes: the special-displacement (thermal) cells and the electron–phonon coupling.

Typical uses:

- lattices that are dynamically unstable in the harmonic approximation but stable at temperature, and the
  temperature above which they become stable (δ-TaN: X −264 cm⁻¹ harmonic, stable in the SSCHA at 2800 K);
- force constants at T for the thermal cells (the ZG cell is a special configuration of the SSCHA Gaussian);
- α²F(T), λ(T), transport and optics with anharmonic phonons (harmonic DFPT vertex, SSCHA frequencies and
  polarisations).

Three layers: the Python API (`PAOFLOW.sscha_*`), the `paoflow-gen` workflow `sscha` that writes a ready-to-run
`main.sscha.py` (local, Slurm or manual launch), and the agentic workflow `agentic/sscha/` that plans, monitors and
interprets campaigns ([Agentic SSCHA](agentic-sscha.md)). Design history: [SSCHA integration plan](sscha-integration-plan.md).

## Theory in brief

The SSCHA minimises the free energy of the ions at temperature $T$ over trial Gaussian density matrices
$\rho_{\mathcal R,\Phi}$ centred on the positions $\mathcal R$ with auxiliary force constants $\Phi$,

$$
\mathcal F[\mathcal R, \Phi] = F_\Phi + \big\langle V - \mathcal V_{\mathcal R,\Phi} \big\rangle_{\rho_{\mathcal R,\Phi}},
$$

where $F_\Phi$ and $\mathcal V_{\mathcal R,\Phi}$ are the free energy and the potential of the harmonic system with
force constants $\Phi$ and $V$ the Born–Oppenheimer energy. The averages are sampled with a *population* of
supercell configurations drawn from $\rho$, whose energies and forces come from pw.x; during the minimisation the
same configurations are reweighted to the current Gaussian until the Kong–Liu effective sample size
$N_\mathrm{eff} = (\sum_i \rho_i)^2 / \sum_i \rho_i^2$ drops below a fraction of $N$, when a new population is drawn.
In high-symmetry crystals the centroids stay at the Wyckoff positions and only $\Phi$ is minimised.

$\Phi$ is positive definite by construction: it describes the Gaussian, not the stability. The physical static
phonons are the eigenvalues of the free-energy Hessian

$$
\frac{\partial^2 F}{\partial \mathcal R_a \partial \mathcal R_b} = \Phi_{ab}
+ \overset{(3)}{\Phi}_{acd}\, \Lambda_{cdef}\, \big[1 - \overset{(4)}{\Phi}\Lambda\big]^{-1}_{efgh}\,
\overset{(3)}{\Phi}_{ghb},
$$

with the third- and fourth-order force constants averaged over the ensemble and $\Lambda$ built from the
eigenvalues of $\Phi$ (Bianco, Errea, Paulatto, Calandra and Mauri, *Phys. Rev. B* **96**, 014111 (2017)); PAOFLOW
uses the bubble term (`include_v4=False`) by default. An imaginary Hessian frequency is a dynamical instability at
$T$; near a second-order displacive transition the softest mode follows $\omega^2 \propto T - T_s$, which gives the
stability temperature $T_s$.

The supercell gives the matrices on its own q-grid only. Other q-points are obtained by interpolating the
*anharmonic correction*: the difference between the SSCHA and the harmonic matrices on the supercell grid is Fourier
interpolated and added to the harmonic matrices of a denser `ph.x` grid (CellConstructor `Interpolate` with both
supports). The result is exact on the supercell q and keeps the harmonic dispersion elsewhere; stability is judged
on a mesh finer than the supercell grid.

**Electrons.** The force runs use the occupations of the parent scf (its smearing type and width), or Fermi–Dirac
occupations at an electronic temperature $T_e$ with `degauss` $= k_B T_e$ (`electronic_temperature`; `True` sets
$T_e = T$). For metals whose instability is tied to the Fermi surface the electronic temperature matters on its own
(δ-TaN X: −264 → −223 cm⁻¹ from cold to Fermi–Dirac electrons at 2800 K), and the SSCHA at $T$ should use $T_e = T$.

## Installation

```bash
pip install -e ".[sscha]"     # python-sscha >= 1.6, cellconstructor >= 1.6, ase >= 3.22
```

Both packages build Fortran extensions with meson: a Fortran compiler and BLAS/LAPACK visible to `pkg-config` are
needed. `require_sscha()` raises an `ImportError` naming the extra when they are missing.

- **Apple silicon (conda env `work`).** `/usr/local/bin` may hold x86 `pkg-config` and `cmake` that cannot run and
  break meson. Install the `pkgconf` wheel, write `blas.pc`, `lapack.pc` and `openblas.pc` pointing at the env's
  OpenBLAS (`-lopenblas`) into a private folder, and build with
  `PATH=$E/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin PKG_CONFIG=$E/bin/pkgconf
  PKG_CONFIG_PATH=<that folder>:$E/lib/pkgconfig LDFLAGS="-L$E/lib -Wl,-rpath,$E/lib" SDKROOT=$(xcrun
  --show-sdk-path) FC=/opt/homebrew/bin/gfortran pip install cellconstructor python-sscha` (`E` = the env prefix).
- **Linux clusters (foss toolchains).** If meson does not find BLAS, link `libblas.so` / `liblapack.so` to OpenBLAS
  in a private folder with a matching `blas.pc`, and export it in the environment script the jobs load.

## Python workflow

```python
from PAOFLOW import PAOFLOW

pao = PAOFLOW.PAOFLOW(savedir='tan.save', outputdir='output')     # scf of the primitive cell
pao.sscha_setup(temperature=2800.0, supercell_matrix=2,
                harmonic_dynamical_matrices='harmonic/tan.dyn',    # ph.x on a grid containing the supercell q
                sscha_dir='output/sscha_T2800', kgrid=(6, 6, 6),
                population_sizes=[100, 100, 200, 200, 300], hessian_size=800,
                electronic_temperature=True)                       # Fermi-Dirac at k_B T in the force runs

status = pao.sscha_status()
while status['stage'] not in ('done', 'failed'):
    if status['stage'] == 'generate':
        pao.sscha_population()                 # cfg_XXXX.in of the next population
    elif status['stage'] == 'forces':
        ...                                    # run pw.x on status['missing'] (cfg_XXXX.out next to each input)
    elif status['stage'] == 'minimize':
        pao.sscha_minimize()                   # pop<p>/dyn*, pop<p>/summary.json
    elif status['stage'] == 'hessian':
        pao.sscha_hessian()                    # hessian/dyn*, hessian/summary.json (stable, softest mode per q)
    elif status['stage'] == 'export':
        pao.sscha_force_constants('auxiliary')     # the Gaussian at T: input of the thermal cells
        pao.sscha_force_constants('hessian')       # the physical phonons: dispersions, electron-phonon
    status = pao.sscha_status()
```

| method | role |
|---|---|
| `sscha_setup(temperature, supercell_matrix, harmonic_dynamical_matrices, sscha_dir, kgrid, population_sizes, hessian_size, min_step_dyn, kong_liu_ratio, meaningful_factor, pp_dir, prefix, conv_thr, mixing_beta, positive_definite, electronic_temperature, start_from)` | the q of the supercell grid from the ph.x run (Γ first), the starting matrices (imaginary modes made positive, or the converged matrices of `start_from`: another run's `sscha_dir` or a `<dir>/dyn` prefix), `state.json`. An existing run with the same temperature, supercell and electrons is continued; a different one is refused |
| `sscha_status(min_fraction=0.95)` | the next step from the files (`stage`: `generate`, `forces`, `minimize`, `hessian`, `export`, `done`, `failed`), the pending inputs and the missing ones, the population history |
| `sscha_population()` | the configurations of the next population (the Hessian population after convergence) and their pw.x inputs; a population that waits for its forces is returned again, never regenerated |
| `sscha_minimize(forces='qe')` | reads the pw.x outputs (at least 95 % finished), minimises, writes the matrices and `summary.json` (free energy and error, Kong–Liu ratio, convergence, lowest auxiliary frequencies) |
| `sscha_hessian(include_v4=False)` | the free-energy Hessian from the final population; `stable` (no mode below −0.5 meV on the supercell grid) |
| `sscha_force_constants(matrices='auxiliary'|'hessian', fine_grid=None)` | the anharmonic correction on the ph.x grid (or `fine_grid`): QE matrices `force_constants_<kind>/<prefix>.dyn*` and the `flfrc` file `<prefix>_sscha_<kind>.fc` (PAOFLOW's q2r, no `q2r.x`) |

All CellConstructor work runs on rank 0 and the results are broadcast; the driver does not need `mpirun`. The pw.x
inputs are written by `phonon/io._namelists` (cutoffs, spin and pseudopotentials of the parent save, `tprnfor`,
`conv_thr` 10⁻¹⁰ and `mixing_beta` 0.3 by default) with `CELL_PARAMETERS` and `ATOMIC_POSITIONS` in Å in the atom
order of the ensemble; each configuration writes to its own `outdir = ./tmp_<prefix>_p<P>_<i>`.

## The `paoflow-gen` workflow `sscha`

`paoflow-gen` → workflow `sscha` asks for the scf save and prefix, the pseudopotential directory, the temperatures,
the supercell, the ph.x matrices (detected under the working directory; written by the `harmonic` phase if missing)
and the q-grid of that run, the k-grid of the supercell, the electrons (`parent`, `ionic` = Fermi–Dirac at each
temperature, `fixed` = one temperature), the population sizes and the Hessian size, the launcher and its settings,
and writes `main.sscha.py` (`-o` to rename, `--force` to overwrite). Without prompts:
`PAOFLOW.gen.sscha_driver.build_sscha_script(cfg)` (the agentic campaign template does this).

```bash
python main.sscha.py harmonic        # ph.x on HARMONIC_QGRID (only when HARMONIC_DYN does not exist)
python main.sscha.py run             # the whole workflow at every temperature (default phase)
python main.sscha.py step            # one decision per temperature (what a Slurm step job runs)
python main.sscha.py status          # where every temperature stands
python main.sscha.py run --temperature 2000
```

| launcher | what `run` does |
|---|---|
| `local` | runs pw.x (`MPI_QE QE_PW -in cfg_XXXX.in > cfg_XXXX.out`) configuration after configuration, minimises, draws the next population, … to the exported force constants; a failed configuration is skipped (the 95 % rule) and its scratch removed |
| `slurm` | writes `pw_array.sh` (one configuration per array task, `SLURM_DIRECTIVES`, `ENV_SETUP`, at most `SLURM_CONCURRENT` at a time) and `step.sh` (`SLURM_STEP_DIRECTIVES`, `--dependency=afterany` on the array) and submits both; each step job minimises and submits the next population, to the end. A population that comes back with fewer than 95 % finished stops the chain; `run` resubmits the missing configurations |
| `manual` | writes the inputs of the next population and lists the missing ones; run them with any launcher and call `run` again |

The state of each temperature lives in `output/sscha_T<T>/state.json`, so every phase can be repeated, interrupted
and resumed, also from another machine after copying the run directory. Constants of the driver (edit freely):
`TEMPERATURES`, `SUPERCELL`, `KGRID`, `ELECTRONIC_TEMPERATURE`, `POPULATION_SIZES`, `HESSIAN_SIZE`, `MIN_STEP_DYN`,
`KONG_LIU_RATIO`, `MEANINGFUL_FACTOR`, `CONV_THR`, `MIXING_BETA`, `START_FROM` (warm starts, e.g.
`{2000.0: sscha_dir(2800.0)}`), `HARMONIC_DYN`, `HARMONIC_QGRID`, `FINE_GRID`, `LAUNCHER`, `MPI_QE`, `QE_PW`,
`QE_PH`, `SLURM_DIRECTIVES`, `SLURM_STEP_DIRECTIVES`, `SLURM_CONCURRENT`, `ENV_SETUP`, `PYTHON`.

## Outputs

| path (under `sscha_dir`) | content |
|---|---|
| `state.json` | temperature, supercell, electrons, `nqirr`, progress (`generated`, `minimized`, `converged`, `hessian_population`, `hessian_done`, `stable`, `exported`), population budget and minimiser settings, harmonic and starting frequencies |
| `start/dyn*`, `pop0/dyn*` | harmonic matrices of the supercell q; starting matrices |
| `pop<p>/cfg_XXXX.in`, `.out` | pw.x inputs and outputs of population p |
| `pop<p>/dyn*`, `pop<p>/summary.json` | matrices after the minimisation of p; free energy and error (Ry), Kong–Liu ratio, size, convergence, lowest auxiliary frequencies (cm⁻¹) |
| `hessian/dyn*`, `hessian/summary.json` | free-energy Hessian; `stable` and the softest mode per q |
| `force_constants_auxiliary/`, `force_constants_hessian/` | QE matrices on the ph.x grid (`<prefix>.dyn0` … `dynN`) |
| `<prefix>_sscha_auxiliary.fc`, `<prefix>_sscha_hessian.fc` | real-space force constants (`flfrc` format) |

## Downstream use

- **Thermal cells.** `special_displacement([T], force_constants_file='sscha')` after
  `sscha_force_constants('auxiliary')` takes `attr['sscha_force_constants_file']` ([Special displacement
  method](special-displacement.md)).
- **Electron–phonon.** `ElphonSystem(..., phonon_source='sscha',
  sscha_dynamical_matrices='output/sscha_T2800/force_constants_hessian/tan.dyn')`: the coarse vertex stays the
  harmonic DFPT one, the dense-q phonons (frequencies and polarisations) come from the SSCHA matrices, which must be
  on the q-grid of the coupling. `coupling_arguments` passes `phonon_at_q` to `eliashberg_dense_q`,
  `carrier_scattering_dense_q`, `phonon_assisted_absorption_dense_q` and `fan_migdal_self_energy`. The cached
  `g(R_e, R_p)` is in the Cartesian displacement basis and independent of the phonons, so it is shared between
  harmonic and anharmonic runs ([Electron–Phonon Coupling](Electron-Phonon-Coupling.md), "Dense-q phonons").
- **Dispersions and stability on a fine mesh.** `PAOFLOW.agentic.sscha.dispersion` and `grid_minimum` read any set
  of QE matrices (harmonic or exported SSCHA).

## Module map

| module | content |
|---|---|
| `phonon/sscha_bridge.py` | `require_sscha`, `dynamical_matrix_files`, `supercell_qpoint_files`, `starting_matrices`, `configuration_input`, `generate_population`, `completed_outputs`, `missing_configurations`, `run_status`, `load_population`, `minimise`, `free_energy_hessian`, `lowest_frequencies`, `interpolate_correction`, `write_force_constants`; constants `RY_TO_CM1`, `AMU_RY`, `UNSTABLE_CM1` |
| `phonon/io.py` | `fermi_dirac_lines(T)`; `_namelists(..., conv_thr, mixing_beta, occupations)` |
| `PAOFLOW.py` | `sscha_setup`, `sscha_status`, `sscha_population`, `sscha_minimize`, `sscha_hessian`, `sscha_force_constants`; `special_displacement(force_constants_file='sscha')` |
| `elphon/workflows/system.py` | `ElphonSystem.phonon_source`, `sscha_dynamical_matrices`, `phonon_interpolator` |
| `gen/sscha_driver.py` | `collect_sscha`, `build_sscha_script` (the `paoflow-gen` workflow) |
| `agentic/sscha/` | `SSCHARun`, `find_runs`, `scan_table`, `stability_temperature`, `next_temperature`, `dispersion`, `grid_minimum`, `plotting` |

## Tests

- `tests/unit/phonon/test_sscha_bridge.py`: q selection (Γ, L, X of a 2³ fcc cell), positive-definite start,
  configuration inputs (parent and Fermi–Dirac occupations), the analytic harmonic test (forces −Φu: the
  minimisation returns Φ and the Hessian equals Φ), interpolation exact on the supercell grid, export roundtrip,
  `missing_configurations`, `ElphonSystem(phonon_source='sscha')`, and the orchestrator end to end with a pw.x
  stand-in (convergence from the unstable harmonic start in four populations, status stages, resume, the
  refusal to overwrite another run, warm start with hot electrons).
- `tests/unit/gen/test_sscha_driver.py`: prompts, the generated text, and the generated driver run end to end with
  the local, Slurm (a fake `sbatch` on `PATH`: array specification, dependency, scripts, stop and resubmission of an
  incomplete population) and manual launchers.
- `tests/unit/agentic/sscha/`: the run adapter and its diagnostics, scans and the stability temperature,
  dispersions, figures.

Fixtures: `tests/unit/phonon/data/sscha_tan/` (δ-TaN 2³ harmonic matrices with Fermi–Dirac electrons at 2800 K and
the converged SSCHA matrices at 2800 K; 52 KB). The CellConstructor tests skip without the extra.

## Limitations

- Fixed cell (no stress, no thermal expansion): $T_s$ refers to the lattice parameter of the parent scf.
- Free energies of competing phases are not compared.
- Polar crystals: the long-range model of the DFPT run is passed to the interpolator of the electron–phonon
  workflow, but the SSCHA supercell matrices have not been validated against the non-analytic term.
- The forces are read through ASE's pw.x parser (python-sscha's `load_from_calculator_output`).

## References

- L. Monacelli, R. Bianco, M. Cherubini, M. Calandra, I. Errea and F. Mauri, *J. Phys.: Condens. Matter* **33**,
  363001 (2021): the SSCHA method and python-sscha.
- R. Bianco, I. Errea, L. Paulatto, M. Calandra and F. Mauri, *Phys. Rev. B* **96**, 014111 (2017): the free-energy
  Hessian and second-order phase transitions.
- I. Errea, M. Calandra and F. Mauri, *Phys. Rev. B* **89**, 064302 (2014): the stochastic implementation.
