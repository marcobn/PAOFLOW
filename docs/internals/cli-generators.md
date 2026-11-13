# CLI Input and Script Generators

Setting up a PAOFLOW calculation involves quite a bit of boilerplate — QE input files, driver scripts, plotting routines. These command-line tools handle that for you, letting you focus on the science rather than the setup. Two further console commands are documented elsewhere: `paoflow-genbasis-ps`, which generates the atomic-orbital basis tables from the pseudopotentials ([Internal basis](internal-basis.md)), and `pyskeaf` / `fermi-plotter` for the quantum-oscillation analysis. Both rely primarily on Python's standard library and produce static, well-commented scripts that are easy to read and modify.

## `paoflow-gen-qe`

Converts an AFLOW database entry into a ready-to-run Quantum ESPRESSO `scf` input file, with sensible defaults for smearing, magnetism, spin-orbit coupling, and the band count needed for PAOFLOW's extended-basis projections.

**Accepted input formats:**
- AFLOWDATA URLs
- Material-page URLs
- Bare AUID tokens (e.g., `aflow:0a66d228d896a855`)

**What it figures out for you:**
- Lattice type (`ibrav` and `celldm` parameters)
- Kinetic energy cutoffs from reference data
- Whether the material is a metal or insulator (via band-gap analysis)
- Spin polarization when needed
- Spin-orbit coupling setup
- Band count optimized for extended PAO basis calculations
- Recommended intersite-V cutoffs for follow-up U+V runs

**Main options:** pseudopotential directory (required), spin-orbit coupling flag, output path, smearing width, and symmetry tolerance.

## `paoflow-gen`

Generates a PAOFLOW property-calculation driver (`main.py`) from a completed QE run, along with an optional plotting script (`plot.py`) that visualizes the properties you selected.

The questions of every workflow are data: the `Question` and `Note` lists of `PAOFLOW.gen.questions` (one module
per workflow), plus an `assemble` function that turns the answers into the configuration the script builders
read. The terminal prompts (`paoflow-gen`) and the browser form (`paoflow-gen --gui`, or the Setup tab of
`paoflow-studio`) both read them, so a new option needs one `Question`. Mark it `cli=False` for a GUI-only refinement
that the CLI leaves at the builder default.

The tests in `tests/unit/gen/` keep the two routes identical:

- `test_questions_parity.py` runs the frozen pre-refactor prompts (`legacy_collect.py`) and the declarative ones
  on the same answers, and requires the same prompts, configuration and files;
- the builder-coverage test fails when a builder reads a key that no question sets.

`PAOFLOW.gen.outputs` lists the files of a workflow before writing (`workflow_outputs`) and its commands
(`run_plan`). Some defaults come from the QE save and the `pw.x` template (`PAOFLOW.gen.system_summary`):
`ibrav`, spin-orbit and 2D; for the electron–phonon workflow, metal, `nelec`, `nbnd` and the coupling source.
See [the browser interface](../tutorials/howtos/browser-gui.md).

**Errors under MPI.** Every driver (`main.py`, `main.acbn0.py`, `main.phonon.py`, `main.raman.py`,
`main.sscha.py` and the electron–phonon drivers that run under `mpirun`) ends with
`run_main(main)` from `PAOFLOW.utils.mpi_guard`. An exception on one rank otherwise ends that rank only, and the
others wait for ever in the next collective operation, keeping their cores busy. `run_main` prints the
traceback of the failing rank and calls `MPI_Abort`, so the job stops with a non-zero status; on one rank, or
without MPI, it just calls `main()`. The plot scripts run as one process and do not need it.

**Launch settings.** Every generated script that starts Quantum ESPRESSO itself keeps its launch commands
(the CLI answers) as defaults and reads the variables of `PAOFLOW.utils.qe_launch` first:
- the scripts: `main.acbn0.py`, the `run` phases of `main.phonon.py` and `main.raman.py`, `main.sscha.py` with
  the local launcher, and `qe/run_qe.sh`, `run_zg.sh` and `run_fd.sh` of the electron–phonon workflows;
- the variables:
  - `PAOFLOW_MPI_QE`: the MPI prefix, empty for a serial run;
  - `PAOFLOW_QE_BIN`: the QE bin folder;
  - `PAOFLOW_QE_OPTIONS`: options of `pw.x`/`ph.x` such as `-nk 4`;
  - for ACBN0 also `PAOFLOW_MPI_HARTREE` (default: `PAOFLOW_MPI_QE`), `PAOFLOW_MPI_PY` and
    `PAOFLOW_PYTHON_BIN`.

The Runs tab of `paoflow-studio` sets them from the ranks, the QE build and the options of the step
(`RunStep.qe_launch`). Its forms hide these questions: `Question(gui=False)` is asked by the CLI, and the GUI
form keeps the default. The SSCHA `pw.x` command uses `gui=<predicate>` and is shown for the Slurm launcher
only. In a terminal: `PAOFLOW_MPI_QE='mpirun -np 8' PAOFLOW_QE_OPTIONS='-nk 4' python main.phonon.py run`.

**Supported workflows** (chosen at the first prompt):

**Workflow A — ACBN0/eACBN0:** Self-consistent Hubbard U calculations, with or without intersite V terms. See [ACBN0 Module](acbn0.md) for details on the underlying implementation.
The launch commands asked by the CLI (MPI commands of `pw.x`, python and the Hartree step, the QE and Python
bin paths) are the defaults of `main.acbn0.py`; the launch variables below override them, with
`PAOFLOW_QE_OPTIONS` passed to `ACBN0(qe_options=...)`.

**Workflow B — Property runs:** Pick from band structure, DOS, transport, Fermi surface, spin texture, spin Hall conductivity, orbital Hall conductivity, anomalous Hall effects, topology, optical properties, and more. The generated `plot.py` includes only the visualization routines for the properties you chose.

The property `nonlinear` (non-linear optics) adds a separate `run_nonlinear()` to `main.py`: a PAOFLOW instance on the **extended** basis of the run's pseudopotentials (the BASIS_PS tables that `main.py` generates) that calls `nonlinear_optics(...)` right after `pao_hamiltonian()` (it needs `HRs`, so it never interpolates). Every option is an editable constant: `NLO_NK`, `NLO_EMIN/NLO_EMAX/NLO_NE`, `NLO_ETA`, `NLO_SCISSOR`, `NLO_COMPONENTS`, `NLO_RESPONSES` (shift, injection, shg), `NLO_DIPOLES` (default `True`, the validated setting; `False` with projwfc projections, which have no internal basis), `NLO_SYMMETRY`, `NLO_SHG_TERMS` (the SHG split into interband / intraband terms at ω and 2ω), `NLO_TWOD` and `NLO_SLAB_HEIGHT` (sheet units; on for 2D systems), `NLO_MOLECULE`, `NLO_COUPLING` and `NLO_COMPLEMENT_SHIFT` (molecules: Γ point, the coupled response of the molecule, default `hartree+alda`), and, for metals and semimetals (question *metal or semimetal*), `NLO_FERMI_SURFACE` (`True`, else `'auto'`, which detects a metallic channel), `NLO_TEMPERATURE` (K), `NLO_TAU` (intraband relaxation time, fs) and `NLO_COHERENCE_DAMPING` (second-order coherences damped with 1 × η, or 2 × η, the velocity-gauge Kubo limit). The metal engine also writes the zero-frequency photocurrent `pcr_/pci_<abc>.dat` (linear and circular polarisation) and its split by channel `pc{r,i}_<channel>_<abc>.dat`. For a metal the coupled kernels default to `none, hartree, hartree+alda` (no bootstrap). For 3D crystals with internal projections the prompts also offer the coupled optical response: `run_nonlinear()` then calls `coupled_optical_response(...)` with `COUPLED_NK`, `COUPLED_KERNELS`, `COUPLED_EMIN/EMAX/NE`, `COUPLED_SHG_ENERGIES` (empty skips the coupled SHG), `COUPLED_COMPONENTS`, `COUPLED_ETA` and `COUPLED_LRC_ALPHA`. The prompts (`collect_nonlinear`) ask for the system type (crystal, 2D from the common question, molecule), the photon window, the broadening, the scissor, the components, the responses, the dipoles, the SHG terms, the molecular coupling and the coupled-response block. `plot.py` gets "Shift current", "Injection current" and "SHG susceptibility" (`GPAO.plot_nonlinear`, units of the run, 3D or sheet), and when they apply "SHG by term" (`GPAO.plot_shg_terms`), "Photocurrent of a metal by channel" (`GPAO.plot_photocurrent_channels`), "Dielectric function at the kernel levels" and "SHG at the kernel levels" (`GPAO.plot_coupled_response`) and "Molecular hyperpolarizability" (`GPAO.plot_molecular_beta`). See [Non-linear optics](nonlinear-optics.md) and `examples/nlo_examples/`.

**Workflow C — Harmonic phonons** (`main.phonon.py`): finite-displacement phonons via phonopy.

**Workflow D — Electron–phonon** (`paoflow-gen`, workflow `elphon`): writes a *system folder* in the layout of `examples/elphon/<System>` instead of a single script (see [Electron–Phonon Coupling](Electron-Phonon-Coupling.md), "The examples and the `paoflow-gen elphon` CLI workflow"):
- **Prompts:** a `pw.x` input of the system (default: `scf.in` or `*.scf.in` in the working directory), the prefix and a display name, the `pseudo_dir` of the run directories, the coupling source (`dfpt`, PAOFLOW's native vertex from the `ph.x` `dvscf` files, the default; `epw`, EPW `epbwrite`, for ultrasoft/PAW pseudopotentials; `fd`, finite differences of an ideal supercell, see below), the coarse k- and q-grids (re-prompted until q divides k), the nscf sampling (irreducible wedge, unfolded by symmetry, or the explicit full list that EPW needs), `nbnd`, the bands to exclude (EPW source), masses (default from `ATOMIC_SPECIES`), `nelec`, whether the system is a metal, the basis directory and `pthr`, then a **multi-select of workflows** (`phonons`, `coupling`, `superconductivity`, `optics`, `transport`, `arpes`, `thermal`; default `superconductivity, transport, arpes` for metals and `optics, transport, arpes` for semiconductors) and the settings of each: dense grids (re-prompted until `nk % nq == 0`), smearing, μ*, the Migdal–Eliashberg temperatures (`Tmin, Tmax, nstemp`), the transport Eliashberg function and the Fermi-surface coupling (superconductivity); photon energies, temperatures, `degaussw`, `fsthick`, the refractive index, the position dipoles, an eACBN0 electrons save or a scissor (target gap, reference save, Varshni), the Poncé screening correction and the slab emissivity with its extended-basis dielectric function (optics); temperatures, doping, window, smearing, electrons save, scissor and screening (mobility of a semiconductor) or temperatures, residual resistivity, Fermi-liquid coefficient and the interband dielectric function (resistivity of a metal); photon energy, work function and the k-path (ARPES); temperatures, the ZG supercell, its k-grid, the bands per atom and the number of configurations per temperature (thermal: the special displacement method, see [Special displacement method](special-displacement.md)).
- **Output:** `system.py` (the only file with values: an `ElphonSystem` plus the `SUPERCONDUCTIVITY`, `OPTICS`, `TRANSPORT`, `ARPES`, `THERMAL`, `PHONONS`, `COUPLING` default dictionaries), `qe/phonon/{scf.in, ph.in}` (`ph.in` with `fildvscf` and, for non-metals, `epsil = .true.`), `qe/nscf/nscf.in` (plus `epw.in` and `write_ukk.py` for the EPW source), `qe/optics/nscf.in` when the dielectric function is wanted, `qe/run_qe.sh` (the QE steps in order), `qe/kmesh.py` (coarse-mesh editor) and the selected workflow folders, each holding copies of the drivers of `src/PAOFLOW/gen/elphon_templates/` — the same files as the examples. Without a `pw.x` input only `qe/nscf/nscf.kpoints` and `qe/phonon/ph.in` are written. `-o/--out` is the system folder (default: the working directory); existing files are kept unless `--force`.
- **Running:** `cd qe && NP=8 bash run_qe.sh`, `paoflow-genbasis-ps --pseudo-dir qe --out BASIS_PS --preset standard`, then `mpirun -np N python <workflow>/<driver>.py`; every driver reads `system.py` and writes `<workflow>/output/`, the coarse vertex is cached in `cache/` and shared. The READMEs of `examples/elphon/{Si,Pb,MgB2,SiC}` document each workflow.
- **Finite-difference source** (`fd`):
  - Questions: the supercell `n` (the coarse k-grid must be a multiple of it; the supercell runs use k/n), `cartesian` (6N cells, the default up to 300 runs) or `random` displacements (`M` configurations, default 3N, with an optional locality cutoff and a seed), and the amplitude (0.06 bohr). The q-grid, `tr2_ph`, the anisotropic Fermi-surface coupling and the dense q-grid are not asked: q is Γ of the supercell.
  - Workflows: superconductivity, and transport for metals (the resistivity reads α²F_tr). Other workflows are refused, because the other dense drivers zero every mode at Γ. An insulator gets none and a note.
  - Output: `qe/phonon/scf.in` and a `run_qe.sh` with the scf only, `qe/finite_differences.py` (`--write`, `--status`), and `system.py` with `coupling_source='fd'`, `fd_dir='qe/fd'`, `save_dir` the reference save `qe/fd/tmp_<prefix>_sc_ref/<prefix>_sc_ref.save`, `prefix` `<prefix>_sc_ref`, the supercell k-grid, `qgrid=(1, 1, 1)`, `nelec` times n³, `long_range=False`, `fd_options`, plus a `FINITE_DIFFERENCES` dictionary (supercell, mode, distance, nconfig, seed, nbnd, primitive save, pseudo_dir, prefix).
  - Plan: scf of the primitive cell, `python qe/finite_differences.py --write`, `bash run_fd.sh` in `qe/fd` (QE launch variables, finished runs skipped), the basis, then `eliashberg.py` on one rank.
  - `--write` calls `PAOFLOW.elphon.workflows.finite_differences.write_supercell_displacements`. It builds the n×n×n supercell from the primitive save and treats it as its own unit cell (`supercell_matrix = 1`). It then writes the reference and displaced inputs with the `electron_phonon` writers, without an extra scf of the ideal supercell.

**Workflow E — Anharmonic phonons (SSCHA)** (`main.sscha.py`, `PAOFLOW.gen.sscha_driver`): the stochastic self-consistent harmonic approximation at a list of temperatures (see [Anharmonic phonons at temperature](sscha.md)).
- **Prompts:** the scf save and prefix, the pseudopotential directory, the temperatures, the supercell, the `ph.x` matrices of the harmonic start (a `<prefix>.dyn0` found under the working directory is proposed; otherwise the `harmonic` phase writes and runs `ph.in`) and the q-grid of that run (a multiple of the supercell), the k-grid of the supercell force runs (blank: the scf grid divided by the supercell), the electrons (`parent` occupations, `ionic` = Fermi–Dirac at each temperature, `fixed` = Fermi–Dirac at one temperature), the configurations per population and of the Hessian population, the launcher (`local`, `slurm`, `manual`) with the `pw.x` command, and for Slurm the partition, nodes, tasks and wall time of one configuration, the account, the cores and wall time of the step job, the concurrency of the arrays, the environment line of the jobs and their python.
- **Output:** `main.sscha.py` (default name; `-o` to change, `--force` to overwrite), every setting an editable constant (`TEMPERATURES`, `SUPERCELL`, `KGRID`, `ELECTRONIC_TEMPERATURE`, `POPULATION_SIZES`, `HESSIAN_SIZE`, `START_FROM`, `HARMONIC_DYN`, `HARMONIC_QGRID`, `FINE_GRID`, `LAUNCHER`, `MPI_QE`, `QE_PW`, `QE_PH`, `SLURM_*`, `ENV_SETUP`, `PYTHON`, ...).
- **Running:** `python main.sscha.py harmonic` (only without harmonic matrices), `python main.sscha.py run` (all temperatures, or `--temperature T`), `python main.sscha.py status`. `local` runs pw.x to the exported force constants; `slurm` submits one job array per population and a step job (afterany) that minimises and submits the next one, to the end; `manual` writes the inputs and stops until `run` is called again. The state of each temperature is in `output/sscha_T<T>/state.json`, so every phase is resumable; `run` resubmits the missing configurations of an incomplete population. Non-interactive generation: `build_sscha_script(cfg)` (used by `agentic/sscha/templates/campaign_template.py`).

**Workflow F — Quantum oscillations (pyskeaf)** (`main.pyskeaf.py`, `PAOFLOW.gen.pyskeaf_driver`, questions in
`PAOFLOW.gen.questions.pyskeaf`): de Haas–van Alphen frequencies from the single-band
`Fermi_surf_band_<n>.bxsf` files that `PAOFLOW.fermi_surface()` writes (workflow B with the Fermi surface).
- **Prompts:**
  - the folder of the BXSF files (default: `output`, or the first `*/output` that has them; a note lists the
    bands found, or says that none exists yet);
  - the bands (`all` or numbers) and the energy (one value or a scan, eV from E_F);
  - the interpolation per reciprocal-cell side;
  - the field direction:
    - `rotation`: θ and φ at the start and the end, and the number of directions (default φ 0→90°, 37
      directions);
    - `non_principal`: one θ, φ;
    - `b1`, `b2`, `b3`: a reciprocal vector;

    θ is the azimuth in the xy plane and φ the polar angle from +z, the `azimuthal` and `polar` of
    `PAOFLOW.pyskeaf`;
  - the minimum frequency, the frequency and orbit tolerances, the wall orbits and the verbose files.
- **Output:** `main.pyskeaf.py`. It initialises PAOFLOW with the output folder only (no QE save) and calls
  `pyskeaf(...)`, which writes `qo_EF_<energy>_freqvsangle_<band>.out` there.
- **Running:** `mpirun -np N python main.pyskeaf.py`; the field directions are spread over the ranks.
  `plot.pyskeaf.py` (asked, default yes) saves the dHvA figures as PDF files in `<folder>/figures`.

**Workflow G — Property run from a TB model** (`main.py` and `plot.py`, `PAOFLOW.gen.tb_model_driver`,
questions in `PAOFLOW.gen.questions.tb_model`): the properties of workflow B on a tight-binding model of
`PAOFLOW.models.models`. There is no DFT run, so no save, projection or basis; the parameters of the model come
instead. Slater–Koster models come later.
- **Models** (`TB_MODELS`): `honeycomb` (gapped graphene, Haldane; 2D), `kane_mele` (spin–orbit and Rashba
  honeycomb; 2D, spinful), `tmd` (MX₂, three d bands; 2D), `rice_mele` (1D chain), `weyl` (two-band Weyl
  semimetal; 3D, spinful), `parabolic` (2, 3 or 4 bands: chain, square, cubic).
- **Parameters:** one question per keyword argument of the model function, from its signature (kind and
  default) and its docstring (prompt, unit, Studio help), so new arguments in `models.py` appear by themselves.
  `SPECIAL_PARAMETERS` turns the TMD material and functional and the number of parabolic bands into choices.
  A note gives the orbitals, the spin, the filling, the dimension and the band range of the model, or the error
  of the model function.
- **Properties:** the menu of workflow B without what needs a PAO basis (orbital Hall, emissivity). Spin texture,
  spin Hall and anomalous Hall are offered for spinful models only, because the Hall routines require
  spin–orbit. Boltzmann transport is offered for 3D models only, because it inverts the 3×3 conductivity, which
  is singular along a direction without hoppings.
- **Defaults from the model:**
  - the band path from its lattice: Γ-X for a chain; Γ-M-K-Γ for a hexagonal 2D lattice, with K computed as a
    corner of its zone; Γ-X-M-Γ for other 2D lattices; Γ-X-M-Γ-R-X|M-R in 3D;
  - the energy window from its band range;
  - the dense grid per periodic direction (400, 96 or 32; 1 along the others), and the non-linear grid
    (2000, 200 or 40).

  The non-linear block leaves out the molecule, the Löwdin dipoles and the coupled response.
- **Output:** `main.py` imports the model function and builds `PAOFLOW.PAOFLOW(model=<function>(**MODEL_PARAMETERS))`
  in `new_paoflow()`. `run_properties()`, `run_optical()` (`gradient_and_momenta()` and `dielectric_tensor`) and
  `run_nonlinear()` each start their own instance. The property calls are those of workflow B
  (`_emit_standard_properties`, `_nonlinear_call_lines`), `IBRAV = 0` with the explicit band path, and `NFFT`
  and `NLO_NK` have 1 along the directions without hoppings. `plot.py` is the one of workflow B.
- **Running:** `python main.py` or `mpirun -np N python main.py`. The plan has one MPI step plus the plot step.

**Plot scripts.** `outputs.PLOT_DEFAULTS` records which plot scripts each workflow writes by default:
- the Property run, phonons and pyskeaf ask "Also generate a plotting script?" (default yes);
- ACBN0 and the electron–phonon folder write theirs;
- SSCHA writes `plot.sscha.py` only with `--plot`.

`cfg['plot'] = False` drops them, including the `plot_*.py` of an electron–phonon folder and its Figures
steps. `plot.sscha.py` and `plot.pyskeaf.py` (`gen/results_plot.py`) render the datasets of the Results layer
with matplotlib and save them, with no display needed. The Studio hides the plot questions and shows one
*Write the plot scripts* switch per workflow, which starts from these defaults.

## A Complete Workflow

Here's what a typical end-to-end session looks like:

```bash
# Generate a QE input file from an AFLOW entry
paoflow-gen-qe --pseudo /path/to/pseudos aflow:0a66d228d896a855

# Run the QE SCF calculation
pw.x < scf.in > scf.out

# Generate the PAOFLOW driver script interactively
paoflow-gen

# Run the property calculation
python main.py

# Visualize the results
python plot.py
```
