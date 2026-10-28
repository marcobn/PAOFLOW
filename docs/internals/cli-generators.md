# CLI Input and Script Generators

Setting up a PAOFLOW calculation involves quite a bit of boilerplate — QE input files, driver scripts, plotting routines. These two command-line tools handle that for you, letting you focus on the science rather than the setup. Both rely primarily on Python's standard library and produce static, well-commented scripts that are easy to read and modify.

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

**Supported workflows** (chosen at the first prompt):

**Workflow A — ACBN0/eACBN0:** Self-consistent Hubbard U calculations, with or without intersite V terms. See [ACBN0 Module](acbn0.md) for details on the underlying implementation.

**Workflow B — Property runs:** Pick from band structure, DOS, transport, Fermi surface, spin texture, spin Hall conductivity, orbital Hall conductivity, anomalous Hall effects, topology, optical properties, and more. The generated `plot.py` includes only the visualization routines for the properties you chose.

The property `nonlinear` (non-linear optics) adds a separate `run_nonlinear()` to `main.py`: a PAOFLOW instance on the standard basis that calls `nonlinear_optics(nk=NLO_NK, emax=NLO_EMAX, ne=NLO_NE, eta=NLO_ETA, scissor=NLO_SCISSOR, components=NLO_COMPONENTS, position_dipoles=NLO_DIPOLES, symmetry=NLO_SYMMETRY)` right after `pao_hamiltonian()` (it needs `HRs`, so it never interpolates); `NLO_DIPOLES` (default `False`) switches on the Löwdin position dipoles and `NLO_SYMMETRY` (default `True`) the irreducible-wedge reduction. The prompts ask for the dense k-grid, the photon window, the broadening, the scissor and the tensor components (`all`, `diag` or e.g. `xyz,yyy`); `plot.py` gets the entries "Shift current", "Injection current" and "SHG susceptibility". See [Non-linear optics](nonlinear-optics.md) and `examples/nlo_examples/`.

**Workflow C — Harmonic phonons** (`main.phonon.py`): finite-displacement phonons via phonopy.

**Workflow D — Electron–phonon** (`paoflow-gen`, workflow `elphon`): writes a *system folder* in the layout of `examples/elphon/<System>` instead of a single script (see [Electron–Phonon Coupling](Electron-Phonon-Coupling.md), "The examples and the `paoflow-gen elphon` CLI workflow"):
- **Prompts:** a `pw.x` input of the system (default: `scf.in` or `*.scf.in` in the working directory), the prefix and a display name, the `pseudo_dir` of the run directories, the coupling source (`dfpt`, PAOFLOW's native vertex from the `ph.x` `dvscf` files, the default; `epw`, EPW `epbwrite`, for ultrasoft/PAW pseudopotentials), the coarse k- and q-grids (re-prompted until q divides k), the nscf sampling (irreducible wedge, unfolded by symmetry, or the explicit full list that EPW needs), `nbnd`, the bands to exclude (EPW source), masses (default from `ATOMIC_SPECIES`), `nelec`, whether the system is a metal, the basis directory and `pthr`, then a **multi-select of workflows** (`phonons`, `coupling`, `superconductivity`, `optics`, `transport`, `arpes`; default `superconductivity, transport, arpes` for metals and `optics, transport, arpes` for semiconductors) and the settings of each: dense grids (re-prompted until `nk % nq == 0`), smearing, μ*, the Migdal–Eliashberg temperatures (`Tmin, Tmax, nstemp`), the transport Eliashberg function and the Fermi-surface coupling (superconductivity); photon energies, temperatures, `degaussw`, `fsthick`, the refractive index, the position dipoles, an eACBN0 electrons save or a scissor (target gap, reference save, Varshni), the Poncé screening correction and the slab emissivity with its extended-basis dielectric function (optics); temperatures, doping, window, smearing, electrons save, scissor and screening (mobility of a semiconductor) or temperatures, residual resistivity, Fermi-liquid coefficient and the interband dielectric function (resistivity of a metal); photon energy, work function and the k-path (ARPES).
- **Output:** `system.py` (the only file with values: an `ElphonSystem` plus the `SUPERCONDUCTIVITY`, `OPTICS`, `TRANSPORT`, `ARPES`, `PHONONS`, `COUPLING` default dictionaries), `qe/phonon/{scf.in, ph.in}` (`ph.in` with `fildvscf` and, for non-metals, `epsil = .true.`), `qe/nscf/nscf.in` (plus `epw.in` and `write_ukk.py` for the EPW source), `qe/optics/nscf.in` when the dielectric function is wanted, `qe/run_qe.sh` (the QE steps in order), `qe/kmesh.py` (coarse-mesh editor) and the selected workflow folders, each holding copies of the drivers of `src/PAOFLOW/gen/elphon_templates/` — the same files as the examples. Without a `pw.x` input only `qe/nscf/nscf.kpoints` and `qe/phonon/ph.in` are written. `-o/--out` is the system folder (default: the working directory); existing files are kept unless `--force`.
- **Running:** `cd qe && NP=8 bash run_qe.sh`, `paoflow-genbasis-ps --pseudo-dir qe --out BASIS_PS --preset standard`, then `mpirun -np N python <workflow>/<driver>.py`; every driver reads `system.py` and writes `<workflow>/output/`, the coarse vertex is cached in `cache/` and shared. The READMEs of `examples/elphon/{Si,Pb,MgB2,SiC}` document each workflow.

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
