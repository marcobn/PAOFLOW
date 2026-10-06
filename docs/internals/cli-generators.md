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

**Workflow B — Property runs:** Pick from band structure, DOS, transport, Fermi surface, spin texture, spin Hall conductivity, anomalous Hall effects, topology, optical properties, and more. The generated `plot.py` includes only the visualization routines for the properties you chose.

**Workflow C — Harmonic phonons** (`main.phonon.py`): finite-displacement phonons via phonopy.

**Workflow D — Electron–phonon** (`main.elphon.py` + `me.elphon.py` + `plot.elphon.py`): Eliashberg α²F, λ and T_c from the PAO interpolation of the DFPT coupling, then the temperature-dependent superconducting gap from the isotropic Migdal–Eliashberg equations (see [Electron–Phonon Coupling](Electron-Phonon-Coupling.md)).
- **Coupling source:** EPW (`epbwrite`) is the default. The ph.x source `ahc` remains selectable as `ahc (Gamma only)`, with a warning that its vertices are gauge-inconsistent for q ≠ Γ. The patched-QE `elphmat` source is disabled in the menu (its code path is kept).
- **Prompts for the EPW source:** a `pw.x` input of the system (default: `scf.in` or `*.scf.in` in the working directory) and the `pseudo_dir` for the run directories; the coarse k and q grids (re-prompted until q divides k); `nbnd`, the bands to exclude (EPW syntax, e.g. `1:5`), masses (default from `ATOMIC_SPECIES`) and `nelec`; dense-q (on by default) with `NQ_DENSE` and `NK_DENSE` (re-prompted until `NK_DENSE % NQ_DENSE == 0`); smearing in eV; μ*; `pthr`; the Migdal–Eliashberg temperatures (`Tmin, Tmax, nstemp`, or `auto`: 25 temperatures up to 1.5 × the Allen–Dynes T_c of the computed α²F).
- **QE/EPW inputs**, in the layout of `examples/elphon_epw_example`: `phonon/scf.in` and `phonon/ph.in` (scf + ph.x on the irreducible q, `fildvscf`), `epw/nscf.in` (nscf on the explicit full k list), `epw/epw.in` (`wannierize = .false.`, no Wannier functions, `dvscf_dir = '../phonon/save'`) and `epw/write_ukk.py`, which writes the placeholder `<prefix>.ukk` carrying the band exclusion plus empty `<prefix>.bvec`/`<prefix>.mmn` stubs so `epw.x` finishes cleanly. The scf and the nscf run in separate directories; copy `charge-density.dat` and `data-file-schema.xml` from `phonon/<prefix>.save` into `epw/<prefix>.save` before the nscf. This route is validated on Pb (identical results to a Wannierized EPW run). An EPW input with a Wannierization and `epbwrite = .true.` also works.
- **`mpirun -np N python main.elphon.py`** builds the PAO electronic structure on the EPW nscf save (`epw/<prefix>.save`) and runs the dense-q (or, with `--coarse-q`, coarse-q) Eliashberg calculation on EPW's coupling.
- **`python me.elphon.py`** (all coupling sources, serial, seconds) solves the isotropic Migdal–Eliashberg equations on `output/eliashberg.npz`: the gap and Z on the imaginary axis, their Padé and analytic continuations to the real axis, Δ(T) and the linearised-kernel T_c. It writes `output/me/` in EPW's formats. `--a2f epw/<prefix>.a2f` runs it on EPW's α²F instead. μ* comes from the generator; `WSCUT`, `DEGAUSSQ`, `DEGAUSSQ_LINEAR` and `NPADE` are EPW-default constants at the top of the script and are also command-line options.
- **`plot.elphon.py`** plots α²F and the cumulative λ, and overlays EPW's own `<prefix>.a2f` (dashed) when it exists. After `me.elphon.py` it also draws the Migdal–Eliashberg figure (`output/me/migdal_eliashberg.png`).

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
