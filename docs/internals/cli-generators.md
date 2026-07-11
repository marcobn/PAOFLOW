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

**Two supported workflows:**

**Workflow A — ACBN0/eACBN0:** Self-consistent Hubbard U calculations, with or without intersite V terms. See [ACBN0 Module](acbn0.md) for details on the underlying implementation.

**Workflow B — Property runs:** Pick from band structure, DOS, transport, Fermi surface, spin texture, spin Hall conductivity, anomalous Hall effects, topology, optical properties, and more. The generated `plot.py` includes only the visualization routines for the properties you chose.

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
