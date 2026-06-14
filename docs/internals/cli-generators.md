# CLI Input and Script Generators

PAOFLOW provides two command-line tools that automate setup for computational studies. Both have minimal dependencies (primarily Python's standard library) and produce static, well-commented scripts that are easy to modify.

## `paoflow-gen-qe`

Converts AFLOW database entries into ready-to-run Quantum ESPRESSO `scf` input files with optimized defaults for smearing, magnetism, spin-orbit coupling, and band counts suitable for PAOFLOW's extended-basis projections.

**Accepted input formats:**
- AFLOWDATA URLs
- Material-page URLs
- Bare AUID tokens (e.g., `aflow:0a66d228d896a855`)

**Key features:**
- Automatic lattice-type detection (`ibrav` and `celldm` parameters)
- Intelligent cutoff selection from reference data
- Metal vs. insulator classification via band-gap analysis
- Automatic spin-polarization setup when needed
- Spin-orbit coupling support
- Band-count optimization for extended PAO basis calculations
- Recommended intersite-V cutoff calculation for follow-up U+V runs

**Main options:** pseudopotential directory (required), spin-orbit coupling flag, output path, smearing width, and symmetry tolerance.

## `paoflow-gen`

Generates PAOFLOW property-calculation driver scripts (`main.py`) from completed Quantum ESPRESSO runs, plus optional plotting scripts (`plot.py`) for visualizing selected properties.

**Two supported workflows:**

**Workflow A — ACBN0/eACBN0:** Self-consistent Hubbard U calculations (on-site only, or with intersite V terms). See [ACBN0 Module](acbn0.md) for module details.

**Workflow B — Property runs:** Selectable calculations including band structure, DOS, transport, Fermi surface, spin texture, spin Hall conductivity, anomalous Hall effects, topology, and optical properties.

Optionally creates a complementary `plot.py` that visualizes only the properties selected in `main.py`.

## End-to-End Example

```bash
# Generate QE input from AFLOW entry
paoflow-gen-qe --pseudo /path/to/pseudos aflow:0a66d228d896a855

# Run QE SCF calculation
pw.x < scf.in > scf.out

# Generate PAOFLOW driver
paoflow-gen

# Run property calculation
python main.py

# Visualize results
python plot.py
```
