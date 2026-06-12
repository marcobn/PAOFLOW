# QE Examples

The `examples/qe_examples/` directory contains 16 Quantum ESPRESSO-based workflows. Each example is self-contained: it ships with pre-computed QE output so you can run PAOFLOW without running QE yourself.

---

## How to run

```bash
cd examples/qe_examples/example01
python main.py
```

Each example also accepts the legacy XML-input style for PAOFLOW v1 compatibility:

```bash
python ../main.py ./ inputfile.xml
```

---

## Example list

| # | Material | Highlights |
|---|---|---|
| 01 | **Si** (silicon) | Bands, DOS, transport with `spd` pseudopotential — the recommended starting point |
| 02 | **Al** | Bands and DOS with `spd` pseudopotential |
| 03 | **Pt** | Collinear spin-polarised calculation (`nspin = 2`, LSDA) |
| 04 | **Fe** | Non-collinear + spin–orbit coupling (SOC); anomalous Hall conductivity |
| 05 | **Pt** | Non-collinear + SOC; spin Hall conductivity |
| 06 | **AlP** | Non-self-consistent ACBN0 Hubbard U correction |
| 07 | **Al** | Starting from atomic projections with overlap matrix (prior to orthogonalisation) |
| 08 | **SnTe** (2D) | 2D material; spin Hall conductivity |
| 09 | **MoP₂** | Weyl point search using Z2Pack interface |
| 10 | **GaAs** | Boltzmann transport with temperature- and energy-dependent τ model |
| 11 | **Te** (left-handed) | Rashba–Edelstein tensor elements |
| 12 | **Bi bilayer** | Nanoribbon; site-projected band structure |
| 13 | **Bi bilayer** | Z₂ topological invariant via Z2Pack; SOC from QE and `ad_hoc_SOC` |
| 14 | **Pt** | Layer-resolved spin Hall conductivity |
| 15 | *(see directory)* | *(description not yet in README — check `example15/` for details)* |
| 16 | *(see directory)* | *(description not yet in README — check `example16/` for details)* |

---

## Recommended starting point

**example01 (Silicon)** is the simplest and best-documented example. It demonstrates the core PAOFLOW workflow:

1. Read pre-computed QE projections
2. Build the PAO Hamiltonian
3. Interpolate band structure
4. Compute DOS and Boltzmann transport tensors

See the [Quickstart](../quickstart.md) page for a step-by-step walkthrough of this example.

---

## What each example produces

All examples write output to an `output/` subdirectory. Typical output files include:

| File pattern | Contents |
|---|---|
| `bands_*.dat` | Interpolated band structure along a k-path |
| `dosdk_*.dat` | Total density of states |
| `pdosdk_*.dat` | Projected DOS per orbital/atom |
| `sigmagauss_*.dat` | Electrical conductivity tensor σ(E) |
| `Seebeckgauss_*.dat` | Seebeck coefficient S(E) |
| `kappagauss_*.dat` | Electronic thermal conductivity κ(E) |
| `ahcdk_*.dat` | Anomalous Hall conductivity (where applicable) |
| `shcdk_*.dat` | Spin Hall conductivity (where applicable) |

A `Reference/` subdirectory in each example contains the expected outputs for regression testing.

---

## Notes

- All examples assume a modern QE XML output format (`data-file-schema.xml`).
- Examples with SOC require `mpi4py` and typically benefit from parallel execution.
- The `npool` parameter in each `main.py` controls MPI k-point distribution; adjust it to match your core count.
