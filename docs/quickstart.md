# Quickstart

This page walks you through running **example01** — Silicon bands, DOS, and transport — which ships with PAOFLOW and includes all the pre-computed QE data needed to run without a Quantum ESPRESSO installation.

---

## What you will do

1. Navigate to the example directory
2. Run a short Python script that builds the PAO Hamiltonian from pre-computed QE output
3. Inspect the output files for band structure, density of states, and transport tensors

**You do not need to run Quantum ESPRESSO yourself.** The `silicon.save/` directory in the example already contains the required `atomic_proj.xml` and `data-file-schema.xml` outputs.

---

## Prerequisites

PAOFLOW installed (see [Installation](installation.md)):

```bash
pip install PAOFLOW
```

---

## Step 1 — Navigate to example01

```bash
cd examples/qe_examples/example01
```

The directory contains:

```
example01/
├── silicon.save/          # Pre-computed QE output (atomic projections + data)
├── main.py                # PAOFLOW driver script
├── scf.in                 # QE SCF input (for reference only)
└── nscf.in                # QE NSCF input (for reference only)
```

---

## Step 2 — Examine the driver script

Open `main.py`. It shows the standard PAOFLOW Python API pattern:

```python
from PAOFLOW import PAOFLOW

def main():
    # Initialise PAOFLOW from a QE .save directory
    paoflow = PAOFLOW.PAOFLOW(
        savedir='silicon.save',
        outputdir='output',
        smearing='gauss',
        npool=1,
        verbose=True,
    )

    # Read pre-computed atomic projections from QE
    paoflow.read_atomic_proj_QE()

    # Drop bands with low projectability onto the PAO basis
    paoflow.projectability()

    # Construct the real-space Hamiltonian H(R)
    paoflow.pao_hamiltonian()

    # Interpolate onto the band path for ibrav=2 (FCC, e.g. Si)
    paoflow.bands(ibrav=2, nk=2000)

    # Double the Monkhorst-Pack grid by Fourier interpolation
    paoflow.interpolated_hamiltonian()

    # Diagonalise H(k) on the full BZ grid
    paoflow.pao_eigh()

    # Compute momentum matrix elements (needed for transport/optics)
    paoflow.gradient_and_momenta()

    # Apply adaptive Gaussian smearing for BZ integration
    paoflow.adaptive_smearing()

    # Compute DOS and transport tensors
    paoflow.dos(emin=-12., emax=2.2, ne=1000)
    paoflow.transport(emin=-12., emax=2.2)

    paoflow.finish_execution()

if __name__ == '__main__':
    main()
```

---

## Step 3 — Run PAOFLOW

```bash
python main.py
```

For a parallel run with MPI (optional, speeds up dense k-grids):

```bash
mpirun -np 4 python main.py
```

The run typically completes in under a minute on a laptop for this small example.

---

## Step 4 — Check the output

After the run, an `output/` directory is created containing:

| File | Contents |
|---|---|
| `bands_0.dat` | Band structure along the default FCC k-path |
| `dosdk_0.dat` | Total density of states |
| `sigmagauss_0.dat` | Electrical conductivity tensor vs. energy |
| `Seebeckgauss_0.dat` | Seebeck coefficient tensor vs. energy |
| `kappagauss_0.dat` | Electronic thermal conductivity tensor vs. energy |

!!! tip "Comparing to the reference"
    The `Reference/` subdirectory contains expected output files.
    You can diff your results against them to confirm a correct installation.

---

## Next steps

- Browse the other examples in [QE Examples](examples/qe-examples.md) for more complex workflows (SOC, transport, topology, ACBN0).
- Use the `paoflow-gen` command-line tool to generate a driver script for your own QE calculation:
  ```bash
  paoflow-gen
  ```
- See the [API Reference](api/paoflow.md) for the full list of methods on the `PAOFLOW` class.
