# Tutorial 01: Silicon Electronic Structure

## Introduction

In this tutorial you will run the complete PAOFLOW post-processing workflow on silicon. By the end you will understand:

- How Kohn–Sham wavefunctions are projected onto pseudo-atomic orbital (PAO) bases
- How a real-space PAO Hamiltonian $H(\mathbf{R})$ is constructed
- How Fourier interpolation produces smooth bands on arbitrarily dense k-grids
- How the density of states and transport tensors are computed from the interpolated eigenvalues

---

:::{note}
If users want to run PAOFLOW themselves, this tutorial can be downloaded as a notebook. Itassumes that a Quantum ESPRESSO `.save` directory is already available. Users may generate this themselves or download the tutorial assets from the [PAOFLOW Releases page](https://github.com/marcobn/PAOFLOW/releases).
:::

---

## Workflow Overview

Starting from the `.save` directory, PAOFLOW performs the following steps:

```text
silicon.save/
        │  atomic_proj.xml  ·  data-file-schema.xml
        ▼
Read atomic projections
        │  projection amplitudes ⟨φ_α|ψ_{n𝐤}⟩ and overlap matrix S_{αβ}(𝐤)
        ▼
Projectability filter
        │  select bands well-represented by the PAO basis
        ▼
PAO Hamiltonian  H(𝐑)
        │  real-space tight-binding matrix elements
        ▼
Band structure interpolation
        │  eigenvalues along Γ–X–W–K–Γ–L–U–W–L–K
        ▼
BZ integration
        │  interpolated grid · adaptive smearing · momentum matrix elements
        ▼
Density of states · transport tensors
```

---

## Step-by-Step PAOFLOW Workflow

### Initialisation

```python
from PAOFLOW import PAOFLOW

paoflow = PAOFLOW.PAOFLOW(
    savedir='silicon.save',
    outputdir='output',
    smearing='gauss',
    npool=1,
    verbose=True,
)
```

`PAOFLOW.PAOFLOW(...)` reads `data-file-schema.xml` for the crystal structure, k-point grid, and Kohn–Sham eigenvalues, and prepares all internal data structures. Output files are written to `outputdir`.

---

### Read Atomic Projections

```python
paoflow.read_atomic_proj_QE()
```

Parses `atomic_proj.xml`, loading the projection matrix $A_{n\alpha}(\mathbf{k}) = \langle\phi_\alpha|\psi_{n\mathbf{k}}\rangle$ and the overlap matrix $S_{\alpha\beta}(\mathbf{k})$ for all k-points in the grid.

---

### Projectability Filter

```python
paoflow.projectability()
```

For each Kohn–Sham band $n$ and k-point $\mathbf{k}$, computes the projectability

$$p_{n\mathbf{k}} = \sum_\alpha |A_{n\alpha}(\mathbf{k})|^2$$

which measures how completely that band is represented by the PAO basis. Bands with $p_{n\mathbf{k}}$ below the threshold (default 0.95) are excluded from the Hamiltonian construction. For silicon, the four valence bands and the four lowest conduction bands are well within the PAO window.

---

### PAO Hamiltonian Construction

```python
paoflow.pao_hamiltonian()
```

Constructs the k-space PAO Hamiltonian

$$\tilde{H}_{\alpha\beta}(\mathbf{k}) = \sum_n A^*_{n\alpha}(\mathbf{k})\,\varepsilon_{n\mathbf{k}}\,A_{n\beta}(\mathbf{k})$$

after Löwdin orthogonalisation of the PAO basis using $S(\mathbf{k})$, then Fourier-transforms it to real space:

$$H_{\alpha\beta}(\mathbf{R}) = \frac{1}{N_k}\sum_{\mathbf{k}} e^{-i\mathbf{k}\cdot\mathbf{R}}\,\tilde{H}_{\alpha\beta}(\mathbf{k})$$

The real-space Hamiltonian decays rapidly with $|\mathbf{R}|$: only a small number of Wigner–Seitz shells carry significant weight. This compact representation is the basis for all subsequent interpolation.

**Output:** `output/HRs.npy`, `output/R.npy`

---

### Band Structure Interpolation

```python
paoflow.bands(ibrav=2, nk=2000)
```

Evaluates $\tilde{H}(\mathbf{k}) = \sum_\mathbf{R} e^{i\mathbf{k}\cdot\mathbf{R}} H(\mathbf{R})$ at 2000 k-points along the FCC high-symmetry path ($\Gamma$–$X$–$W$–$K$–$\Gamma$–$L$–$U$–$W$–$L$–$K$) and diagonalises it. `ibrav=2` selects the FCC lattice convention.

Each k-point requires only a Fourier sum over real-space shells and an 8×8 diagonalisation, making interpolation on thousands of k-points essentially free.

**Output:** `output/bands_0.dat` — two-column file (k-path coordinate, energy in eV), one block per band.

:::{tip}
Plot the band structure with matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('output/bands_0.dat')
plt.plot(data[:, 0], data[:, 1], 'k-', lw=0.8)
plt.axhline(0, color='gray', ls='--', lw=0.5)
plt.xlabel('k')
plt.ylabel('Energy (eV)')
plt.show()
```

:::

---

### Interpolated BZ Grid

```python
paoflow.interpolated_hamiltonian()
paoflow.pao_eigh()
```

`interpolated_hamiltonian()` doubles the k-grid by Fourier interpolation (e.g. 12×12×12 → 24×24×24), providing a denser sampling of the Brillouin zone for integration.

`pao_eigh()` diagonalises $\tilde{H}(\mathbf{k})$ at every k-point on the interpolated grid, storing eigenvalues and eigenvectors for downstream calculations.

---

### Momentum Matrix Elements and Adaptive Smearing

```python
paoflow.gradient_and_momenta()
paoflow.adaptive_smearing()
```

`gradient_and_momenta()` computes the k-gradient $\nabla_\mathbf{k} H(\mathbf{k}) = i\sum_\mathbf{R}\mathbf{R}\,H(\mathbf{R})\,e^{i\mathbf{k}\cdot\mathbf{R}}$ and the momentum matrix elements $\mathbf{p}_{nm}(\mathbf{k})$, which enter the group velocities needed for transport.

`adaptive_smearing()` assigns each (band, k-point) a Gaussian smearing width proportional to the local band gradient $|\nabla_\mathbf{k}\varepsilon_{n\mathbf{k}}|$. This correctly broadens flat bands more than dispersive ones, improving the accuracy of BZ integrals without over-smearing sharp features.

---

### Density of States

```python
paoflow.dos(emin=-12., emax=2.2, ne=1000)
```

Evaluates

$$g(\varepsilon) = \frac{1}{N_k}\sum_{n,\mathbf{k}}\delta(\varepsilon - \varepsilon_{n\mathbf{k}})$$

broadened with the adaptive Gaussian widths, at 1000 energy points between −12 eV and +2.2 eV (spanning the full valence band and the lower conduction edge).

**Output:** `output/dosdk_0.dat` — two-column file (energy in eV, states/eV/cell).

---

### Transport Tensors

```python
paoflow.transport(emin=-12., emax=2.2)
paoflow.finish_execution()
```

Computes the electrical conductivity, Seebeck coefficient, and electronic thermal conductivity as functions of chemical potential within the Boltzmann transport framework:

$$\sigma_{\mu\nu}(E) = e^2\tau\frac{1}{N_k}\sum_{n,\mathbf{k}}v^\mu_{n\mathbf{k}}\,v^\nu_{n\mathbf{k}}\left(-\frac{\partial f}{\partial\varepsilon}\right)\bigg|_{\varepsilon_{n\mathbf{k}}}$$

where $v^\mu_{n\mathbf{k}} = \hbar^{-1}\partial\varepsilon_{n\mathbf{k}}/\partial k_\mu$ and $\tau$ is the relaxation time (assumed constant).

`finish_execution()` writes remaining output and frees internal resources.

**Output:**

| File                        | Contents                                    |
| --------------------------- | ------------------------------------------- |
| `output/sigmagauss_0.dat`   | Electrical conductivity $\sigma(E)$         |
| `output/Seebeckgauss_0.dat` | Seebeck coefficient $S(E)$                  |
| `output/kappagauss_0.dat`   | Electronic thermal conductivity $\kappa(E)$ |

---

## Complete Script

```python
from PAOFLOW import PAOFLOW

def main():
    paoflow = PAOFLOW.PAOFLOW(
        savedir='silicon.save',
        outputdir='output',
        smearing='gauss',
        npool=1,
        verbose=True,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.bands(ibrav=2, nk=2000)

    paoflow.interpolated_hamiltonian()
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(emin=-12., emax=2.2, ne=1000)
    paoflow.transport(emin=-12., emax=2.2)
    paoflow.finish_execution()

if __name__ == '__main__':
    main()
```

Run with:

```bash
python main.py
```

Or in parallel with MPI:

```bash
mpirun -np 4 python main.py
```

---

## Physics Background

### The PAO Basis

PAOFLOW uses **pseudo-atomic orbitals** (PAOs) as a localized basis. These are the bound-state solutions of the isolated-atom Kohn–Sham equation for each angular momentum channel included in the pseudopotential. For silicon with the $3s^2\,3p^2$ valence configuration, the PAO set consists of one $s$ orbital and three $p$ orbitals per atom — eight basis functions for the two-atom unit cell.

Projecting the extended Bloch wavefunctions $|\psi_{n\mathbf{k}}\rangle$ onto this localized set compresses the electronic structure information into a compact representation: the PAO Hamiltonian is an $8\times8$ matrix at each k-point rather than a matrix in the full plane-wave basis.

### Real-Space Representation and Interpolation

The Fourier transform of $\tilde{H}(\mathbf{k})$ onto real-space lattice vectors gives the tight-binding hopping integrals $H_{\alpha\beta}(\mathbf{R})$, analogous to hopping parameters in an empirical tight-binding model — but derived entirely from first principles. Because $H(\mathbf{R})$ decays exponentially with $|\mathbf{R}|$, a small number of shells is sufficient to recover the full band structure. Fourier-transforming back to any k-point is then exact within the PAO subspace, making dense k-grids (millions of points) computationally accessible after a single modest DFT calculation.

### Band Structure of Silicon

The interpolated band structure of silicon shows:

- **Valence band maximum at $\Gamma$** — formed from bonding $sp^3$ combinations
- **Conduction band minimum between $\Gamma$ and $X$** (the $\Delta$ point) — giving the indirect gap
- **PBE-GGA band gap of ~0.6 eV** — systematically below the experimental 1.1 eV due to the approximate exchange-correlation functional
- **Total valence bandwidth of ~12 eV** — reflecting the $s$–$p$ energy separation

### Density of States

The DOS $g(\varepsilon)$ reflects the same physics integrated over the full BZ. A gap separates the valence and conduction manifolds; Van Hove singularities appear as steps or peaks at band edges and saddle points. The total weight under the valence DOS equals the number of valence electrons per cell (eight for silicon).

### Boltzmann Transport

Within the constant relaxation time approximation, all transport tensors are proportional to $\tau$ and are reported per unit relaxation time. To obtain temperature-dependent conductivity, convolve $\sigma(E)$ with $-\partial f/\partial E$ at the desired temperature using the Fermi–Dirac distribution.

---

## Important Notes

:::{important}
`atomic_proj.xml` must contain an overlap matrix (`lwrite_overlaps = .true.` in the QE `&projwfc` namelist). Without it, `pao_hamiltonian()` will fail.
:::

:::{tip}
To verify the PAO representation quality, compare the interpolated band structure against the original DFT eigenvalues. Deviations larger than ~10 meV within the PAO window indicate a poorly conditioned Hamiltonian — typically caused by including bands with low projectability.
:::

:::{note}
The energy range `emin=-12.` to `emax=2.2` covers the full silicon valence band and the lower conduction band edge. Adjust `emax` upward if you need transport or DOS into the upper conduction bands.
:::

---

## Further Reading
