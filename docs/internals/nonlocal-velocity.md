# Non-local Velocity Correction

## The Problem

PAOFLOW builds the velocity/momentum operator from the k-derivative of the PAO Hamiltonian. For non-local pseudopotentials, this misses a gauge-invariant correction term — the commutator [V_NL, r] — that the PAO projection doesn't capture. The effect is noticeable: without the correction, copper's d→s absorption peak comes out roughly 2× too small compared to Quantum ESPRESSO reference calculations.

## The Fix

A correction term Δp is added to the velocity operator before band-resolved momentum matrix elements are computed. For Kleinman-Bylander pseudopotentials, Δp is assembled from projector-PAO overlaps and screened interaction coefficients, built in k-space via Fourier sums of real-space two-center integrals.

## Turning It On

```python
pf.gradient_and_momenta(nonlocal_velocity=True)
```

Or equivalently, set `attr['nonlocal_velocity'] = True` before calling `gradient_and_momenta`.

## How It Was Built

The correction was implemented incrementally across several phases:

| Phase | What was added |
|-------|----------------|
| 1–4 | Scalar (spin-less) path for norm-conserving pseudopotentials. Validated on Cu and Si — restores cubic isotropy and matches QE `epsilon.x` peak heights. |
| 4.5 | Ad-hoc spin-orbit support via block-diagonal tiling of the scalar correction. |
| E | Fully-relativistic UPF path building Δp directly in the (j, m_j) coupled basis. Eliminates basis-mismatch artifacts and achieves machine-precision cubic isotropy on Pt. |
| F | Performance optimization: per-pair memoization of expensive radial integrals and spherical harmonics. About a 19× speedup — from 161.5 s down to 8.6 s on a 1728-point k-grid. |

## Design Decisions Worth Knowing

**Covariance alignment.** PAOFLOW's tesseral Y_lm convention includes the Condon–Shortley phase, which differs from some standard conventions. The correction applies a sign-flip post-multiply for odd magnetic quantum numbers to stay consistent.

**Injection sign.** +1 for scalar/ad-hoc-SO paths; −1 for fully-relativistic jm-kspace. This is resolved automatically per path — you don't need to set it manually.

**Energy-diagonal is preserved.** By Hellmann–Feynman, group velocity stays exact. The correction only redistributes interband oscillator strength, not the diagonal energies.

## What It Affects

| Property | Effect of correction |
|----------|---------------------|
| Dielectric function, EELS, optical conductivity, AHC, SHC | High sensitivity — the correction matters here |
| Band structure, transport (Boltzmann), effective mass | No effect — group velocity is protected by Hellmann–Feynman |
| Fermi-sea geometric quantities (SHC, AHC) | Minor — high-energy redistribution gets suppressed by energy-denominator weighting |

## Caveats

- Norm-conserving pseudopotentials only for now. USPP/PAW augmentation is a separate piece of work.
- Fully-relativistic paths need a calibrated injection sign per path — a mismatch produces Hellmann–Feynman violations.

## Validation

- **Cu FCC:** ε₂ peak ≈ 32 eV near 1.7 eV with the correction — matches QE.
- **Pt (fully-relativistic):** cubic isotropy ε_xx = ε_yy = ε_zz to machine precision; SHC and EELS match QE references.
