# Non-local Velocity Correction

## Problem

The velocity/momentum operator in PAOFLOW's PAO basis is built from the k-derivative of the PAO Hamiltonian. Non-local pseudopotentials introduce a gauge-invariant correction term — the commutator [V_NL, r] — that the PAO projection misses. Without this correction, optical absorption spectra are significantly underpredicted: copper's d→s peak is roughly 2× too small compared to Quantum ESPRESSO reference calculations.

## Solution

A correction term Δp is added to the velocity operator before band-resolved momentum matrix elements are computed. For Kleinman-Bylander pseudopotentials, Δp involves projector-PAO overlaps and screened interaction coefficients assembled in k-space via Fourier sums of real-space two-center integrals.

## Public API

Enable the correction via:

```python
pf.gradient_and_momenta(nonlocal_velocity=True)
```

Or set `attr['nonlocal_velocity'] = True` before calling `gradient_and_momenta`.

## Implementation Phases

The correction was implemented in successive phases:

| Phase | Scope |
|-------|-------|
| 1–4 | Scalar (spin-less) path for norm-conserving pseudopotentials. Validated on Cu and Si. Restores cubic isotropy and matches QE `epsilon.x` peak heights. |
| 4.5 | Ad-hoc spin-orbit support via block-diagonal tiling of the scalar correction. |
| E | Fully-relativistic UPF path building Δp directly in the (j, m_j) coupled basis. Eliminates basis-mismatch artifacts; achieves machine-precision cubic isotropy on Pt. |
| F | Performance optimization: per-pair memoization of expensive radial integrals and spherical harmonics. ~19× speedup (161.5 s → 8.6 s on a 1728-point k-grid). |

## Design Decisions

**Covariance alignment.** PAOFLOW's tesseral Y_lm convention (Condon–Shortley phase included) differs from standard conventions. The correction applies a sign-flip post-multiply for odd magnetic quantum numbers.

**Injection sign.** +1 for scalar/ad-hoc-SO paths; −1 for fully-relativistic jm-kspace. This is auto-resolved per path.

**Energy-diagonal preservation.** By Hellmann–Feynman, group velocity remains exact. The correction only redistributes interband oscillator strength.

## Scope

| Property | Effect |
|----------|--------|
| Dielectric function, EELS, optical conductivity, Hall effects (AHC/SHC) | High sensitivity — correction is important |
| Transport (Boltzmann), band structure, effective mass | No effect by design — group velocity is protected |
| Geometric Fermi-sea quantities (SHC, AHC) | Minor effect via high-energy oscillator-strength redistribution |

## Caveats

- Norm-conserving pseudopotentials only. USPP/PAW augmentation is deferred as separate work.
- Fully-relativistic paths require a calibrated per-path injection sign. Mismatched signs produce Hellmann–Feynman violations.
- The correction is a no-op for properties that depend only on group velocity.

## Validation

- **Cu FCC with NL correction:** ε₂ peak ≈ 32 eV near 1.7 eV — matches QE.
- **Pt fully-relativistic:** cubic isotropy ε_xx = ε_yy = ε_zz to machine precision; SHC and EELS match QE references.
