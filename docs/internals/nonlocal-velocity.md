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
(Since October 2026 this is a diagnostic option; production runs use
`gradient_and_momenta(position_dipoles=True)`, see the last section.)

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

## October 2026: Löwdin transformation, orbital signs, and what the term is

An orbital-basis comparison with the exact plane-wave commutator ⟨φ̃ᵢ|i[V_NL, r]|φ̃ⱼ⟩ between
PAOFLOW's own Löwdin orbitals (Si, `EPW_opt_audit/compare/nl_orbital_check.py`,
`nl_fixed_check.py`) found the assembled matrix exact between the *raw* atomic orbitals
(1e-3) but missing two steps into the basis PAOFLOW works in:

1. the Löwdin transformation S^{-1/2}(k) Δp S^{-1/2}(k) with the Bloch overlap of the orbitals
   (`lowdin_transform`; not a small correction for a diffuse basis: the largest elements were
   4–5× too small, cosine 0.37 → 0.999 with it);
2. the orbital signs (−1)^l (−1)^{odd |m|} (`pao_orbital_signs`; the odd-|m| factor alone
   flipped every s–p and p–d element).

Both are now applied in `build_nonlocal_velocity_kspace` (and, with the scalar→(j, m_j)
rotation of the overlap, in the jm path), fed from the overlap tables of
`position_dipoles.py` by `PAOFLOW.nonlocal_velocity_correction` and by the electron–phonon
wrapper. The operator agrees with the exact commutator to 2–3 % at every tested k.

**What the term is, and when to use it.** PAOFLOW's velocity is the commutator
v = i[H, r] with H the PAO Hamiltonian, which already contains the non-local pseudopotential.
Its only approximation is the position operator: with r reduced to the orbital centres
(`dHksp`), the missing part is i[H, d] with the position dipoles of the basis
(`position-dipoles.md`), not i[V_NL, r]. Measured against the exact plane-wave ε₂ of Si
(12³ grid, 21 bands, extended basis): plain 1.02–1.04, fixed non-local term 0.90–0.93,
dipoles 0.99–1.01, dipoles + non-local term 0.87 (double counting); against EPW's direct term
with the standard basis: 1.06, 0.97, 0.98, 0.88. The term as it was (1.09 / old pipeline
+7 % over the plain velocity) moved ε₂ the wrong way. Recommendation: do not add the term to the commutator
velocity at all — `POSITION_DIPOLES = True` / `gradient_and_momenta(position_dipoles=True)`
everywhere, `nonlocal_velocity` kept for diagnostics only. On Fe and Pt (`dipole_hall/`) the
repaired term still shifts the AHC/SHC at E_F by +10 % / −7 % and doubles the Fe AHC peak at
−1.35 eV, where the 1/Δε² weight of the Kubo formula amplifies the double counting at the
d-band near-degeneracies, while the dipoles change them by −1 % / +3.5 %. The Cu and Pt
`epsilon.x` comparisons above predate the fix and were obtained with the defective term; note
also that `epsilon.x` itself uses the local momentum only (no i[V_NL, r]): on Si it reproduces
the plane-wave local-only kernel to 2 % and lies 15–19 % above the exact spectrum
(`EPW_opt_audit/pao_dielectric/epsilon_x`), so it cannot serve as the reference for the
velocity operator.
