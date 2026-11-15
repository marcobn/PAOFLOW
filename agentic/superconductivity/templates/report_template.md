# <System>: <superconducting observable(s)> — PAOFLOW vs experiment

*Analysis of `<system folder>/superconductivity/<output>` (`me/`, `me_aniso/`), <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Provenance and definition caveat.** <Which numbers are measured / transcribed / vector-extracted / digitised /
> inverted (tunnelling α²F with its μ*). The definitions used: gap (order parameter / edge / peak), Tc (kernel / Δ² /
> onset / midpoint), λ (with which N_F). μ* = <run's value>; Tc also quoted at 0.10–0.13. Temperatures computed /
> interpolated. Grids, smearing (fixed or adaptive), N_F, equal grids for the anisotropic run, convergence flags, code
> version. Model limits: harmonic phonons, scalar-relativistic bands, Coulomb as μ* only.>

## Bottom line

| Observable | Agreement | Main discrepancy | Likely cause |
|---|---|---|---|
| <Tc, λ, ω_log, α²F shape, Δ(T), gap ratio, sheet gaps, tunnelling DOS> | <metric vs baseline> | <...> | <...> |

## 1. Data inventory

| Dataset (file) | Source | Method, definition | Form → method | Quantity, range | Uncertainty |
|---|---|---|---|---|---|

## 2. Model run

<`describe()` tables of the adapters; the convergence series available; how each observable is evaluated at the
experimental conditions.>

## 3. Results

<The Tc table (model by formula and μ*, measured, inverted λ); Δ(T) against the BCS baseline on both sides; the gap
ratios (model, measured, BCS, strong coupling); α²F per phonon branch with λ(ω); the sheet gaps of a multi-gap metal;
the figures.>

## 4. Diagnosis and next steps

<Ranked causes with evidence (convergence of N_F and λ first, then μ*, anisotropy, SOC); proposed runs as exact driver
commands with cost; out-of-scope physics.>

## 5. Reproduce

```bash
python digitize_sources.py && python analysis.py && python plot_figures.py
```
