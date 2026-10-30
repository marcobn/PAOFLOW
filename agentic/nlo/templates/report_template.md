# <System>: <second-order response(s)> — PAOFLOW vs experiment

*Analysis of `<output folder>`, <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Conventions and run caveat.** <Which numbers are d or χ, at which photon energy, which component and
> orientation, relative to which reference crystal; 3D vs sheet with the thickness; provenance of every
> source. The run: nk, η, scissor (gap), components, degeneracy tolerance, basis and projectability, nosym
> or not; symmetry error bar <value> from `tables/symmetry.csv`; independent particles (no excitons, no
> local fields).>

## Symmetry audit

| kind | energy (eV) | forbidden present | leakage | equivalent-component spread | Kleinman |
|---|---|---|---|---|---|

## Bottom line

| Observable | Agreement | Main discrepancy | Likely cause |
|---|---|---|---|

## 1. Data inventory

| Dataset (file) | Source | Quantity, convention | Form → method | Range | Uncertainty |
|---|---|---|---|---|---|

## 2. Model run

<`describe()`; main.py settings; QE inputs; basis.>

## 3. Results

<Static values and peaks (percent deviations with the symmetry error bar), spectra per window, onsets vs the gap,
component ratios; the figures.>

## 4. Diagnosis and next steps

<Ranked causes with evidence; proposed runs as main.py settings with cost; out-of-scope physics.>

## 5. Reproduce

```bash
python digitize_sources.py && python analysis.py && python plot_figures.py
```
