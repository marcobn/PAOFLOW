# <System>: <ARPES observable(s)> — PAOFLOW vs experiment

*Analysis of `<system folder>/arpes/<output>`, <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Provenance and protocol caveat.** <Which numbers are measured / transcribed / vector-extracted / digitised
> (band-map images: colour-scale assumption, pixel size in data units, calibration residuals); extracted Σ and λ carry
> the authors' bare band. The bare-band protocol used on both sides: <window / method>. Alignments: k − k_F with each
> side's own k_F; E_F offset <value> as a diagnostic; resolution <model = measured?>; impurity offset Im Σ(0) <value>.
> Model limits: Fan term only, no impurity / e-e Σ, scalar-relativistic bands, <Fan-Migdal q-grid and widths>.>

## Bottom line

| Observable | Agreement | Main discrepancy | Likely cause |
|---|---|---|---|
| <dispersion, λ (by method), kink energy, Re/Im Σ, MDC widths, linewidths vs T> | <metric vs baseline> | <...> | <...> |

## 1. Data inventory

| Dataset (file) | Source | Conditions (hν, pol., T, resolution, cut) | Form → method | Quantity, range | Uncertainty |
|---|---|---|---|---|---|

## 2. Model run

<`describe()`: cut and points, photon energy, polarisation, resolution, window, self-energies and temperatures,
Fan-Migdal parameters; how each observable is extracted (MDC method, windows); the protocol.>

## 3. Results

<Dispersion versus k − k_F per window; the λ table by method and side; kink energy against the α²F peaks; Σ curves
against the authors' and the tunnelling baseline; linewidths versus T with slopes; the figures.>

## 4. Diagnosis and next steps

<Ranked causes with evidence (the spread of λ with the protocol first); proposed runs as exact driver commands with
cost; out-of-scope physics.>

## 5. Reproduce

```bash
python digitize_sources.py && python analysis.py && python plot_figures.py
```
