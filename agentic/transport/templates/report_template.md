# <System>: <transport observable(s)> — PAOFLOW vs experiment

*Analysis of `<system folder>/transport/<tag>` (and `superconductivity/output`), <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Provenance and correction caveat.** <Which numbers are measured / transcribed / vector-extracted / digitised,
> from which figures; replots; digitisation accuracy. The sample: doping, purity (ρ₀ / RRR), method (Hall or
> drift, four-probe). Corrections applied: Hall factor <value, source>; ρ₀ <value, source> added to the model;
> Matthiessen impurity mobility <value, source> as a diagnostic. Model temperatures matched within <tol> K or
> interpolated (law stated). The flags of every run were read from its log / metadata.>

## Bottom line

| Observable | Agreement | Main discrepancy | Likely cause |
|---|---|---|---|
| <μ_e(T), μ_h(T), ρ(T), λ_tr, ω_p, emissivity> | <metric vs baseline> | <...> | <...> |

<Two or three sentences: what the model gets right without adjustment, and the independent systematic errors.>

## 1. Data inventory

| Dataset (file) | Source | Sample, method | Form → method | Quantity, range | Uncertainty |
|---|---|---|---|---|---|

<Decisions taken during ingestion, each with its evidence.>

## 2. Model run

<`describe()` tables of the adapters: grids, smearing, doping, gap / scissor, long-range and screening flags,
electrons provenance; channels, λ, λ_tr, ω_p, Drude model. The convergence series available. How each observable
is evaluated at the experimental conditions; interpolation rules; corrections.>

## 3. Results

<One paragraph or table per comparison, with numbers, the baseline (published calculation, second measurement,
power-law / Bloch-Grüneisen fit, Matthiessen offset) and the figure. Exponents and slopes of experiment and model
over the same window.>

## 4. Diagnosis and next steps

<Ranked causes with evidence; proposed runs as exact driver commands with `--tag`, cost and what each settles;
out-of-scope physics (impurities, films, phonon drag, SOC).>

## 5. Reproduce

```bash
python digitize_sources.py && python analysis.py && python plot_figures.py
```
