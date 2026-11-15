# <System>: <observable(s)> — PAOFLOW vs experiment

*Analysis of `<model output>`, <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Provenance caveat.** <Which numbers are measured / transcribed / vector-extracted / digitised, from which
> figures; replots; digitisation accuracy; assumptions (axis readings, label inferences); model temperatures
> matched within <tol> K or interpolated (labelled).>

## Bottom line

| Observable | Agreement | Main discrepancy | Likely cause |
|---|---|---|---|
| <...> | <metric vs baseline> | <...> | <...> |

<Two or three sentences: what the model gets right without adjustment, and the independent systematic errors.>

## 1. Data inventory

| Dataset (file) | Source | Form → method | Quantity, range | Uncertainty |
|---|---|---|---|---|

<Decisions taken during ingestion, each with its evidence.>

## 2. Model run

<Driver, grids, approximations, energy grid, temperatures; how each observable is evaluated at the experimental
geometry; interpolation rules.>

## 3. Results

<One paragraph or table per comparison, with numbers, the baseline and the figure.>

## 4. Diagnosis and next steps

<Ranked causes with evidence; proposed runs with parameters, cost and what each settles; out-of-scope physics.>

## 5. Reproduce

```bash
python digitize_sources.py && python analysis.py && python plot_figures.py
```
