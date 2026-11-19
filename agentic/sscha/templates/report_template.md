# <System>: anharmonic phonons and stability at temperature (SSCHA)

*Analysis of `<run folder>`, <date>. Every number comes from `tables/` (written by `analysis.py`) unless stated.*

> **Settings and caveats.** Supercell <n×n×n> (q-points <...>), k-grid <...>, electrons <parent occupations
> (type, width) / Fermi–Dirac at T>, populations <sizes>, Hessian population <size>, harmonic support <ph.x grid,
> electrons>; fixed cell at <a> Å (no thermal expansion); functional <...>; bubble-only Hessian
> (`include_v4=False`).

## Bottom line

| Question | Answer | Evidence | Main limitation |
|---|---|---|---|

## 1. Harmonic picture

<Softest harmonic modes (q, cm⁻¹) on the fine mesh, cold and hot electrons; the dispersion figure.>

## 2. Runs

| T (K) | electrons | populations (sizes) | converged | F ± σ (Ry) | Kong–Liu | softest Hessian, grid (cm⁻¹, q) | fine mesh (cm⁻¹, q) | verdict |
|---|---|---|---|---|---|---|---|---|

<Convergence figures; diagnostics that mattered and what was done.>

## 3. Stability temperature

<T_s with the bracket and the local fit (`tables/stability.md`); the scan figure; comparison with the experimental
reference as a bound or range, with what each number measures.>

## 4. Anharmonic dispersions

<Dispersions at the computed temperatures against the harmonic one; renormalisation of the soft branch.>

## 5. Downstream use

<Which runs fed the thermal cells / electron-phonon calculations (paths, settings).>

## 6. Limitations and next steps

<Fixed volume, supercell size, electrons, functional, stochastic error, competing phases; proposed runs with cost.>

## 7. Reproduce

```bash
python campaign.py        # drivers (already run)
python analysis.py && python plot_figures.py
```
