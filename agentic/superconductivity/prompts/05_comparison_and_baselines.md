# 05: Comparison metrics with baselines

**When:** model curves at the experimental conditions exist.

```
Compute tables/tc.csv (model Tc by formula and mu* versus the measured Tc with its definition, the Tc-inverted lambda
next to the ab initio lambda) and tables/metrics.csv: one row per comparison x dataset x condition x window, with
curve_deviation in the planned space (Delta(T) in meV against the BCS baseline scaled to each side; alpha^2F per
phonon branch with the cumulative lambda at the window edges; the tunnelling DOS; gap peaks per sheet) and a baseline
in the same window (EPW on the same grids, BCS, strong_coupling_gap_ratio, second measurement). Add the derived
tables: gap ratios (model by definition, measured, BCS, strong-coupling), lambda and N_F by method, lambda_nk
statistics and sheet gaps. Write tables/summary.md.
```

**Check:** every model number has a baseline next to it; the Tc table shows the μ* sensitivity; nothing is tuned without saying so.
