# 06: Comparison metrics with baselines

**When:** model curves at the experimental conditions exist.

```
Compute tables/metrics.csv: one row per comparison x dataset x condition x window (below / above the kink), with
curve_deviation in meV (dispersion vs k - k_F, Re Sigma, Im Sigma - Im Sigma(0), MDC widths) and a baseline in the same
window (the authors' analysis, the tunnelling-alpha^2F Sigma, the linear bare band, a second measurement). Add the
derived tables: lambda by method and side (tables/lambda.csv), kink energies against the alpha^2F peaks, v_F and k_F,
dGamma/dT of the linewidths for the same state. Report best_shift in E only as a diagnostic next to the unshifted rms.
Write tables/summary.md.
```

**Check:** every model number has a baseline next to it; λ values carry their method; nothing is shifted without saying so.
