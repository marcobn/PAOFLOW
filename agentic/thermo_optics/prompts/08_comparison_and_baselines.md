# 08: Comparison metrics with baselines

**When:** model curves at the experimental conditions exist.

```
Compute tables/metrics.csv: one row per comparison x dataset x condition x physics window, with curve_deviation in the
planned space, a baseline (published theory / second measurement / trivial model) in the same window, and for
interpolated conditions the bias at the two bracketing model temperatures. Add the derived tables (onsets with
threshold_crossing, activation energies with arrhenius_fit, power-law exponents, slopes) for experiment, model and
reference. Report best_shift only as a diagnostic next to the unshifted rms. Write tables/summary.md.
```

**Check:** every model number has a baseline next to it; nothing is shifted or scaled without saying so.
