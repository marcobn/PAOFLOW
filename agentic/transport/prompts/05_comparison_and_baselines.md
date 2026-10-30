# 05: Comparison metrics with baselines

**When:** model curves at the experimental conditions exist.

```
Compute tables/metrics.csv: one row per comparison x dataset x condition x window, with curve_deviation in the
planned space, a baseline in the same window (published calculation such as EPW values, second measurement,
fit_power_law / fit_bloch_gruneisen / constant), and for interpolated conditions the bias at the two bracketing
model temperatures. Add the derived tables: power-law exponents (fit_power_law, power_law_exponent) of experiment
and model over the same window, slopes drho/dT (fit_linear, ResistivityModel.slope), the Bloch-Gruneisen
parameters of the experiment (rho0, A, theta), the Matthiessen offset (matthiessen_offset: rho0, rms before/after,
slope ratio), RRR, lambda_tr and omega_p against published values. Report corrections (Hall factor, rho0,
impurity mobility) next to the uncorrected deviation. Write tables/summary.md.
```

**Check:** every model number has a baseline next to it; nothing is corrected or shifted without saying so.
