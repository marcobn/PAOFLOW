# 04: Model at the experimental conditions

**When:** datasets are accepted; before any metric.

```
Open the model with EliashbergModel.from_system(<folder>), GapModel.from_system(<folder>, output='<output/me>') and/or
AnisotropicGapModel.from_system(<folder>, output='<output/me_aniso>') and evaluate it at every experimental condition:
Tc for the run's mu* and for 0.10 and 0.13 by every formula (tc(mu_star, formula), the ME Tc's); gaps with the measured
definition ('imag' / 'pade' / 'acon'; gap_peaks / sheet_gaps for multi-gap data with the band assignment stated) at
the measured temperatures (match_conditions; linear interpolation in T labelled); alpha^2F on the experimental omega
grid with the cumulative lambda; the quasiparticle DOS at the measurement temperature. Add the baselines as model
curves too: bcs_gap_vs_t scaled to the model's and to the measured Delta(0) and Tc, lambda_from_tc at the model's
omega_log and mu*. Write every curve to data/model_curves.csv (provenance 'calculated', mu*, definition and
interpolation in note).
```

**Check:** μ* and the definition are named for every model number; interpolated values are labelled; nothing is tuned.
