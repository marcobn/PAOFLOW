# 04: Model at the experimental conditions

**When:** datasets are accepted; before any metric.

```
Open the model with MobilityModel.from_system(<folder>, output='<tag>') and/or
ResistivityModel.from_system(<folder>, output='<tag>') and evaluate it at every experimental condition:
match temperatures with match_conditions(tolerance=<K>), interpolate where needed with the physical law
(power law for mu, activated for n, linear for rho above theta_D/5, power below; label it and keep the two
bracketing values); apply the stated corrections only (Hall factor <r_H, source>, rho0 <value, source> added to
the model, Matthiessen impurity mobility <mu_imp, source> as a diagnostic); use the measured observable
(drift vs Hall, sigma only at matched doping). Write every curve to data/model_curves.csv (provenance
'calculated', interpolation and corrections in note) and list the condition pairs with |dT|.
```

**Check:** no silent use of a model temperature beyond the tolerance; every correction is named with its source; interpolated and corrected curves are labelled.
