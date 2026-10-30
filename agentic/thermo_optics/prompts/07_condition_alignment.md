# 07: Model at the experimental conditions

**When:** datasets are accepted; before any metric.

```
Write (or extend) the model adapter for <observable> (templates/model_adapter_template.py; for PAOFLOW optics use
OpticsModel) and evaluate the model at every experimental condition: recompute the observable at the measured geometry
(<thickness, angle, polarisation, resolution>), match temperatures with match_conditions(tolerance=<K>), interpolate
where needed (ln y vs 1/T for activated quantities, linear otherwise) and keep the two bracketing curves.
Write every curve used to data/model_curves.csv (provenance 'calculated', interpolation in note) and list the
condition pairs with |dT|.
```

**Check:** no silent use of a model temperature farther than the tolerance; interpolated curves are labelled.
