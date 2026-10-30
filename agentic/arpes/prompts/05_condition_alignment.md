# 05: Model at the experimental conditions

**When:** datasets are accepted; before any metric.

```
Open the model with ArpesModel.from_system(<folder>, output='<output>') and align it to every measured condition: the
same cut and segment (or say what differs), k - k_F with each side's own k_F (fermi_momentum), the same MDC method and
windows, the model resolution equal to the measured one (if not: propose the rerun of arpes.py with
ARPES['resolution'] set, seconds for the isotropic case, after renaming output/), the temperature (recompute the
isotropic Sigma at the measured T from eliashberg.npz with eliashberg_self_energy if the run lacks it), the E_F offset as a
diagnostic (best_shift, stated). Write every model curve to data/model_curves.csv (provenance 'calculated', alignment
and protocol in note).
```

**Check:** no hand-shifted curves; the resolution and the method match; interpolations and offsets are labelled.
