# 03: Preflight

```
Run preflight_pw_input on every input of the plan (displaced=..., metallic=..., interpolated=..., qgrid=..., cores=...,
billing_granularity=...). For each finding give the evidence and the fix (reduce_cell_input for FFT or k-point
inflation, >= 3 points per axis for interpolated Hamiltonians, damped mixing for large metallic cells, a coupling grid
the q-grid divides, a pool layout that uses the billed cores). Dry-run the scripts that will write and submit the runs
and say what they would do. Record the table in PLAN.md. Submit nothing.
```

**Check:** every input appears in the preflight table with its findings and fix (or the reason it is accepted).
