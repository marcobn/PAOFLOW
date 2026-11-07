# 08: Downstream

```
Hand the SSCHA force constants of <run(s)> to <thermal cells / electron-phonon>:
- thermal cells: sscha_force_constants('auxiliary') then special_displacement([T], force_constants_file='sscha');
- electron-phonon: ElphonSystem(..., phonon_source='sscha', sscha_dynamical_matrices='<run>/force_constants_hessian/
  <prefix>.dyn') on the q-grid of the coupling.
Write the exact paths and the settings of the run they come from into PLAN.md; state the cost of the downstream runs
and wait for my go.
```

**Check:** each downstream input is traceable to one run directory and its settings.
