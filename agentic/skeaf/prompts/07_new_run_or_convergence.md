# 07: A new (denser / different) calculation arrived

```
A new calculation is in <path> (<what changed: grid, num_interpolation, bands, angles, SOC, E_F>).
1. FIRST check that the previous run is archived (do not use or overwrite a folder that was replaced);
   if old raw files are gone, say which numbers can no longer be reproduced.
2. Register the previous run with `add_reference_run(...)` and call `fermi_surface_inventory()`, `sweep_comparison()`,
   `compare_runs(frequency_windows=[(<fmin>, <fmax>)])`.
3. Report per-group median shifts, which sheets are converged (< 0.2 % for large sheets), which small pockets
   moved, and whether any orbit is genuinely NEW (verify against the old raw data at all angles).
4. Update the assignments and the reports; add a dated correction note to any earlier statement that changed.
```
