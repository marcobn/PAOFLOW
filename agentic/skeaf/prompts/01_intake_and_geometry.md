# 01: Intake and geometry check

**When:** after kickoff, or whenever a calculation or experimental input changes.

```
Run `fermi_surface_inventory()` and `field_directions(...)` in <analysis.py>. Then:
1. List every qo file with rows and angle coverage; flag empty files and other E_F tokens.
2. Confirm bands crossing E_F vs computed bands; list the degenerate groups and the Fermi-surface pieces per group.
3. From the NumOrbCopy histogram, propose min_copies and justify it (is it bimodal? where are paired,
   smoothly tracking orbits?). Plan to repeat key results at 100.
4. For every experimental direction give (azimuthal, polar) in BOTH conventions and the indices of the
   nearest sweep grid angle; check statements in the experimental source (e.g. "[110] is 30 deg off [100]")
   and translate any index labels back with field_geometry.field_angles_to_direction.
Report discrepancies as questions; do not guess the rotation plane.
```

**Check:** `geometry/geometry.md` shows OK for every direction; any mismatch comes with an explanation.
