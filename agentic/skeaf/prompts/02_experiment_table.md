# 02: Experimental data from tables

```
The experimental data are tables: <files>. Convert them into the canonical files
<project folder>/data/exp_LK_table.csv (single directions) and <project folder>/data/exp_sweep_peaks.csv (rotation),
following templates/exp_*_template.csv. Keep units in tesla and m_e; record peaks looked for but not seen as
observed = no; put the provenance (file, table, page, raw/fitted) in the '#' header lines. Map the
experimental angle to the calculation angle (read_experiment(angle_scale=..., angle_offset=...)) and explain the mapping.
Show me the first rows of each file and any inconsistencies (duplicates, different sessions, units).
```

**Check:** every row traces back to the source; masses from different sessions are kept as separate rows (`day`).
