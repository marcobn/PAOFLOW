# 12: A new or improved model run arrived

**When:** a new output folder (other resolution, segment, self-energy kind, denser q-grid, new temperatures).

```
A new run is in <system folder>/arpes/<output folder> (changes: <...>). Do not overwrite the previous comparison: copy
tables/ and figures/ to tables_<old label>/ and figures_<old label>/, point analysis.py to the new run
(ArpesModel.from_system(folder, output='<output folder>')), check its settings with describe(), re-run analysis.py and
plot_figures.py, and report the change of every metric (old -> new, with the baseline), which discrepancies the run was
meant to settle and whether it did.
```

**Check:** old and new numbers side by side; the settings of the new run verified; conclusions updated with dated notes.
