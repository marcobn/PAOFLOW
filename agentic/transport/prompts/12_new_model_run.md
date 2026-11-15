# 12: A new or improved model run arrived

**When:** a new output (finer grids, other flags, new temperatures, extra channels).

```
A new run is in <system folder>/transport/<tag> (or superconductivity/output, or <study>/boltzmann) (changes: <...>).
Do not overwrite the previous comparison: copy tables/ and figures/ to tables_<old label>/ and figures_<old label>/,
point analysis.py to the new run (MobilityModel.from_system(folder, output='<tag>') /
ResistivityModel.from_system(folder, output='<tag>')), check its flags in the log and describe(), re-run analysis.py
and plot_figures.py, and report the change of every metric (old -> new, with the baseline), which discrepancies the
run was meant to settle and whether it did.
```

**Check:** old and new numbers side by side; the flags of the new run verified; conclusions updated with dated notes.
