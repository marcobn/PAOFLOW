# 14: A new or improved model run arrived

**When:** a new output folder (finer grid, other parameters, new temperatures).

```
A new run is in <system folder>/optics/<outdir> (changes: <...>). Do not overwrite the
previous comparison: copy tables/ and figures/ to tables_<old label>/ and figures_<old label>/, point analysis.py to
the new run (OpticsModel.from_system(folder, output='<outdir>')),
re-run analysis.py and plot_figures.py,
and report the change of every metric (old -> new, with the baseline), which discrepancies the run was meant to settle
and whether it did.
```

**Check:** old and new numbers side by side; conclusions updated with dated notes.
