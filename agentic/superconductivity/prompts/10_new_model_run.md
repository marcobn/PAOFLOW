# 10: A new or improved model run arrived

**When:** a new output (finer grids, adaptive smearing, equal grids, other temperatures, spin-orbit).

```
A new run is in <system folder>/superconductivity/<output folder> (changes: <...>). Do not overwrite the previous
comparison: copy tables/ and figures/ to tables_<old label>/ and figures_<old label>/, point analysis.py to the new run
(EliashbergModel.from_system(folder, output='<...>'), GapModel / AnisotropicGapModel likewise), check its settings with
describe(), re-run analysis.py and plot_figures.py, and report the change of every metric (old -> new, with the
baseline), which discrepancies the run was meant to settle and whether it did.
```

**Check:** old and new numbers side by side; the settings of the new run verified; conclusions updated with dated notes.
