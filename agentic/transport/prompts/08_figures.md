# 08: Publication figures

**When:** the comparison is settled.

```
Write plot_figures.py (templates/plot_template.py) reading only data/*.csv and tables/: experiment as markers (filled
direct, open indirect/replotted; one marker shape per sample), this work as solid lines (SERTA dashed if both are
shown), published theory gray dashed, ordered conditions (doping, temperature) on the ordinal ramp (<= 5 per panel),
log-log axes for mu(T) with the exponents annotated, linear rho(T) with rho0 stated, direct labels plus a style
legend, panel labels above the frame, a wrapped provenance note naming the corrections (Hall factor, rho0). Size for
<journal: 3.4 in single / 7.0 in double column>. Save vector PDF + 600 dpi PNG with save_figure. Render, look at every
PNG and fix collisions before reporting.
```

**Check:** look at the PNGs: no legend over data, no labels outside the frame, corrections and interpolations stated.
