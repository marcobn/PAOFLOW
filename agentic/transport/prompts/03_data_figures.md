# 03: Experimental data as figures

**When:** the data are a figure (vector PDF, embedded bitmap, scan, image, slide).

```
Digitise <source> Fig. <N> following Stage 2 and the raster recipe of agentic/thermo_optics/AGENT_WORKFLOW.md
(vector: pdf_frame_ticks + pdf_vector_polylines; raster: find_frame, find_ticks, Axis (log axes: read the decades),
masks, markers / tracers, assign_to_curves). Transport figures are usually log-log mu(T) or linear rho(T) with
several samples: one dataset per sample / carrier / method, with its doping, purity and method in conditions.
Write data/<source>_fig<N>.csv (provenance 'vector-extracted' or 'digitised', y_rel_uncertainty from the marker
size and the calibration residuals, header with every anchor, seed and assumption) and
checks/<source>_fig<N>_overlay.png. Show me the overlay and a replot in data coordinates; give marker counts per
series against the source.
```

**Check:** open the overlay yourself; count markers; each series is tied to a sample with stated conditions.
