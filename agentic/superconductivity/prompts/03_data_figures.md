# 03: Δ(T) curves, α²F spectra, tunnelling conductances and gap distributions as figures

**When:** the data are figures (vector PDF, embedded bitmap, scan, image, slide).

```
Digitise <source> Fig. <N> following Stage 2 and the raster recipe of agentic/thermo_optics/AGENT_WORKFLOW.md
(vector: pdf_frame_ticks + pdf_vector_polylines; raster: find_frame, find_ticks, Axis with residuals, masks,
markers / tracers, assign_to_curves). One dataset per curve with its method, definition, sheet and temperature in
conditions (gap vs T: x = temperature; alpha^2F: x = omega in meV; conductance: x = energy in meV with
temperature_k). Write data/<source>_fig<N>.csv (provenance 'vector-extracted' or 'digitised', y_rel_uncertainty from
the marker / line size and the residuals, header with every anchor, seed and assumption) and
checks/<source>_fig<N>_overlay.png. Show me the overlay and a replot in data coordinates; give marker counts per
series against the source; anchor Delta(0) and the alpha^2F peaks against values quoted in the text.
```

**Check:** open the overlay yourself; count markers; each curve carries its definition and temperature.
