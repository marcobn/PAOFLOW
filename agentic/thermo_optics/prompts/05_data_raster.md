# 05: Raster figures (scans, embedded bitmaps, images)

**When:** the figure is a bitmap (embedded image or scanned page) or an image file.

```
Digitise <source> Fig. <N>, following the raster recipe of AGENT_WORKFLOW.md Stage 2:
1. Get the bitmap: extract_pdf_images (check orientation/resolution) or render_pdf_pages at 500 dpi; save to sources/.
2. Frame and ticks: find_frame (skew_tolerance for scans) and find_ticks, or projections; pair ticks with the printed
   values; say how you read the labels and whether the scale is uniform (piecewise Axis if not); give residuals.
3. Series: colour or dark mask, erase labels/legends; markers (filled_markers / ring_markers / marker_centroids),
   solid lines (trace_curve / trace_path with seeds), dashed lines (dash_segments + link_dashes); markers on lines:
   trace with and without the markers, then assign_to_curves.
4. Write data/<source>_fig<N>.csv (provenance 'digitised', y_rel_uncertainty from line/marker size and residuals,
   header with every anchor, seed and assumption) and checks/<source>_fig<N>_overlay.png.
5. Show me the overlay and a replot in data coordinates; give marker counts per series against the source.
```

**Check:** open the overlay yourself; count markers; any assumption (axis reading, label inference, curve ordering) is in the header.
