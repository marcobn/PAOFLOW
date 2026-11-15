# 03: A band-map image

**When:** the source is an intensity map (figure or image file) rather than extracted points.

```
Calibrate and analyse the band map <source> Fig. <N>:
1. Get the bitmap (extract_pdf_images or render_pdf_pages at 500 dpi; save to sources/); crop the panel.
2. Frame and ticks: find_frame / find_ticks or projections; Axis for k (columns) and E (rows) with the printed values;
   give the residuals; note the colour scale (grey, sequential, or a non-monotonic map: state the channel you use).
3. band_map_from_image(image, k_axis, e_axis, rows, cols, channel, dark_is_high) -> data/<source>_map.npz (k, energy_ev,
   intensity); checks/<source>_map_overlay.png with the frame, ticks and the k_F / E_F anchors.
4. MDC analysis with curves.mdc_dispersion (method 'lorentzian', energy and k windows you state, min_contrast) -> a
   'dispersion' dataset (method=mdc-from-image) and a 'mdc_fwhm' dataset, provenance 'digitised', uncertainty from the
   pixel size in data units and the calibration residuals.
5. Validate against the authors' fits where they exist (overlay in data coordinates); report k_F and the kink energy
   you find.
```

**Check:** the colour-scale assumption and the pixel size in data units are in the header; the overlay shows the extracted dispersion on the image.
