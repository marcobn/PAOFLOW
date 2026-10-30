# 04: Figures drawn as PDF vectors

**When:** `ingest.pdf_drawing_summary` shows coloured or dashed paths with many segments on the figure's page.

```
Extract <paper> Fig. <N> (page <p>) from the vector paths: pdf_drawing_summary (colours, segment counts),
pdf_frame_ticks(pdf, page, rect) for the frame and the major/minor ticks (PDF points), pair ticks with the printed
values (pdf_words), build the AxisCalibration and report its residuals; pdf_vector_polylines(pdf, page, colour, rect)
for each series; identify the curves (order, label positions, legend). Provenance 'vector-extracted' (and "as replotted
in ..." if the figure replots older data). Write data/<source>_fig<N>.csv and replot it next to a render of the page.
```

**Check:** curve count = series in the figure; the replot overlays the render; shifts or scalings stated in the caption are recorded.
