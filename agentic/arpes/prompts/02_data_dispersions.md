# 02: Dispersions, widths, self-energies, λ and linewidths from tables, text or figures

**When:** the data are E(k) points, MDC widths, Re/Im Σ(ω) curves, λ values or linewidths versus T.

```
Ingest <file / figure N / sentence> into the canonical format with the ARPES conventions of AGENT_WORKFLOW.md Stage 2
(quantities dispersion / mdc_fwhm / re_self_energy / im_self_energy / lambda / kink_energy / linewidth; conditions
photon_ev, polarization, cut, kz, resolution_mev, method, bare_band, impurity_offset_mev, band, state). For figures follow
the vector / raster recipes of agentic/thermo_optics (frame, ticks, Axis with residuals, markers or tracers). Keep the
units as printed (meV or eV, 1/A, relative to the authors' k_F if so published: k_reference=kF). Provenance 'measured',
'transcribed', 'vector-extracted' or 'digitised'; an extracted Sigma or lambda carries the authors' bare band in
conditions. Write data/<source>_<quantity>.csv with write_dataset and a header naming the file/figure, the conditions and
every assumption; checks/<source>_overlay.png for figures. Show me the overlay and a replot.
```

**Check:** open the overlay yourself; the bare-band protocol and the resolution of every extracted quantity are recorded.
