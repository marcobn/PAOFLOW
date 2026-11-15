# 02: SHG coefficients, spectra and photocurrents from tables, text or figures

```
Ingest <file / table / figure / sentence> into the canonical format with the conventions of AGENT_WORKFLOW.md Stage 2
(quantities shg_d, shg_chi, shift_current, injection_current; conditions component, reference, method, sample,
thickness_nm, scissor_ev, functional, grid, eta_ev for calculations): read or transcribe (tables, text) or digitise
(vector / raster recipes of agentic/thermo_optics) with the photon energy or wavelength as printed; provenance
'measured', 'transcribed', 'vector-extracted', 'digitised' or 'calculated'; y_rel_uncertainty from the source.
Write data/<source>.csv with a header naming the file, the convention and every assumption; checks/ overlays for
figures. Show me the overlay and a replot; list the conversions you will apply later (d = chi/2, pm/V from esu using
the source's own factor, sheet / effective 3D).
```

**Check:** the convention of every number is in `conditions`; overlays looked at; no conversion applied silently.
