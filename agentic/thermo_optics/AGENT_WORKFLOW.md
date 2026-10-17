# Thermo-optics workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) comparing PAOFLOW's **temperature-dependent optical response** (absorption, free carriers, refractive index, emissivity) with **experimental data of any form** for **any material**. The stages are not specific to optics: another observable needs only a new model adapter (Stage 4). It distils a complete study into stages, decision rules and checks: Si phonon-assisted absorption and thermal emissivity against five figures from three papers (October 2026).

The tooling is `PAOFLOW.agentic.thermo_optics`. A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `digitize_sources.py` | the sources (PDF, image, table, deck) | `data/<source>.csv` (canonical, provenance header), `checks/*_overlay.png` |
| `analysis.py` | `data/*.csv` + the model output (read-only) | `data/model_curves.csv`, `tables/metrics.csv`, `tables/*.csv`, `tables/summary.md` |
| `plot_figures.py` | CSV files only | `figures/*.pdf` (vector) + `*.png` (600 dpi) |

The split matters. Digitisation is reproducible and inspectable on its own. The comparison is a pure function of the CSV files. Figures can be restyled without touching the model.

```python
from PAOFLOW.agentic.thermo_optics import digitize as dg, ingest, metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset, read_dataset
from PAOFLOW.agentic.thermo_optics.paoflow_optics import OpticsModel     # or your adapter
```

Read `LESSONS_LEARNED.md` once before starting.

---

## 0. Operating rules (always)

1. **Read-only inputs.** Never write into calculation folders or source folders. Renders and extracted images go to the study's `sources/`.
2. **Ask before compute.** Ask before any new calculation or anything above ~10 min of CPU. Comparing existing output needs no new run. When a new run would settle a question, *propose* it: parameters, cost, what it decides.
3. **Provenance on every number.** The `provenance` column takes one of five values:
   - `measured`: raw data from the experimentalists;
   - `transcribed`: typed from a table or text;
   - `vector-extracted`: exact coordinates from PDF paths, though possibly a *replot*;
   - `digitised`: from a bitmap;
   - `calculated`: models and published theory.

   Quote it in every table and figure caption. A replot in a later paper is not the original: say "as replotted in …".
4. **Uncertainty on every digitised number.** State the calibration residuals (pixels → data units), the line or marker size in data units, and any assumption (axis reading, label inference) in the dataset header.
5. **Look at every overlay.** A digitisation is accepted only after you have viewed its `checks/*_overlay.png` and re-plotted the data in data coordinates next to the source. Count markers against the source.
6. **A baseline with every metric.** A deviation means nothing alone. Quote it next to at least one of:
   - (a) a published calculation of the same data;
   - (b) the spread between independent measurements;
   - (c) a trivial model, such as a constant at the experimental mean, or a one-parameter fit.
7. **Conditions are matched, not assumed.** Temperature, thickness, angle, doping, polarisation, wavelength. Use `mt.match_conditions` with a stated tolerance. Anything evaluated between model grid points is an *interpolated model*: label it, and quote the sensitivity using the bracketing grid points.
8. **No silent fitting.** Rigid shifts (`mt.best_shift`) and scale factors are *diagnostics*, quoted next to the unshifted deviation. Never replace the model curve with a shifted one in a figure without saying so.
9. **Compare the observable that was measured.** Recompute the model observable at the experimental geometry: slab thickness, angle, polarisation, resolution. Do not compare a model property at another geometry (e.g. emissivity of the default slab) with the measurement.
10. **Correct yourself explicitly.** Add dated correction notes to affected reports, and say so in the next message.
11. **PAOFLOW conventions.** Inputs are Python arguments; there are no config files. New library code comes with fast tests and NumPy docstrings.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: model run, sources, audience | adapter `.describe()`, `ingest.pdf_inventory` | inventory, open questions |
| 1 | Physics map: observable ↔ model quantity ↔ conditions ↔ baseline | (reasoning, the user) | the comparison plan (table) |
| 2 | Ingestion by form → canonical datasets | `ingest.*`, `digitize.*`, `datasets.*` | `data/*.csv`, `checks/` |
| 3 | Validation of the ingestion | overlays, replots, internal cross-checks | accepted datasets |
| 4 | Model adapter and condition alignment | adapter, `mt.match_conditions`, interpolation | `data/model_curves.csv` |
| 5 | Comparison metrics with baselines | `mt.curve_deviation`, `best_shift`, `threshold_crossing`, `arrhenius_fit` | `tables/metrics.csv` + derived tables |
| 6 | Diagnosis | decision tree; cheap checks on existing output | ranked causes, proposed runs |
| 7 | Publication figures | `plotting.*` | `figures/` |
| 8 | Report | `templates/report_template.md` | `REPORT.md` |

---

## Stage 0: intake

1. **Model.** Find the driver and its parameters: grids, temperatures, energy grid, scissor, smearing, approximations. Read the output files and their units. For PAOFLOW optics, `OpticsModel(output).describe()` gives the table. Note what the model *cannot* contain: missing physics, energy-grid limits, absent temperatures.
2. **Sources.** For each file, record its form:
   - text-PDF vs scanned PDF: `ingest.pdf_inventory`, which lists words, images and vector paths per page;
   - embedded bitmap vs vector figure: `ingest.image_placements`, `ingest.pdf_drawing_summary`;
   - table, deck or image.
   Record the figures and tables that hold data, and the *conditions* stated in the text: sample, thickness, doping, geometry, temperature, method.
3. **Audience and deliverable.** Ask for a paper figure, an internal check or a collaborator package.
4. Report the inventory and the questions before digitising anything.

## Stage 1: physics map (decide what is compared)

Build a table with one row per comparison: *observable (source, figure) ↔ model quantity ↔ conditions to match ↔ comparison space ↔ window ↔ baseline*. Rules:

- **Choose the space.**
  - `log10` for quantities spanning decades (α, conductivity, rates);
  - `linear` for bounded ones (emissivity, reflectance);
  - `relative` for n or energies.
- **Split by physics windows** where different mechanisms dominate, e.g. opaque / edge / free-carrier / lattice bands. A single global rms hides everything.
- **Prefer derived, robust quantities** where the model has known offsets: onset energies (`threshold_crossing`), activation energies (`arrhenius_fit`), exponents (log-log slopes), slopes (dn/dT).
- **Flag out-of-scope physics up front.** Note windows where the model lacks a mechanism (e.g. multiphonon bands), and report them as such, not as failures.
- **Confirm with the user** before ingestion.

## Stage 2: ingestion by form

Everything downstream reads the canonical long format (`datasets.COLUMNS`):

`dataset, quantity, x, x_unit, y, y_unit, temperature_k, conditions, provenance, source, locator, y_rel_uncertainty, note`

Build the frames with `make_dataset(...)` (conditions as a dict → `key=value; ...`), and write them with `write_dataset(path, frame, header_lines)`. The header lines are the provenance record: source, figure, method, calibration residuals, assumptions, overlay file.

| Form | Procedure |
|---|---|
| **Table** (CSV, XLSX, paper table) | `ingest.read_table`; rename columns into the schema; `provenance='measured'` (raw files) or `'transcribed'`; keep units as printed and convert with `units`. |
| **Free text** (paper, notes, e-mail) | Transcribe value + unit + condition with the sentence in `note`; `'transcribed'`. Check every row against the source. |
| **Vector PDF figure** | `ingest.pdf_drawing_summary(pdf, page)` → colours and segment counts; `ingest.pdf_frame_ticks(pdf, page, rect)` → frame and major/minor ticks in PDF points (exact); `ingest.pdf_vector_polylines(pdf, page, color, rect=...)` → one polyline per curve; calibrate with `dg.Axis(tick_points, tick_values)`; `'vector-extracted'`. Identify curves by order or label positions (`ingest.pdf_words`). |
| **Embedded bitmap** in a PDF | `ingest.extract_pdf_images(pdf, page, sources/)` (native resolution; check orientation: scans may be rotated, so render the page instead with `ingest.render_pdf_pages(..., dpi=400-600)`). |
| **Scanned page** | `ingest.render_pdf_pages(pdf, [page], sources/, dpi=500)`. |
| **Raster figure** (any of the two above, or an image file) | see "Raster recipe" below; `'digitised'`. |
| **Slide deck** | `ingest.extract_slide_deck(deck, sources/)` → `slides.md` + images; tables from the slide text, figures as raster. |

### Raster recipe

1. **Mask.** `dg.load_image`, then `dg.dark_mask(image, threshold)` for black ink, or `dg.color_mask(image, rgb, tolerance, min_saturation)` for coloured series.
2. **Frame.** `dg.find_frame(mask, min_fraction, skew_tolerance=3-5)` on a crop around one panel. With skewed scans or bent hand-drawn axes, project rows and columns and read the frame lines from the profile.
3. **Ticks.**
   - `dg.find_ticks(mask, frame, 'x' | 'y', length)`. For bent axes, detect ticks relative to the local frame line.
   - Pair ticks with printed values. Read the labels yourself, and check label positions against a uniform scale.
   - Build `dg.Axis(pixels, values, 'linear' | 'log')`. Use `piecewise=True` when the drawn tick spacing departs from one scale (hand-drawn axes); `residuals()` then shows the departure.
4. **Clean.** Erase legends and labels with `dg.erase_box(mask, rows, cols)`.
5. **Extract.**
   - Filled dots on fitted lines: `dg.filled_markers(mask, radius)`, an opening that removes strokes thinner than the dots.
   - Open circles, including JPEG-broken ones: `dg.ring_markers(mask, radius, width)`.
   - Generic blobs: `dg.marker_centroids`.
   - Solid single-valued lines: `dg.trace_curve`.
   - Solid lines of any slope: `dg.trace_path(mask, seed, direction, ...)`. Use several seeds per curve, with `max_turn_deg` up to 80 for sharp peaks.
   - Dashed lines packed closer than their gaps: `dg.dash_segments` + `dg.link_dashes(segments, seed, direction, max_link)`.
   - Markers on lines: trace the lines on the ink with **and** without the markers (`LESSONS_LEARNED.md` #12), then `dg.assign_to_curves(markers, lines, max_distance)`. Glyphs far from every line get −1.
6. **Map and resample.** `calibration.to_data(pixels)`; thin dense traces to a fixed cell (5 px), or resample with `dg.resample(points, x, log_y)`.
7. **Check.**
   - `dg.overlay_check(image, pixels, checks/<name>.png)`;
   - a replot in data coordinates;
   - marker counts against the source;
   - internal consistency where two figures share points.

## Stage 3: validation of the ingestion

- **Calibration.** Report residuals in pixels and in data units (e.g. "0.3 % of a decade").
- **Sanity anchors.** Compare with known values: 1−R of a well-known material, a band gap, a quoted number in the text. This is how the Sato wavelength axis was decided (lesson 4).
- **Cross-figure checks.** Shared points of two figures (Sato Figs. 8/9 agree within 10 %).
- **Assumptions.** Each one goes in the header and in the report's caveat box.

## Stage 4: model adapter and condition alignment

- **Adapter.** A small class that loads the output read-only and returns the *observable* at given conditions (`templates/model_adapter_template.py`). For PAOFLOW optics, `OpticsModel` provides:
  - `absorption_coefficient(T, constant_index, interpolate)`: α = 2ωκ/c from ε, or the driver's constant-n α;
  - `refractive_index(T, interpolate)`;
  - `emissivity(T, thickness_m, angle_deg, interpolate)`: the slab at the experimental geometry;
  - `dielectric(T, interpolate)`: ε₁ linear in T, ln ε₂ linear in 1/T;
  - `bracket(T)`, `describe()`.
- **Temperatures.** `mt.match_conditions(exp_T, model_T, tolerance)`. Beyond the tolerance, either interpolate (label it, and quote the bracketing biases) or drop the point. Thermally activated quantities interpolate in ln y vs 1/T; smooth ones linearly in T.
- **Write every model curve used** to `data/model_curves.csv` (`provenance='calculated'`, the interpolation in `note`).

## Stage 5: comparison metrics with baselines

| Tool | Use |
|---|---|
| `mt.curve_deviation(x_e, y_e, x_m, y_m, space, window)` | n, bias, rms, median, max\|res\|; model interpolated at the experimental x (`log10`: in log y) |
| `mt.best_shift(..., shifts)` | rigid abscissa shift (diagnostic): rms before/after |
| `mt.threshold_crossing(x, y, level)` | onsets (absorption edge, transparency of a slab) |
| `mt.arrhenius_fit(T, y)` | activation energies (carriers, rates) |
| log-log `np.polyfit` | power-law exponents |
| `mt.interpolate_in_condition` | model family to one condition value |

Write one row per (comparison, dataset, condition, window) to `tables/metrics.csv`, with `baseline`, `baseline_rms` and `note`. Derived quantities (onsets, E_a, exponents, slopes) get their own tables. Void a baseline that has no meaning, such as a constant baseline on a flat spectrum.

## Stage 6: diagnosis

Look for **one cause explaining several discrepancies** before listing many causes. In the Si study, an n excess of 7–13 % explained three discrepancies:
- the opaque-emissivity deficit;
- the long-wavelength high-T emissivity deficit;
- part of the α deficit when α is computed from κ.

Cheap checks on existing output come before any new run:
- parameter sensitivity already stored (e.g. α vs η in `absorption.npz`);
- component split (indirect vs direct ε₂);
- the bracketing temperatures;
- the constant-n vs n(ω) α.

Then propose runs, each with its expected effect and cost. See the decision tree below.

## Stage 7: publication figures

- **Encodings.**
  - Experiment as markers: filled for direct data, open for indirect or replotted data.
  - This work as solid lines.
  - Published theory as gray dashed lines (`plotting.REFERENCE_THEORY`).
  - Temperature (or any ordered condition) on a single-hue ordinal ramp: `plotting.temperature_colors`, at most 5 per panel, so facet larger sets. A second ordered variable gets `TEMPERATURE_RAMP_ALT`.
- **Labels.** Direct labels (`plotting.label_line`) for ≤ 4–5 series, plus a style legend for the encodings. Text in ink colours.
- **Layout.** `plotting.publication_style(width_in=3.4 | 7.0, height_in)`. Panel labels go above the frame (`panel_label`), and a wrapped `provenance_note` says what is digitised or interpolated.
- **Output.** `save_figure(fig, figures/<name>)` → vector PDF + 600 dpi PNG.
- **Look at every figure.** Check collisions, labels outside the frame, and legends over data.

## Stage 8: report

Follow `templates/report_template.md`:
1. a provenance caveat box;
2. a bottom-line table: observable, agreement, discrepancy, likely cause;
3. the data inventory with methods and uncertainties;
4. the model run;
5. results per comparison, with numbers and files;
6. diagnosis and proposed runs;
7. how to reproduce.

---

## Decision tree for a discrepancy

```
model - experiment discrepancy
 ├─ same observable and geometry?        -> recompute at the experimental thickness / angle / polarisation
 ├─ same conditions? (T, doping, sample)  -> match_conditions; interpolation sensitivity (bracketing T)
 ├─ ingestion error?                      -> overlay, replot, anchors, axis assumptions, label inference
 ├─ grid limits? (energy step, range)     -> onsets within one step? KK range? low-energy start?
 ├─ a global offset? (gap, n, scale)      -> best_shift / ratio: uniform over windows and T? one cause for several?
 ├─ method choice?                        -> sensitivity in existing output (smearing, eta, constant n, components)
 ├─ missing physics?                      -> window where the mechanism is absent (multiphonon, excitons, impurities)
 └─ none                                  -> report unexplained, with what was excluded and which run or measurement would decide
```

## Ask the user at the start

1. The calculation folder(s) (read-only), the driver, and older runs.
2. The sources, with their figures or tables and their provenance. Is any of them a replot?
3. The experimental conditions: sample, thickness, doping, geometry, temperatures.
4. The deliverable and its audience.
5. The compute budget: may new runs be launched?

## Worked example (Si, October 2026)

Study folder `EPW_Si_large/EPW_Si/thermo_optics/`, on the user's workstation and not in the repository. Model: PAOFLOW phonon-assisted + direct absorption and slab emissivity, 48³ k/q, scissor to Varshni E_g(T).

| Source | Form | Tools |
|---|---|---|
| Noffsinger PRL 2012 Fig. 2 (Macfarlane 1958 replot + GW theory) | vector PDF | `pdf_frame_ticks`, `pdf_vector_polylines` |
| Noffsinger Fig. 3 (Braunstein 1958 circles + theory lines) | embedded JPEG | `color_mask`, `ring_markers`, `trace_curve` |
| Fan 1956 Fig. 4 | scanned line | `trace_path` (6 seeds) |
| Sato 1967 Fig. 3 (emissivity, solid + dashed) | scanned | piecewise-log `Axis`, `trace_path`, `dash_segments` + `link_dashes` |
| Sato Figs. 5, 8, 9 (dots on lines) | scanned | `filled_markers`, traces with and without dots, `assign_to_curves` |

Comparisons:
- edge onsets (`threshold_crossing`);
- α in log10 with the GW reference and the exp-exp spread;
- emissivity in linear space, by physics windows, interpolated in T with brackets;
- free-carrier α, E_a and λ exponents;
- n and dn/dT.

The report is `REPORT.md` in the study folder.
