# Thermo-optics: optical response vs experimental data of any form

`PAOFLOW.agentic.thermo_optics` compares PAOFLOW's temperature-dependent
optical response with experimental data. It is agnostic of the material and of
the form of the data:
tables, free text, vector or scanned PDF figures, slide decks or images. The
same tools served an absorption edge, emissivity spectra, free-carrier
absorption and refractive indices of Si. Only `paoflow_optics` is specific to
optics; another observable needs another adapter.

The agent playbook, lessons learned, prompts and templates live outside the
package, in `agentic/thermo_optics/`.

## Data model

Every source becomes a **canonical long table** (`datasets.COLUMNS`):

```
dataset, quantity, x, x_unit, y, y_unit, temperature_k, conditions,
provenance, source, locator, y_rel_uncertainty, note
```

- **`provenance`** is one of five values:
  - `measured`: raw data;
  - `transcribed`: typed from a table or text;
  - `vector-extracted`: exact coordinates from PDF paths;
  - `digitised`: from a bitmap;
  - `calculated`: models and published theory.

  `read_dataset` rejects any other value.
- **`conditions`** holds `key=value` pairs (`format_conditions`, `parse_conditions`).
- **Header.** `write_dataset(path, frame, header)` writes the provenance record as leading `#`
  lines: source, method, calibration residuals and assumptions. `dataset_header_lines` reads it
  back. Only the *leading* `#` lines are comments: a `#` inside a field is data.

## Modules

| module | role |
|---|---|
| `datasets` | canonical table, provenance, conditions |
| `units` | photon energy ↔ wavelength ↔ wavenumber ↔ frequency, temperatures |
| `ingest` | `pdf_inventory`, `render_pdf_pages`, `extract_pdf_images`, `image_placements`, `pdf_words`, `pdf_drawing_summary`, `pdf_vector_polylines`, `pdf_frame_ticks`, `read_table`, `extract_slide_deck` (PyMuPDF, imported lazily) |
| `digitize` | `Axis` (linear/log, `piecewise=True` for hand-drawn axes), `AxisCalibration`; masks (`dark_mask`, `color_mask`, `erase_box`); `find_frame` (`skew_tolerance`), `find_ticks`; markers (`marker_centroids`, `filled_markers`, `ring_markers`); curves (`trace_curve`, `trace_path`, `dash_segments` + `link_dashes`); `assign_to_curves`, `resample`, `overlay_check` |
| `metrics` | `interpolate`, `curve_deviation` (log10/linear/relative), `best_shift`, `threshold_crossing`, `arrhenius_fit`, `match_conditions`, `interpolate_in_condition` |
| `paoflow_optics` | `OpticsModel`, the adapter for `absorption.npz` / `emissivity.npz` of `elphon.phonon_assisted_absorption` |
| `plotting` | `publication_style`, `temperature_colors` (validated ordinal ramps, ≤ 5 steps), `label_line`, `panel_label`, `provenance_note`, `save_figure` (PDF + 600 dpi PNG) |

## Design choices

- **Digitisation and comparison are separate.** A study writes `data/*.csv`, then compares, then
  plots. Each step reads only the previous step's files, so a figure can be restyled, or the
  model swapped, without touching the digitisation.
- **Adapters are thin and read-only.** An adapter maps an output folder to an *observable* at
  given conditions. `OpticsModel` provides:
  - `absorption_coefficient`: α = 2ωκ/c from ε, or the driver's constant-n α;
  - `refractive_index`;
  - `emissivity(T, thickness_m, angle_deg)`: recomputed for the experimental slab with
    `response.emissivity.emissivity_from_dielectric`.

  A new observable needs a new adapter, not a new pipeline.
- **Interpolated models are labelled.** `OpticsModel.dielectric(T, interpolate=True)` takes ε₁
  linear in T and ln ε₂ linear in 1/T. Absorption interpolates ln α in 1/T. `bracket(T)` gives the
  two model temperatures, used to quote the sensitivity.
- **Tracers for real scans.**
  - `trace_path` steps by arc length with a turn limit, so it follows vertical segments and sharp
    peaks.
  - `link_dashes` chains dash blobs along their principal axes, with a lateral penalty. This
    separates dashed lines packed closer than their gaps, where a straight continuation lands on
    the neighbour.
  - `filled_markers` uses a morphological opening that erases strokes thinner than the dots.
  - `assign_to_curves` gives each marker to the polyline nearest it, and returns −1 for label
    glyphs.
- **Metrics come with baselines.** The library computes deviations; the study is expected to quote
  each one next to a published calculation, a second measurement, or a trivial model (see the
  playbook).

## Tests

`tests/unit/agentic/thermo_optics/`:
- dataset round trip (header, `#` in fields, provenance check);
- units;
- metrics on analytic cases (log ratio, recovered shift, Arrhenius energy, exact 1/T
  interpolation);
- synthetic rasters: skewed frame, ticks, dots on a line with glyphs, JPEG-broken rings, a steep
  tanh curve, two dashed lines 12 px apart;
- a generated vector PDF (exact frame, ticks and curve);
- `OpticsModel` on synthetic outputs;
- the plotting helpers.
