# SKEAF agentic workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) interpreting **PAOFLOW pyskeaf / SKEAF** extremal-orbit calculations against **experimental quantum-oscillation data** (dHvA, SdH, torque) for *any* compound. It distils a complete analysis (NbSn2, October 2026) into stages, decision rules and checks.

The tooling is `PAOFLOW.agentic.qo_analysis`, driven from a PAOFLOW-style analysis script. Paths below are relative to this folder (`PAOFLOW/agentic/skeaf/`). A template is in `templates/analysis_template.py`; a worked example is `../../examples/qo_analysis_examples/NbSn2/main.py`. Each study lives in its own folder `projects/<system>/` (analysis script, canonical CSVs, outputs, reports); calculation folders stay where they are and are only read.

```python
from PAOFLOW.agentic.qo_analysis import QuantumOscillationAnalysis
qo = QuantumOscillationAnalysis(skeaf_dir='output', primitive_cell=..., ...)
qo.fermi_surface_inventory()
...
```

Every method writes its tables, figures and a short report to `output_dir/<step>/` and returns a summary. Read `LESSONS_LEARNED.md` once before starting.

---

## 0. Operating rules (always)

1. **Never overwrite calculation output.** pyskeaf writes fixed file names. New runs go to new folders: `skeaf_rotation(...)` refuses non-empty destinations. If an `output/` folder changes under you, record which run your numbers came from.
2. **Ask before heavy compute.** Ask before new pyskeaf runs, `breakdown_network` (minutes per angle and core), or anything above ~10 min of CPU, and give an estimate. Smoke-test one angle at low `num_interpolation` first.
3. **Every claim is backed by a file.** Quote numbers from method outputs and name the file. Distinguish *measured*, *transcribed*, *digitised* and *calculated* numbers.
4. **Every match rate comes with a chance baseline.** The methods return log-uniform and angle-shuffled baselines; the latter is stricter. A rate within ~2σ of chance is not evidence.
5. **Assign from shape, not from single-angle coincidences.** On a dense multi-sheet Fermi surface a ±5 % window contains a calculated orbit 30–40 % of the time.
6. **Verify before calling something new, missing or fixed.** Compare with the archived previous run before saying "new orbit". Run `single_angle_check` before saying "this band has no orbits".
7. **Correct yourself explicitly.** Add dated correction notes to affected reports; say it in the next message.
8. **Units.** BXSF energies are in Ry. pyskeaf's `fermi_energy` is in eV. BXSF reciprocal vectors are in a.u.⁻¹ without 2π. SKEAF frequencies are in kT. F[T] = 10 476 × A[Å⁻²].
9. **Do not tune thresholds to improve agreement.** Justify each from the data, then show the conclusions survive a stricter value (e.g. `min_copies=100`).
10. **Mind the audience.** Collaborator deliverables need a provenance caveat box, a how-to-read section per figure, and a request list.
11. **PAOFLOW conventions.** Inputs are Python arguments in the analysis script (no config files). Parallel work goes through `resolve_n_jobs` (the methods already do this). New code comes with fast tests.

---

## Stage overview

| Stage | Goal | Method(s) | Key output |
|---|---|---|---|
| 0 | Intake: inputs, run parameters | `fermi_surface_inventory()` | `inventory/inventory.md` |
| 1 | Geometry: measured directions → SKEAF angles | `field_directions(directions)` | `geometry/geometry.md` |
| 2 | Bands, multiplets, Fermi-surface topology | `fermi_surface_inventory()` | groups, missing bands, pieces |
| 3 | Experiment → canonical tables | `read_experiment(...)`, `digitize_fft_map(...)`, `experiment_data.parse_text_peaks`, `experiment_data.extract_slide_deck` | canonical CSVs, validation |
| 4 | Robustness cut | `fermi_surface_inventory()['copy_histogram']` | justified `min_copies` |
| 5 | Single-direction candidates | `direction_analysis()` | candidate tables |
| 6 | Continuous sweep comparison and assignments | `sweep_comparison(...)` | overlays, per-angle listing, chance tests |
| 7 | Anisotropy ratios, mass enhancement | `direction_analysis()` | verdicts, 1 + λ |
| 8 | Validation | `add_reference_run`, `compare_runs`, `fermi_level_scan`, `single_angle_check`, `skeaf_runs.skeaf_rotation` | numerical vs physical |
| 9 | Magnetic breakdown and weak peaks | `breakdown_gaps`, `open_orbit_contacts`, `breakdown_network`, `weak_peak_analysis` | contacts, network orbits, mixing tests |
| 10 | Reporting and collaborator package | `collaborator_package(...)` | figures, data, scripts, README |

---

## Stage 0: intake

1. Find the pyskeaf driver (or the call to `paoflow.pyskeaf(...)`). Record `fermi_energy` (eV), `num_interpolation`, `azimuthal`, `polar`, `num_angles`, the tolerances, `allow_wall_orbits` and `bands`, plus the BXSF grid (file header).
2. Write the analysis script `projects/<system>/analysis.py` from `templates/analysis_template.py` and run `fermi_surface_inventory()`. Check:
   - **Empty frequency files.** A red flag: run `single_angle_check` (Stage 8) before interpreting.
   - **Angular coverage and grid step.** The grid may not contain exactly 45°.
   - **Other Fermi tokens.** They belong to Fermi-level scans.
   - **Older runs.** Register them with `add_reference_run(...)`.
3. Experimental inputs: list each file and its form (table, text, image, deck). For decks, use `experiment_data.extract_slide_deck(deck, folder)` and read `slides.md`.
4. Report the run parameters and open questions (current run? rotation plane? labels?).

## Stage 1: geometry

- SKEAF: polar is measured from +z; azimuthal is in the xy-plane from +x, in the BXSF Cartesian frame.
- **Real vs reciprocal.** A field normal to (hkl) uses the reciprocal convention; a field along [hkl] uses the real one. They differ for mixed indices in non-cubic cells (NbSn2 [110]: 29.95° vs 60.05°).
- Give `primitive_cell` (or `scf_file`) and `conventional_from_primitive` for centred cells. The row order matters; check the conventional lengths.
- `field_directions(...)` prints both conventions and flags mismatches. `field_geometry.field_angles_to_direction` converts a rotation angle back to indices. Use it to check labels in the experimental source (NbSn2: "[11y], y ≈ 2.4" at 45° is really [1 1 3.90]).
- Map the measured rotation angle onto the calculation angle with `read_experiment(angle_scale=..., angle_offset=...)`, and confirm the rotation plane with the user.

## Stage 2: bands, multiplets, topology

- **Bands crossing E_F** that were not computed mean missing sheets; ask for them.
- **Degenerate groups** (`degenerate_groups='auto'`). With SOC in a centrosymmetric, non-magnetic crystal every band is a Kramers pair: one Fermi surface. Count one band per pair. The residual frequency difference (0.3–4 %) is a numerical error bar. The partners are pseudospin states, not "spin up/down".
- **Fermi-surface pieces** per group (periodic connectivity). The number of frequency rows is the number of **extremal orbits**, not of surfaces. Repeated equal frequencies within one band are symmetry-equivalent orbits at different k.
- With SOC and non-SOC runs, set `parent_bands` (SOC band → SR band).

## Stage 3: experimental data

Everything downstream reads two canonical CSV files (`templates/exp_*_template.csv`). Put the provenance in their `#` header lines.

| Form | Procedure |
|---|---|
| Table (CSV, XLSX, paper) | Rename the columns into the schema. Record `observed = no` for peaks looked for but absent. |
| Free text (notes, slide text) | `experiment_data.parse_text_peaks(text)` as a first pass; **check every row** against the source and keep the source line. |
| Image (FFT magnitude vs angle) | Calibrate (two frequency ticks, two angle ticks, colour bar, data range). `digitize_fft_map(...)` samples **only the measured angles**, masks contour lines, validates against the LK rows of the configured directions (aim for a median deviation ≲ 1–2 %), and reports evenly spaced runs (striping). Magnitudes are semi-quantitative. |
| Deck (pptx; PDF via any extractor) | `extract_slide_deck`; LK tables from the slide text, maps from the images. |

The continuity of labels across angles is the experimentalists' interpretation. Record inconsistencies (masses differing between measurement days, two datasets for one direction) as questions.

## Stage 4: robustness cut

- Inspect `copy_histogram`. Genuine orbits appear in many supercell copies; wall artefacts in 1–10. At `num_interpolation` ≈ 200–320 the distribution may not be bimodal: paired, smoothly tracking genuine orbits can have 50–90 copies. Justify the cut (NbSn2: 50) and repeat the key steps at 100.
- `merge_fraction` (sheets within a group). Each candidate is compared with the cluster anchor. Use ~0.015 when symmetry-equivalent orbits are close but distinct; a large `spread_pct` means over-merging.
- The `extremum` column marks maxima (curvature < 0) and **minima** (necks); both are extremal orbits. The tips of pockets are not.

## Stage 5: single-direction candidates

`direction_analysis()` gives the sheet table per direction (nearest grid angle), dF/dE (Onsager) and candidates within 5 %. **They are candidates, not results.**

## Stage 6: continuous sweep comparison (decisive)

`sweep_comparison(min_copies=..., assignments=..., colors=..., labels=..., axis_label=...)` writes:
- the overlay;
- the overlay on the experimental image (if digitised);
- `per_angle_listing.txt`, the main reasoning tool;
- the summary with chance baselines.

Procedure:
1. Follow each measured branch across angles. Find a calculated branch of one group with the **same frequency and slope** over several angles, and note the deviation range.
2. Check carrier type, curvature (amplitude) and mass enhancement (Stage 7).
3. Classify: secure / medium / tentative / unexplained. An isolated calculated point that matches at one or two angles is not a branch.
4. Write the assignments CSV (`templates/branch_assignments_template.csv`), including unassigned measured branches, and re-run to label the overlay.
5. If the rate is within ~2σ of chance, say that frequency coincidences prove nothing.

## Stage 7: anisotropy and mass enhancement

From `direction_analysis()`: same-group candidate pairs ranked by

score = |F₂/F₁ (calc) − F₂/F₁ (exp)| + mean |ΔF|/F + w · mean |ln(m_exp/m_band)|

with the verdicts `anisotropy reproduced`, `uniform offset`, `right sign, wrong magnitude` and `WRONG SIGN`.
- An empty result means no common label; it is not an error.
- A ratio "reproduced" by an isolated calculated point is not a confirmation.
- 1 + λ = m_LK/m_band should be > 1 and cluster for secure branches (NbSn2: 2.1–2.9).

## Stage 8: validation

| Check | How | Reading |
|---|---|---|
| Grid / interpolation convergence | new run with `skeaf_rotation(..., destination=new_folder)`; `add_reference_run(old)`; `compare_runs()` | Large sheets < 0.2 %; small pockets (< 50 T) often 1–30 % (provisional). Verify "new" orbits against the old raw data. |
| Branch refinement | single band, angular window, higher interpolation; `compare_runs(frequency_windows=[(fmin, fmax)])` | Unchanged coverage means a physical gap at this E_F. |
| Fermi level | shifted runs (eV); `fermi_level_scan(dir, bands, tokens, energy_step_meV)` | Median measured/Onsager ≈ 1 confirms units and a rigid response. Check orbits that appear or disappear. Reject global shifts that break any good match or differ between directions. |
| SOC | SOC run plus SR reference with `parent_bands`; `compare_runs()` | Report what improves **and** what worsens. |
| Empty / missing band | `single_angle_check(band, az, pol)` | Orbits found locally mean the production job died; re-run that band. |
| Sorted-band artefacts | `breakdown_gaps` without SOC | Gaps ≈ 0 with ΔE of a few meV mean crossing bands; sorted-band orbits there are not physical. |

## Stage 9: magnetic breakdown and weak peaks

1. `breakdown_gaps(bands=one per group)`: k-gap, B₀ ≈ (π/2)(ħ/e)k_g² ≈ 1.03 × 10⁵ T Å² · k_g², and splitting. Contacts with B₀ below the maximum field are active. Gaps below the voxel size are resolution-limited.
2. `open_orbit_contacts(open_band, other_bands, angles)`: whether open trajectories meet other sheets where orbits are missing.
3. `breakdown_network(bands, open_band, ...)`: **heavy**. Ask first and smoke-test with `angles=[one]`. Robust network orbits are compared with the unexplained peaks against chance.
   - Extended near-contacts produce dense ladders (point-junction artefact): judge them statistically.
   - A family in the right F/mass range with tiny R is "viable but unconfirmed"; state its gap sensitivity (R ∝ P^(n/2), P = e^(−B₀/B)).
4. `weak_peak_analysis(network=...)`:
   - weak peaks vs network orbits;
   - the model-independent combination test (2F, F_i ± F_j of strong peaks at the same angle) vs chance;
   - the parent check (breakdown needs **touching** sheets; same or non-touching sheets point to mixing);
   - striping.
5. "Expected" (small B₀) ≠ "demonstrated". The experimental discriminant is the field dependence: breakdown has an extra exp(−nB₀/2B), while mixing scales with the product of the parents' amplitudes.

## Stage 10: reporting

- **Internal report.** Inputs and parameters, the robustness justification, the assignment table, tests with chance baselines, rejected template verdicts and why, corrections, and open items.
- **Collaborator package.** Run `collaborator_package(destination, ...)`, then `plot_figures.py` and `weak_peak_tests.py` inside it. Fill `README.md` from the template:
  - a provenance caveat box;
  - how to read each figure;
  - assignments, tests, bottom line;
  - requests to the experimentalists;
  - an FAQ on the orbit table (Kramers pairs, symmetry duplicates, extremum, copies).
- If the user renames or archives a delivered folder, update that copy and ask them to re-send.

---

## Decision tree for an unexplained branch

```
unexplained branch
 ├─ sheet present? (inventory: bands crossing E_F, empty files)  -> single_angle_check, re-run band
 ├─ branch at neighbouring angles? (per-angle listing)            -> marginal extremum near an axis?
 ├─ converged? (denser run, compare_runs, coverage windows)
 ├─ Fermi level? (fermi_level_scan: window opening within a few meV? same shift for all?)
 ├─ SOC? (compare_runs against the SR reference)
 ├─ breakdown? breakdown_gaps -> open_orbit_contacts -> breakdown_network (vs chance)
 ├─ mixing / harmonics? weak_peak_analysis (combination test + parent contact check)
 └─ none -> report unexplained, with what was excluded and which measurement would decide
```

## Ask the user at the start

1. The current calculation (and archived older runs), BXSF files, the pyskeaf driver, and the cell.
2. Experimental data in any form, with provenance (raw, transcribed, digitised).
3. Measured directions, the rotation plane, the angle convention, measurement days.
4. The deliverable and its audience.
5. The compute budget (cores; may new runs or the network be launched?).
