# Lessons learned (Si absorption and emissivity, October 2026)

Read together with `AGENT_WORKFLOW.md`. Function names refer to `PAOFLOW.agentic.thermo_optics`.

Each entry: **symptom → cause → what to do**, with the Si instance. These are the reasons behind the rules in `AGENT_WORKFLOW.md`.

## Sources and provenance

| # | Symptom | Cause | What to do | Si instance |
|---|---|---|---|---|
| 1 | "Experiment" curves suspiciously smooth, theory matching onsets exactly | The figure is a *replot* in a later paper, and the theory was rigidly shifted | Name the original and the replot ("Macfarlane 1958, as replotted in Noffsinger 2012"); read the caption and text for shifts; record them in the header | Noffsinger Fig. 2: GW theory shifted by 0.15–0.23 eV onto the experimental onsets; used as a reference, not as an independent prediction. |
| 2 | Digitising a figure that is exactly available | The figure was vector-drawn | Run `ingest.pdf_drawing_summary` first; vector paths + `pdf_frame_ticks` give exact coordinates | Noffsinger Fig. 2: 20 curves exact to 10⁻¹³ pt; Fig. 3 of the same paper is a JPEG. |
| 3 | Extracted image sideways or blurry | Scanned PDFs embed rotated or low-resolution images | Render the page instead (`render_pdf_pages`, 500 dpi) | Sato's native images are rotated; 500 dpi renders used. |
| 4 | Wavelength axis inconsistent: a label sits one tick off a uniform scale | Hand-drawn 1967 axis; drawn spacing departs from log below 1 μm and above 10 μm | Piecewise axis through every tick (`Axis(..., piecewise=True)`). Decide ambiguous readings with a **physical anchor** and state the assumption | Left frame at 0.40 μm (stated range) vs 0.35 μm; visible emissivity 0.53–0.56 ≈ 1−R(0.40 μm) = 0.52, not 0.43. |
| 5 | A curve label contradicts its position | Misprint, or OCR-like misreading of the scan | Infer from the ordering and from other figures; record both readings | "473 K" between 643 and 693 K; Fig. 9 dots at 673 K confirm it. |
| 6 | Unlabelled or coincident curves | Labels placed between curves; two curves drawn as one heavy line | Assign by a **physical ordering** (here: long-λ emissivity rises with T), say so; give coincident curves the shared line | 873 and 1073 K share the λ > 1 μm line. |
| 7 | Dashed curves entangled with label arrows and braces | Hand lettering over the data | Do not force an assignment: keep a band (temperature range) and say "not resolved" | Sato visible branches 0.4–0.9 μm. |
| 8 | Model and experiment disagree for a trivial reason | Different geometry (thickness, angle) or sample (doping) | Recompute the observable at the experimental geometry; note the sample | Model default slab 500 μm vs Sato 1.77 mm; n-type 15 Ω cm vs intrinsic model. |

## Digitisation

| # | Symptom | Cause | What to do | Si instance |
|---|---|---|---|---|
| 9 | Open circles found as two or three blobs | JPEG compression breaks thin rings | `ring_markers` (annulus template), merge plateau duplicates | Noffsinger Fig. 3: 39 circles. |
| 10 | Column tracer loses steep segments | `trace_curve` assumes one y per column | `trace_path` (arc-length), several seeds per curve | Fan Fig. 4: 6 seeds. |
| 11 | Tracer jumps to a neighbouring dashed curve | Dash gaps larger than the curve spacing; straight continuation lands on the neighbour | `dash_segments` + `link_dashes` (chain dashes by direction with a lateral penalty) | Sato: five dashed branches 4–15 px apart. |
| 12 | Line trace stops at a marker | A dot attached to the line deflects the tracer onto its edge | Trace on the ink with **and** without the markers (erased disks), union the traces | Sato Fig. 8: lower-left ends missing until both traces were used. |
| 13 | `find_frame` finds one line only | Skewed scan, or a bent hand-drawn axis | `skew_tolerance`, row/column projections; detect ticks relative to the local frame line | Sato Fig. 8 bottom line bends by 15 px. |
| 14 | Extra "markers" on labels | Thick glyph strokes survive the opening | `assign_to_curves(..., max_distance)`: glyphs lie 40–70 px from every line | Sato Fig. 5: 12 glyph blobs rejected. |
| 15 | Trace stops at a sharp peak | Turn per step above `max_turn_deg` | `max_turn_deg` 80, step 2, extra seeds after the peaks | Sato 543/623 K lattice peaks. |
| 16 | A few markers missing | The traced line does not reach the curve end | Count markers per curve against the source; add a seed near the end | Sato Fig. 8: 7/8 → 8/8 and 6/7 → 7/7. |

## Comparison

| # | Symptom | Cause | What to do | Si instance |
|---|---|---|---|---|
| 17 | The comparison flips sign between neighbouring model temperatures | Strongly activated observable between coarse model temperatures | Interpolate in 1/T (ln y), quote both bracketing biases; propose a run at the measured T | Emissivity at 623 K: bracketing biases −0.13 / +0.29. |
| 18 | Two "model α" disagree by 15–20 % | α from κ (n(ω)) vs the driver's constant-n α | State which; the difference measures the n error | n(ω) = 4.0–4.2 vs n_r = 3.4. |
| 19 | Onsets and edges kinked; ε₁ rises at long λ and high T | 0.05 eV photon grid; Kramers–Kronig starting at 0.05 eV misses the low-ω carrier weight | Quote onsets "within one grid step"; flag long-λ high-T model values; propose a finer grid and a lower start | ε₁(0.05 eV) 15.1 → 20.9 from 50 to 1200 K. |
| 20 | Baseline rms = 0 | Constant baseline on a flat spectrum | Void it (no shape to predict); use another baseline | Sato 873/1073 K. |
| 21 | Huge n in a metric | Dense vector or traced curves | Point counts measure digitisation density, not statistics: never derive standard errors from n | 1100–1500 points per Noffsinger curve. |
| 22 | Many discrepancies, many candidate causes | One underlying offset propagates | Look for one cause explaining several (n excess → opaque emissivity, long-λ saturation, α from κ) | ε∞ = 15.1 vs 11.7. |
| 23 | Proposing a run to test a parameter already stored | Not reading the output | Check sensitivities in the existing output first | α vs η in `absorption.npz`: < 0.2 % change, so η ruled out. |
| 24 | A global rms says "bad" where the model is right in part of the range | Mechanisms differ across the range | Physics windows; mark out-of-scope mechanisms | Multiphonon bands at 6–15 μm absent from the model; the free-carrier window agrees within 0.03–0.13 dex. |
| 25 | A run looks 2.5× better or worse than another in α | One curve rigidly shifted onto the data, the other not; on a steep edge a 0.06 eV onset offset is −0.4 dex | Apply the same diagnostic (`best_shift`, then the median ratio) to every run before comparing magnitudes | PBE+scissor "×0.41" was ×0.72–0.77 after a −0.06 eV shift, the same as eACBN0; the Levine–Allan explanation was withdrawn. |
| 26 | Fixing the gap with a scissor degrades another observable | The model's error is not rigid in k (eACBN0: direct gap right, indirect 0.22 eV high); one conduction-band shift moves the direct transitions too | Check which gaps the scissor should move; shift the external direct ε only by the thermal part when its direct gap is already right | eACBN0+scissor: ε₁(0.05 eV) 12.4 → 14.0, n −7…+4 % → +4…+11 %; with `direct_scissor='thermal'` 12.4 and −5.5…+4.7 %, opaque emissivity −0.05 → −0.025 at 1073 K. |
| 27 | A quantity that should be monotonic in a model parameter is not | A numerical artifact in the transformation the parameter enters, not physics | Scan the parameter on the stored output before interpreting; find the term that breaks monotonicity | ε₁(0.05 eV, 1073 K) fell from 14 to 11 as the KS scissor grew more negative: ((ω−Δ)/ω)² blew up the broadening tail of the external ε₂ (fixed; 6–15 μm emissivity 0.01–0.02 lower). |
| 28 | dn/dT too high or too low depending on a convention | The scissor velocity convention scales ε₂ by ((ω−Δ)/ω)² (KS) or not (Levine–Allan), and the T-dependent Δ then sets dn/dT | Recompute n(T) with both conventions on the stored output; quote the experiment as bracketed | Thermal shift of the eACBN0 direct ε: KS ×1.1–1.8, renormalized ×0.65–0.9 of Sato's dn/dT. |

## Figures

| # | Symptom | Cause | What to do |
|---|---|---|---|
| 29 | Figure becomes extremely wide | Long one-line provenance note enlarges the tight bounding box | `provenance_note` wraps to the figure width |
| 30 | Direct labels missing | Placed at curve ends outside the axes | Place them at a level crossing inside the frame (`threshold_crossing`), else at the frame edge |
| 31 | Shaded sensitivity bands swamp the panel | Bracketing curves differ by more than the signal | Drop the bands, quote the sensitivity in the table and caption |
| 32 | Seven temperatures in one panel | More than the ramp's 5 distinguishable steps | Facet (543–693 K / 743–1073 K) |
| 33 | Legend on top of data or panel labels | Default placement | Panel labels above the frame; style legend in an empty region or above the axes; look at the PNG |
