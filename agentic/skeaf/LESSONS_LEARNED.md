# Lessons learned (NbSn2, October 2026)

Read together with `AGENT_WORKFLOW.md`. Method names refer to `PAOFLOW.qo_analysis`.

Each entry: **symptom → cause → what to do**, with the NbSn2 instance. These are the reasons behind the rules in `AGENT_WORKFLOW.md`.

## Data handling and provenance

| # | Symptom | Cause | What to do | NbSn2 instance |
|---|---|---|---|---|
| 1 | Old raw results disappear | pyskeaf overwrites `output/qo_*` with fixed names on every run | Write each run into a new folder (`skeaf_runs.skeaf_rotation` refuses non-empty ones). Keep per-run analysis folders. | 49³/200 raw data overwritten by 81³/320; later found again as `output.48/`. |
| 2 | "New orbit appeared with the denser grid" | Compared against a coarse *listing* that sampled other angles, not against the old raw data | Before calling anything new, check the old raw files at all angles. | The 1407–1487 T band-7 piece was already in 49³/200; the claim was corrected in the report. |
| 3 | A band yields an empty qo file | The production job died before writing (output is written only after the last angle) | `single_angle_check`; compare the band's BXSF with its scalar-relativistic parent; ask for a re-run | SOC bands 17/18 empty; the local check gave 283/516 T; the re-run confirmed it. |
| 4 | Collaborators ask about data that changed | The user renamed or tarred the delivered folder | Update the renamed copy; tell the user to re-tar | `NbSn2_QO_summary_for_collaborators` → `NbSn2_SOC_summary` + `NbSn_soc.tar`. |
| 5 | Experimental numbers treated as raw | They were digitised from a slide image or transcribed from slide text | State provenance in CSV headers and in a caveat box in every deliverable | All torque data came from a pptx. |

## Geometry and units

| # | Symptom | Cause | What to do | NbSn2 instance |
|---|---|---|---|---|
| 6 | Directions off by tens of degrees | Real-space [hkl] used instead of the (hkl) normal, or vice versa | Decide the convention explicitly; print both; check against statements like "[110] is ~30° off [100]" | Reciprocal [110] → θ = 29.95°; real space → 60.05°. |
| 7 | Experimental index label inconsistent with the cell | Cubic-style estimate in the source | Use `field_geometry.field_angles_to_direction` and report the correct indices | "[11y], y ≈ 2.4" at 45° is really [1 1 3.90]; [1 1 2.4] would be 58.4°. |
| 8 | Energies wrong by 13.6× | BXSF energies are in Ry, `fermi_energy` in eV | Check with the Onsager slope (`fermi_level_scan`); read the pyskeaf docstrings | A "4 meV" band difference was really 4 mRy (≈ 54 meV); corrected in the report. |
| 9 | Requested angle not on the grid | The sweep step (90/89°) doesn't hit 45° | Use the nearest grid point and say which | 45° → 44.49 / 45.51°. |

## Statistics and interpretation

| # | Symptom | Cause | What to do | NbSn2 instance |
|---|---|---|---|---|
| 10 | "Good agreement": many peaks within 5 % | Dense spectrum; chance alone matches 30–40 % | Always quote the log-uniform **and** angle-shuffled baselines | 44 % vs 40 ± 4 % (SOC); 34 % vs 36 ± 4 % (SR). |
| 11 | A ratio test says "anisotropy reproduced" | The partner orbit exists at a single angle only | Check that the calculated branch is continuous over the rotation | δ 0°/45° ratio rested on an isolated 1177 T point. |
| 12 | Template branch labels: "cylinder-like", "UNRELIABLE" | Nearest-frequency continuation hops between branches on dense 0–90° sweeps | Use the overlay and the per-angle listing instead | Many SR/SOC sheets mislabelled. |
| 13 | Template recommends a rigid E_F shift "with no casualties" | Fits 2–3 peaks per direction; the "degraded" guard only fires below 3 % error | Check every peak; require the same shift in all directions | +39 / +9 / +34 meV; β1 3.1 % → 20 %. Rejected. |
| 14 | Template crash in the cross-direction stage | No common candidate group → empty table without columns | Treat empty as a result (fixed in `orbit_matching.anisotropy_candidates`) | [11y]/[110] pair at 81³/320. |
| 15 | The NumOrbCopy cut removes good orbits | The histogram is not bimodal at numint 200–320 | Justify the cut (paired, smoothly tracking orbits at 50–90 copies); test at 100 | Cut 50; conclusions identical at 100. |
| 16 | "This band only has maxima" assumption | Minima (necks) are extremal orbits too | Use the `extremum` column; the tips of pockets are not orbits | 32 % of SOC orbits are minima; β1/β2 at [001] are necks of the band-7 sheet. |
| 17 | Doubled frequencies interpreted as spin up/down | With SOC + inversion, Kramers pairs are degenerate bands with the same Fermi surface | Count one band per pair; residual differences are numerical error | Bands 11/12 repeat each other within 0.3–4 %. |
| 18 | Equal frequencies repeated within one band | Symmetry-equivalent orbits at different k | Expected; they are distinct pockets, not artefacts | 22.95 T twice in band 11 (k_z = 0 and the zone boundary). |
| 19 | Small pockets disagree after refinement | Pockets < 50 T are the least converged quantity (1–30 % drift) | Mark them provisional; don't build λ arguments on them | γ dome moved 21 → 24 → 17–19 T across grids/SOC. |

## Magnetic breakdown and weak peaks

| # | Symptom | Cause | What to do | NbSn2 instance |
|---|---|---|---|---|
| 20 | Sorted bands "touch" without SOC | True band crossings; sorted indices stitch two physical bands | Gap map without and with SOC; SOC opens gaps | Band-7 contacts: SR ≈ 0.001 Å⁻¹, SOC 0.004–0.01 Å⁻¹. |
| 21 | First-order breakdown "explains" peaks | Too many combinations; extremal pieces need not coincide | Slice-level stationary combined areas, with chance baselines | 0–1/90 matched vs 0–2 % chance. |
| 22 | A dense ladder of network frequencies | An extended near-contact discretised into point junctions | Treat statistically; report it as an artefact | Bands 6/7: 25–52 frequencies per angle, ~5 T apart. |
| 23 | Weak peaks "overlap" breakdown orbits by eye | Density, again | Angle-shuffled test | 10 % vs 13 ± 2 %; no excess. |
| 24 | Weak peaks are combinations of strong ones (4σ) | Mixing *or* breakdown | Parent check: breakdown needs touching sheets | 0 of 12 combinations on touching sheets → mixing. |
| 25 | The weak-peak population contains regular runs | Contour-image striping | `experiment_data.regular_peak_runs`; ask for the raw FFTs | 5° and 65° runs (11 T, 16 T spacing); the 70° ladder. |
| 26 | "Breakdown is certainly active" | Small B₀ shows that breakdown is *expected*, not *demonstrated* | Phrase carefully; propose field-window FFTs | README wording corrected after the collaborators' question. |

## Process

| # | Lesson |
|---|---|
| 27 | The most productive diagnostic was the plain per-angle side-by-side listing of experimental and calculated frequencies; build it first. |
| 28 | The decisive evidence came from continuous rotations in **both** theory and experiment, not from more single directions. |
| 29 | Each "fix" (denser grid, E_F, SOC, breakdown) can move different sheets in different directions. Report what improved **and** what got worse. |
| 30 | When the user questions a statement, re-test it quantitatively before answering, and update the deliverable if the wording was too strong. |
| 31 | A synthetic test with a known answer (two spheres a known gap apart) exposed a bug that the real-data run hid: network orbits through a closed contour with a **single** junction lost that contour's area. Every new physics routine needs a synthetic test with an analytic answer, not only a regression test on real data. |
| 32 | Code run through joblib/loky starts fresh interpreters: anything patched in-process (paths, monkeypatches) is missing in the workers. Validate parallel code the way it will really be run. |
