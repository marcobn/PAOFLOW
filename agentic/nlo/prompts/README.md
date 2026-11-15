# Prompts to start and steer the non-linear optics workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent opened in `agentic/nlo/`. For tables, vector PDF figures, scans and decks the ingestion prompts of `agentic/thermo_optics/prompts/` (02–06) apply unchanged.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + symmetry audit + physics map) |
| 01 | `01_intake_and_physics_map.md` | revise the plan after answers or new inputs |
| 02 | `02_data_coefficients.md` | SHG coefficients, d values, spectra, photocurrents as tables, text or figures |
| 03 | `03_symmetry_audit.md` | point-group constraints, leakage, equivalent components, Kleinman |
| 04 | `04_condition_alignment.md` | model at the measured conditions (convention, photon energy, scissor, 3D/sheet) |
| 05 | `05_comparison_and_baselines.md` | metrics with baselines and the symmetry error bar |
| 06 | `06_diagnosis.md` | causes of discrepancies, proposed runs |
| 07 | `07_figures.md` | publication figures |
| 08 | `08_report.md` | the report |
| 09 | `09_challenge_and_review.md` | make the agent re-test its claims |
| 10 | `10_new_model_run.md` | a new model run arrived |
| 11 | `11_resume_session.md` | continue in a new session |

**Control phrases**: "Ask before any run longer than <N> minutes.", "Write only to <folder>.", "Quote every number with the file it comes from.", "Give the baseline next to every metric.", "Show me the overlay before writing the dataset.", "Written for <audience>."
