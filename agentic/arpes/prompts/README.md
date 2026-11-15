# Prompts to start and steer the ARPES workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent (e.g. Claude Code) opened in `agentic/arpes/`. Each prompt says when to use it, what the agent should do, and what you should check in its answer. For tables, vector PDF figures, scans and decks, the ingestion prompts of `agentic/thermo_optics/prompts/` (02–06) apply unchanged.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + physics map only) |
| 01 | `01_intake_and_physics_map.md` | revise the plan after answers or new inputs |
| 02 | `02_data_dispersions.md` | dispersions, widths, Σ, λ, linewidths as tables, text or figures |
| 03 | `03_data_band_map_image.md` | a band-map image to calibrate and analyse |
| 04 | `04_self_energy_extraction.md` | extract Σ and λ the same way on both sides |
| 05 | `05_condition_alignment.md` | model at the experimental conditions (k_F, E_F, resolution, T) |
| 06 | `06_comparison_and_baselines.md` | metrics with baselines |
| 07 | `07_diagnosis.md` | causes of discrepancies, proposed runs |
| 08 | `08_figures.md` | publication figures |
| 09 | `09_report.md` | the report |
| 10 | `10_challenge_and_review.md` | make the agent re-test its claims |
| 11 | `11_resume_session.md` | continue in a new session |
| 12 | `12_new_model_run.md` | a new model run arrived |

**Control phrases** you can add to any prompt:
- "Ask before any run longer than <N> minutes." / "You may use up to <N> cores."
- "Write only to <folder>."
- "Quote every number with the file it comes from."
- "Give the baseline next to every metric."
- "Show me the overlay before writing the dataset."
- "Written for <audience>." (sets the tone of reports and READMEs)
