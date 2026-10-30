# Prompts to start and steer the thermo-optics workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent (e.g. Claude Code) opened in `agentic/thermo_optics/`. Each prompt says when to use it, what the agent should do, and what you should check in its answer.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + physics map only) |
| 01 | `01_intake_and_physics_map.md` | revise the plan after answers or new inputs |
| 02 | `02_data_table.md` | data as tables (CSV, XLSX, paper tables) |
| 03 | `03_data_text.md` | numbers quoted in text |
| 04 | `04_data_pdf_vector.md` | figures drawn as PDF vectors |
| 05 | `05_data_raster.md` | scanned figures, embedded bitmaps, images |
| 06 | `06_data_slide_deck.md` | slide decks (pptx or PDF) |
| 07 | `07_condition_alignment.md` | model at the experimental conditions (adapter, interpolation) |
| 08 | `08_comparison_and_baselines.md` | metrics with baselines, derived quantities |
| 09 | `09_diagnosis.md` | causes of discrepancies, proposed runs |
| 10 | `10_figures.md` | publication figures |
| 11 | `11_report.md` | the report |
| 12 | `12_challenge_and_review.md` | make the agent re-test its claims |
| 13 | `13_resume_session.md` | continue in a new session |
| 14 | `14_new_model_run.md` | a new model run arrived |

**Control phrases** you can add to any prompt:
- "Ask before any run longer than <N> minutes." / "You may use up to <N> cores."
- "Write only to <folder>."
- "Quote every number with the file it comes from."
- "Give the baseline next to every metric."
- "Show me the overlay before writing the dataset."
- "Written for <audience>." (sets the tone of reports and READMEs)
