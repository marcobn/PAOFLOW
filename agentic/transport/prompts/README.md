# Prompts to start and steer the transport workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent (e.g. Claude Code) opened in `agentic/transport/`. Each prompt says when to use it, what the agent should do, and what you should check in its answer. For figures (vector PDF, scans, decks), the ingestion prompts of `agentic/thermo_optics/prompts/` (04, 05, 06) apply unchanged.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + physics map only) |
| 01 | `01_intake_and_physics_map.md` | revise the plan after answers or new inputs |
| 02 | `02_data_tables_and_text.md` | data as tables, numbers in text |
| 03 | `03_data_figures.md` | figures (vector or raster), with the transport conventions |
| 04 | `04_condition_alignment.md` | model at the experimental conditions (adapter, Hall factor, ρ₀, doping) |
| 05 | `05_comparison_and_baselines.md` | metrics with baselines, exponents, slopes, Bloch-Grüneisen |
| 06 | `06_boltzmann_channels.md` | a Boltzmann run with impurity / residual / e-e channels |
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
