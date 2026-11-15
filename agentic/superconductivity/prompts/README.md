# Prompts to start and steer the superconductivity workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent (e.g. Claude Code) opened in `agentic/superconductivity/`. Each prompt says when to use it, what the agent should do, and what you should check in its answer. For figures (vector PDF, scans, decks), the ingestion prompts of `agentic/thermo_optics/prompts/` (04, 05, 06) apply unchanged.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + physics map only) |
| 01 | `01_intake_and_physics_map.md` | revise the plan after answers or new inputs |
| 02 | `02_data_tables_and_text.md` | Tc, gaps, λ, ratios, α²F as tables or text |
| 03 | `03_data_figures.md` | Δ(T) curves, α²F spectra, tunnelling conductances, gap distributions as figures |
| 04 | `04_condition_alignment.md` | model at the experimental conditions (μ*, definition, T) |
| 05 | `05_comparison_and_baselines.md` | metrics with baselines (Tc table, BCS, inverted λ) |
| 06 | `06_diagnosis.md` | causes of discrepancies, proposed runs |
| 07 | `07_figures.md` | publication figures |
| 08 | `08_report.md` | the report |
| 09 | `09_challenge_and_review.md` | make the agent re-test its claims |
| 10 | `10_new_model_run.md` | a new model run arrived |
| 11 | `11_resume_session.md` | continue in a new session |

**Control phrases** you can add to any prompt:
- "Ask before any run longer than <N> minutes." / "You may use up to <N> cores."
- "Write only to <folder>."
- "Quote every number with the file it comes from."
- "Give the baseline next to every metric."
- "Show me the overlay before writing the dataset."
- "Written for <audience>." (sets the tone of reports and READMEs)
