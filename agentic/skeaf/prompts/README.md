# Prompts to start and steer the agentic workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent (e.g. Claude Code) opened in `agentic/skeaf/` (your study goes in `projects/<system>/`). Each prompt says when to use it, what the agent should do, and what you should check in its answer.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new system |
| 01 | `01_intake_and_geometry.md` | after kickoff, or when inputs change |
| 02 | `02_experiment_table.md` | experimental data as tables (CSV, XLSX, paper tables) |
| 03 | `03_experiment_text.md` | experimental data as free text (notes, slide text, e-mails) |
| 04 | `04_experiment_image.md` | experimental data as images (FFT-vs-angle maps, spectra), incl. pptx/pdf decks |
| 05 | `05_sweep_and_assignments.md` | main comparison over a rotation; branch assignments |
| 06 | `06_directions_and_masses.md` | single directions, anisotropy ratios, 1 + λ |
| 07 | `07_new_run_or_convergence.md` | a new or denser calculation arrived |
| 08 | `08_fermi_level_scan.md` | test E_F sensitivity (requests or analyses shifted runs) |
| 09 | `09_soc_vs_sr.md` | compare SOC and scalar-relativistic runs |
| 10 | `10_missing_or_empty_band.md` | a band is missing or its file is empty |
| 11 | `11_breakdown.md` | magnetic-breakdown analysis (gap map → open check → network) |
| 12 | `12_weak_peaks_and_mixing.md` | are weak peaks breakdown or frequency mixing? |
| 13 | `13_collaborator_package.md` | produce a self-contained package for experimental collaborators |
| 14 | `14_challenge_and_review.md` | make the agent re-test its own claims |
| 15 | `15_resume_session.md` | continue in a new session |
| 16 | `16_answer_collaborator_questions.md` | answer questions coming back from collaborators |

**Control phrases** you can add to any prompt:
- "Ask before any run longer than <N> minutes." / "You may use up to <N> cores."
- "Do not modify <folder>; write only to <out_dir>."
- "Smoke-test first (one angle, low numint), then show me the estimate."
- "Quote every number with the file it comes from."
- "Give the chance baseline for every match rate."
- "Written for <audience>." (sets the tone of reports and READMEs)
