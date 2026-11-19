# Prompts to start and steer the SSCHA workflow

Copy a prompt, replace the `<...>` placeholders, and paste it to the agent opened in `agentic/sscha/`.

| # | File | When |
|---|---|---|
| 00 | `00_kickoff.md` | start a new study (intake + harmonic picture) |
| 01 | `01_plan.md` | supercell, k-grid, electrons, populations, temperatures, launcher, cost |
| 02 | `02_harmonic_start.md` | the ph.x run of the starting matrices and the harmonic dispersion |
| 03 | `03_drivers_and_launch.md` | write the drivers and launch |
| 04 | `04_monitor.md` | progress of every run |
| 05 | `05_diagnose.md` | a run does not converge, fails or looks odd |
| 06 | `06_hessian_and_stability.md` | the verdict at each temperature, fine-mesh check, dispersions |
| 07 | `07_temperature_scan.md` | bracket the stability temperature, compare with experiment |
| 08 | `08_downstream.md` | thermal cells and electron-phonon with the SSCHA force constants |
| 09 | `09_figures_and_report.md` | figures and the report |
| 10 | `10_challenge_and_review.md` | make the agent re-test its claims |
| 11 | `11_resume_session.md` | continue in a new session |

**Control phrases**: "Ask before launching more than <N> core-hours.", "Write only to <folder>.", "Quote every number
with the file it comes from.", "Give the settings next to every frequency.", "Written for <audience>."
