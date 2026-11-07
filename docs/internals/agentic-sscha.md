# Agentic SSCHA: anharmonic phonon campaigns

`PAOFLOW.agentic.sscha` serves the agentic workflow `agentic/sscha/`: an AI agent plans, launches, monitors and
interprets SSCHA campaigns ([Anharmonic phonons at temperature](sscha.md)) for any crystal on any machine. The
engine is PAOFLOW's SSCHA interface driven by the `main.sscha.py` drivers of `paoflow-gen` (launchers `local`,
`slurm`, `manual`); this package reads the run directories only (no python-sscha import), so runs can be monitored
wherever their directories are visible. The playbook, lessons learned, prompts and templates live in
`agentic/sscha/`.

Unlike the comparison workflows (transport, superconductivity, ARPES, non-linear optics), this one also *drives*
calculations: its stages go from the harmonic picture and the plan (supercell, k-grid, electrons, populations,
temperatures, cost) through launch, monitoring and diagnosis to the stability verdicts, the stability temperature
against experiment and the hand-over of the force constants to the thermal cells and the electron–phonon workflows.

## Modules

| module | role |
|---|---|
| `run` | `SSCHARun(sscha_dir)`: `temperature`, `electronic_temperature`, `supercell`, `status()` (stage and next step, through `phonon.sscha_bridge.run_status`), `history()`, `hessian()`, `lowest_hessian()`, `force_constants(kind)`, `failed_configurations()`, `diagnostics(fine_mesh)` (codes `not_converged`, `kong_liu_every_population`, `free_energy_drift`, `frequencies_moving`, `failed_configurations`, `unstable`, `unstable_between_grid_points`, `cold_electrons`), `describe()`; `find_runs(root)` |
| `scan` | `scan_table(runs, fine_mesh)`; `signed_square`; `stability_temperature(T, softest)`: zero of ω\|ω\| interpolated in the bracketing pair, local fit ω² = a(T − T_s) over the pair and one neighbour each side, `extrapolated` without a bracket; `next_temperature(T, softest, tolerance_k)`: safeguarded secant step inside the bracket, or the fit's estimate limited to twice the computed range |
| `dispersion` | `phonon_model(dyn_prefix)` (PAOFLOW's q2r/matdyn: atom-pair Wigner–Seitz images, simple ASR), `dispersion(dyn_prefix, path, labels, npoints)`, `grid_minimum(dyn_prefix, mesh)` |
| `plotting` | `plot_convergence(run)`, `plot_scan(rows, result, experiment)`, `plot_dispersions(curves, reference)`; the style of the thermo-optics toolkit (`publication_style`, `save_figure`) |

## Design choices

- **State from files.** Every decision reads `state.json` and the population folders (`run_status`), the same
  function `PAOFLOW.sscha_status` uses; log greps gave false alarms in the δ-TaN campaign.
- **The Hessian decides, on a fine mesh.** The auxiliary matrices are positive by construction; `diagnostics` and
  `scan_table` evaluate the exported Hessian matrices on a mesh finer than the supercell grid.
- **The stability temperature is a bracket.** The mean-field soft-mode law holds only near T_s; the estimate comes
  from the two converged runs that bracket it, the local fit is its consistency check, and an extrapolation is
  flagged as such.
- **Machine-agnostic.** The launcher is a driver setting, the scheduler directives are plain lines, and the
  campaign template writes drivers without prompts (`build_sscha_script`).

## Tests

`tests/unit/agentic/sscha/`: the adapter on synthetic run directories (stages, history, Hessian, exported matrices,
`describe`), every diagnostic code, `find_runs` and `scan_table` with the δ-TaN matrices (fine-mesh softest modes
−223.4 and +100.3 cm⁻¹), the stability temperature of an exact soft-mode law (bracket, local fit, extrapolation,
next temperature), dispersions (X frequencies of the SSCHA matrices) and the figures.
