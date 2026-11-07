# Agent instructions: SSCHA workflow

You are running the SSCHA workflow: anharmonic phonons at temperature with PAOFLOW (stability of harmonically
unstable lattices, the stability temperature against experiment, temperature-dependent force constants for the
thermal cells and the electron-phonon coupling), for any crystal, on any machine.

1. **Read before acting:** `AGENT_WORKFLOW.md` and `LESSONS_LEARNED.md`. Follow them.
2. **Use the library:**
   - The engine: `PAOFLOW.sscha_setup(temperature, supercell_matrix, harmonic_dynamical_matrices, sscha_dir, kgrid,
     population_sizes, hessian_size, electronic_temperature, start_from, ...)`, `sscha_status()`,
     `sscha_population()`, `sscha_minimize()`, `sscha_hessian()`, `sscha_force_constants('auxiliary'|'hessian')`;
     `special_displacement(..., force_constants_file='sscha')`; `ElphonSystem(phonon_source='sscha',
     sscha_dynamical_matrices=...)`.
   - The drivers: `paoflow-gen` workflow `sscha` writes `main.sscha.py` (phases `harmonic`, `run`, `step`,
     `status`; launchers `local`, `slurm`, `manual`). Non-interactively: `PAOFLOW.gen.sscha_driver.build_sscha_script
     (cfg)` (`templates/campaign_template.py`).
   - `PAOFLOW.agentic.sscha`: `SSCHARun(sscha_dir)` (`status`, `history`, `hessian`, `lowest_hessian`,
     `force_constants`, `failed_configurations`, `diagnostics(fine_mesh)`, `describe`), `find_runs`, `scan_table`,
     `stability_temperature`, `next_temperature`, `dispersion`, `grid_minimum`, `phonon_model`; `plotting`
     (`plot_convergence`, `plot_scan`, `plot_dispersions`, `publication_style`, `save_figure`).
   - Experimental data in any form: the toolkit of `PAOFLOW.agentic.thermo_optics` (`datasets`, `ingest`,
     `digitize`, `units`).
   Every study has `campaign.py`, `analysis.py`, `plot_figures.py` from `templates/`. Inputs are Python arguments;
   no config files.
3. **Where to write:** the study folder and the run directories you create for it; never into another campaign's
   run directories or the source folders; do not modify `src/PAOFLOW/` or this folder unless asked (propose
   library changes with a test).
4. **Non-negotiable:**
   - state the compute cost (configurations × cost of one pw.x run × populations) and ask before launching;
     never cancel, resubmit or delete jobs and run directories of another campaign;
   - read the state of a run from its files (`SSCHARun.status()`), never from log greps;
   - stability verdicts come from the free-energy Hessian, never from the auxiliary matrices (positive by
     construction), and are checked on a q-mesh finer than the supercell grid (`grid_minimum`);
   - every SSCHA number carries its settings: supercell, k-grid, electrons (parent occupations or Fermi–Dirac at
     which temperature), populations and sizes, the Hessian population size, the harmonic support;
   - every experimental number carries its provenance and uncertainty, and what was measured (transition,
     eutectoid, decomposition, metastability limit; on heating or cooling; composition);
   - on clusters, run in scratch and copy the run directories' `*.json` and `dyn*` files to permanent storage
     after every finished temperature;
   - quote every number with its file; correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`; do what they ask and report the **Check** line.
