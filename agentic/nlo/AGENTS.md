# Agent instructions: non-linear optics workflow

You are running the non-linear optics workflow: comparing PAOFLOW's second-order responses (SHG
susceptibility and d coefficients, shift current, injection currents, magnetic responses; 3D or
2D sheet values) with experimental data and published calculations in any form.

1. **Read before acting:** `AGENT_WORKFLOW.md` and `LESSONS_LEARNED.md`. Follow them.
2. **Use the library:**
   - `PAOFLOW.agentic.nlo`: `NonlinearModel(output_dir, two_d, slab_height_nm, system)` (spectra,
     `tensor(kind, energy)`, `static`, `peak`, `onset`, `symmetry_check(point_group, kind, energy)`,
     `kleinman_check`, `d_coefficient`, `effective_3d`, `describe`); `symmetry`
     (`point_group_operations`, `forbidden_components`, `independent_relations`, `symmetrize`,
     `symmetry_error`, `kleinman_error`).
   - The shared toolkit `PAOFLOW.agentic.thermo_optics`: `ingest`, `digitize`, `datasets`,
     `units`, `metrics`, `plotting`.
   - The model is an output folder of `PAOFLOW.nonlinear_optics` (`examples/nlo_examples/<System>`
     layout). New runs are `main.py` edits (NK, ETA, SCISSOR, COMPONENTS, `twoD`) run into a new
     `--outputdir`/`output_<label>`; never into an existing output. `nonlinear_optics` needs `HRs`
     (call it right after `pao_hamiltonian`), 10 meV degeneracy tolerance for DFT Hamiltonians,
     and, for ultrasoft / PAW inputs, the augmented projections of October 2026 or later (`LESSONS_LEARNED.md` #2) for symmetric results.
   Every study has `digitize_sources.py`, `analysis.py`, `plot_figures.py` from `templates/`
   (figures: `agentic/thermo_optics/templates/digitize_template.py`). Inputs are Python
   arguments; no config files.
3. **Where to write:** only inside the study folder; never into calculation or source folders; do
   not modify `src/PAOFLOW/` or this folder unless asked (propose library changes with a test).
4. **Non-negotiable:**
   - never overwrite calculation output; ask before runs above ~10 min of CPU (a 60³ GaAs run is
     50 min on 8 ranks);
   - every experimental number carries its provenance and uncertainty, and its *convention*
     (d vs χ, the component and orientation, the photon energy, 3D vs sheet, the reference
     material of the measurement);
   - every digitisation has an overlay you have looked at;
   - every metric is quoted with a baseline;
   - every comparison is preceded by the symmetry audit of the run (forbidden components,
     relations between equivalent components, Kleinman at ω → 0): their leakage is the error bar;
   - scissor, broadening and k-grid are stated with every model number;
   - quote every number with its file; correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`; do what they ask and report the
   **Check** line.
