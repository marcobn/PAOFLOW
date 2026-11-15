# Agent instructions: superconductivity workflow

You are running the superconductivity workflow: comparing PAOFLOW's electron-phonon superconductivity (the Eliashberg function and its moments, McMillan / Allen–Dynes / Migdal–Eliashberg critical temperatures, isotropic and anisotropic gaps versus temperature, gap distributions of multi-gap metals, quasiparticle densities of states, mass enhancements) with experimental data in any form (tables, text, PDF figures, decks, images).

1. **Read before acting:**
   - `AGENT_WORKFLOW.md` (the model pipeline, operating rules, Stages 0–8, decision tree);
   - `LESSONS_LEARNED.md` (pitfalls).
   Follow them; do not re-derive the method.
2. **Use the library:**
   - `PAOFLOW.agentic.superconductivity`: the adapters `EliashbergModel` (`superconductivity/output/eliashberg.npz`: α²F, α²F_tr, λ, ω_log, ω₂, Tc for any μ*, cumulative and windowed λ, mode-resolved λ_qν), `GapModel` (`output/me/migdal_eliashberg.npz`: Δ(T), gap edges, Z, Tc, gap ratio, quasiparticle DOS, real-axis solutions) and `AnisotropicGapModel` (`output/me_aniso/migdal_eliashberg_aniso.npz`: Δ_nk(T), distributions and their peaks, sheet averages, λ_nk), all with `.from_system(<system folder>, output=...)`; `baselines` (moments, McMillan and Allen–Dynes Tc, `lambda_from_tc`, BCS gap and `bcs_gap_vs_t`, the strong-coupling gap ratio).
   - The shared toolkit `PAOFLOW.agentic.thermo_optics`: `ingest`, `digitize`, `datasets` (canonical CSV with provenance), `units`, `metrics`, `plotting`.
   - The model is an electron-phonon **system folder** (`examples/elphon/<System>` layout: `system.py` with the `SUPERCONDUCTIVITY` defaults, `superconductivity/`, `cache/`). New runs are driver commands of that folder (`superconductivity/eliashberg.py ...`, `me.py ...`, `me_aniso.py ...`); the drivers write `output/`, `output/me/`, `output/me_aniso/`: rename the previous folder to `<name>_<label>` before a rerun and open it with `output='<name>_<label>'`.

   Every study has three scripts, started from `templates/` (and from `agentic/thermo_optics/templates/digitize_template.py` for figures): `digitize_sources.py` (sources → `data/*.csv`), `analysis.py` (model at experimental conditions → `data/model_curves.csv`, `tables/`) and `plot_figures.py` (CSV only → `figures/`). Inputs are Python arguments; do not use config files.
3. **Where to write:**
   - Write only inside the study folder (`projects/<study>/`, or the folder the user names).
   - Never write into calculation folders or into the source folders (papers, decks).
   - Do not modify `src/PAOFLOW/` or the files in this folder unless the user asks.
   - If the library needs a fix or a new tool, propose it with a test.
4. **Non-negotiable:**
   - never overwrite calculation output;
   - ask before new calculations or anything above ~10 min of CPU (a dense Eliashberg run or an anisotropic Migdal–Eliashberg solution is one);
   - every experimental number carries its provenance (`measured`, `transcribed`, `vector-extracted`, `digitised`) and an uncertainty;
   - every digitisation has an overlay check you have looked at;
   - every metric is quoted with a baseline;
   - μ* is a stated parameter: quote Tc for the run's μ* and for 0.10–0.13, never tune it silently; a λ inverted from the measured Tc is a diagnostic;
   - gaps are compared with the same definition (order parameter Δ(iω₀), spectral gap edge, coherence-peak position) and at the same temperature;
   - every model value off the computed grid is labelled interpolated;
   - quote every number with its file;
   - correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`. If they paste one, do what it asks and report what its **Check** line names.
