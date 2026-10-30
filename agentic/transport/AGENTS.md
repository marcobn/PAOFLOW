# Agent instructions: transport workflow

You are running the transport workflow: comparing PAOFLOW's electron-phonon transport (semiconductors: phonon-limited mobilities, carrier densities, conductivity, Seebeck; metals: resistivity from α²F_tr with residual and electron-electron channels, plasma frequency, infrared optics, thermal emissivity; and the Boltzmann bridge with impurity channels) with experimental data in any form (tables, text, PDF figures, slide decks, images).

1. **Read before acting:**
   - `AGENT_WORKFLOW.md` (the model pipeline, operating rules, Stages 0–8, decision tree);
   - `LESSONS_LEARNED.md` (pitfalls).
   Follow them; do not re-derive the method.
2. **Use the library:**
   - `PAOFLOW.agentic.transport`: the adapters `MobilityModel` (`transport/<tag>/mobility.dat`) and `ResistivityModel` (`transport/<tag>/resistivity.dat`, `emissivity.npz`, the `eliashberg.npz` scalars), both opened with `.from_system(<system folder>, output=<tag>)`; `baselines` (Bloch-Grüneisen, power-law and linear fits, the Matthiessen residual offset, Matthiessen's mobilities); `channels` (channel lists for `PAOFLOW.transport(scattering_channels=...)`: e-ph + residual + Fermi liquid for metals, energy-resolved e-ph + impurities for semiconductors).
   - The shared toolkit `PAOFLOW.agentic.thermo_optics`: `ingest`, `digitize`, `datasets` (canonical CSV with provenance), `units`, `metrics`, `plotting`.
   - The model is an electron-phonon **system folder** (`examples/elphon/<System>` layout: `system.py`, `transport/`, `superconductivity/`, `cache/`). New runs are driver commands of that folder (`transport/mobility.py --tag ...`, `superconductivity/eliashberg.py --transport`, `transport/resistivity.py ...`), always with `--tag`/a new output so earlier outputs are kept; `transport/resistivity.py` recomputes ρ(T), the optics and the emissivity for other channels (`--rho0`, `--b-ee`) or geometry in seconds, without a vertex run (see the pipeline table in `AGENT_WORKFLOW.md`).

   Every study has three scripts, started from `templates/` (and from `agentic/thermo_optics/templates/digitize_template.py` for figures): `digitize_sources.py` (sources → `data/*.csv`), `analysis.py` (model at experimental conditions → `data/model_curves.csv`, `tables/`) and `plot_figures.py` (CSV only → `figures/`). Inputs are Python arguments; do not use config files.
3. **Where to write:**
   - Write only inside the study folder (`projects/<study>/`, or the folder the user names).
   - Never write into calculation folders or into the source folders (papers, decks).
   - Do not modify `src/PAOFLOW/` or the files in this folder unless the user asks.
   - If the library needs a fix or a new tool, propose it with a test.
4. **Non-negotiable:**
   - never overwrite calculation output;
   - ask before new calculations or anything above ~10 min of CPU;
   - every experimental number carries its provenance (`measured`, `transcribed`, `vector-extracted`, `digitised`) and an uncertainty;
   - every digitisation has an overlay check you have looked at;
   - every metric is quoted with a baseline;
   - every model value off the computed grid is labelled interpolated;
   - compare like with like: drift vs Hall mobility (the Hall factor), phonon-limited vs doped samples (Matthiessen with an impurity channel), ρ(T) with and without the residual resistivity, the same temperature window;
   - quote every number with its file;
   - correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`. If they paste one, do what it asks and report what its **Check** line names.
