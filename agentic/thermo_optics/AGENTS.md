# Agent instructions: thermo-optics workflow

You are running the thermo-optics workflow: comparing PAOFLOW's temperature-dependent optical response (absorption, free carriers, refractive index, emissivity; any material, and other observables through a new adapter) with experimental data in any form (tables, text, vector or scanned PDF figures, slide decks, images).

1. **Read before acting:**
   - `AGENT_WORKFLOW.md` (operating rules, Stages 0–8, decision tree);
   - `LESSONS_LEARNED.md` (pitfalls).
   Follow them; do not re-derive the method.
2. **Use the library** `PAOFLOW.agentic.thermo_optics`:
   - `ingest` (PDF inventory, renders, embedded images, words, vector paths, tables, decks);
   - `digitize` (calibration, masks, markers, tracing, dash linking, overlays);
   - `datasets` (canonical CSV with provenance);
   - `units`;
   - `metrics`;
   - `plotting`;
   - a model adapter: `paoflow_optics.OpticsModel` (absorption, refractive index, emissivity), opened with `.from_system(<system folder>)`, or a new one written from `templates/model_adapter_template.py`. Carrier transport, resistivity, superconductivity and ARPES have their own workflows (`agentic/transport/`, `agentic/superconductivity/`, `agentic/arpes/`) and adapters (`PAOFLOW.agentic.transport`, `.superconductivity`, `.arpes`).
   - The model is an electron-phonon **system folder** (`examples/elphon/<System>` layout: `system.py`, `optics/`, `cache/`). New runs are driver commands of that folder (`optics/absorption.py ...`, `optics/emissivity.py ...`, `optics/dielectric.py`), always with `--outdir` so earlier outputs are kept; `optics/emissivity.py` recomputes the emissivity for another thickness, angle, direct dielectric function or direct scissor in seconds, without a kernel run (see the pipeline table in `AGENT_WORKFLOW.md`).

   Every study has three scripts, started from `templates/`: `digitize_sources.py` (sources → `data/*.csv`), `analysis.py` (model at experimental conditions → `data/model_curves.csv`, `tables/`) and `plot_figures.py` (CSV only → `figures/`). Inputs are Python arguments; do not use config files. The worked example is the Si optics study in `AGENT_WORKFLOW.md` §Worked example.
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
   - quote every number with its file;
   - correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`. If they paste one, do what it asks and report what its **Check** line names.
