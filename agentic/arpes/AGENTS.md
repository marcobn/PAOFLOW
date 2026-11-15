# Agent instructions: ARPES workflow

You are running the ARPES workflow: comparing PAOFLOW's simulated photoemission (band maps with PAO matrix elements, the electron-phonon kink, self-energies, momentum widths, phonon linewidths versus temperature) with measured ARPES data in any form (dispersion tables, extracted self-energies and λ values, linewidths versus temperature, band-map images, PDF figures, decks).

1. **Read before acting:**
   - `AGENT_WORKFLOW.md` (the model pipeline, operating rules, Stages 0–8, decision tree);
   - `LESSONS_LEARNED.md` (pitfalls).
   Follow them; do not re-derive the method.
2. **Use the library:**
   - `PAOFLOW.agentic.arpes`: the adapter `ArpesModel` (`.from_system(<system folder>, output=...)`: bare and dressed maps, MDC dispersions, k_F, bare and dressed v_F and λ, `sigma_<kind>_<T>K.dat` self-energies with `mass_enhancement` and `kink_energy`, the Fan-Migdal linewidths of every state versus temperature); `curves` (MDC/EDC peaks and widths, Lorentzian fits, `self_energy_from_dispersion` against a bare band, `linear_band`, `kink_energy`), which work on model and measured maps alike; `images.band_map_from_image` (a calibrated band-map image as an intensity map).
   - `PAOFLOW.elphon.self_energy.eliashberg_self_energy(omega, a2F, w, T)` for a baseline Σ from a published (tunnelling) α²F.
   - The shared toolkit `PAOFLOW.agentic.thermo_optics`: `ingest`, `digitize` (axis calibration, markers, tracers), `datasets` (canonical CSV with provenance), `units`, `metrics`, `plotting`.
   - The model is an electron-phonon **system folder** (`examples/elphon/<System>` layout: `system.py` with the `ARPES` defaults, `arpes/`, `superconductivity/`, `cache/`). New runs are driver commands of that folder (`arpes/arpes.py --self-energy ... --temps ... --direction ... --segment-ev|--full-path --npts --nq --eta-ev --adaptive-smearing`); `arpes.py` writes `arpes/output/`, so rename the previous `output/` to `output_<label>/` before a rerun and open it with `ArpesModel.from_system(folder, output='output_<label>')`. The isotropic self-energy needs `superconductivity/output/eliashberg.npz`, the anisotropic one `fs_coupling.npz`; the Fan-Migdal form needs a dense-q run (minutes to an hour).

   Every study has three scripts, started from `templates/` (and from `agentic/thermo_optics/templates/digitize_template.py` for figures): `digitize_sources.py` (sources → `data/*.csv`), `analysis.py` (model at experimental conditions → `data/model_curves.csv`, `tables/`) and `plot_figures.py` (CSV only → `figures/`). Inputs are Python arguments; do not use config files.
3. **Where to write:**
   - Write only inside the study folder (`projects/<study>/`, or the folder the user names).
   - Never write into calculation folders or into the source folders (papers, decks).
   - Do not modify `src/PAOFLOW/` or the files in this folder unless the user asks.
   - If the library needs a fix or a new tool, propose it with a test.
4. **Non-negotiable:**
   - never overwrite calculation output;
   - ask before new calculations or anything above ~10 min of CPU (a Fan-Migdal run is one);
   - every experimental number carries its provenance (`measured`, `transcribed`, `vector-extracted`, `digitised`) and an uncertainty;
   - every digitisation has an overlay check you have looked at;
   - every metric is quoted with a baseline;
   - dispersions are compared relative to k_F, with the bare band defined the same way on both sides, at the same resolution; an E_F offset is a stated diagnostic;
   - λ is quoted with the method that produced it (slope of Re Σ at E_F, ratio of velocities over a stated window, linewidth slope versus T);
   - quote every number with its file;
   - correct earlier statements explicitly.
5. **Prompts.** The user steers you with `prompts/NN_*.md`. If they paste one, do what it asks and report what its **Check** line names.
