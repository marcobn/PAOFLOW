# PAOFLOW agentic workflows

Each folder here is a self-contained workflow that an AI coding agent (Claude Code, Codex, or similar) runs on top of PAOFLOW's library code. The library code lives in `src/PAOFLOW/`. The workflows only import it, and their files are never installed with the package. The transport, superconductivity, ARPES and non-linear optics workflows share the data toolkit of `thermo_optics` (`datasets`, `units`, `ingest`, `digitize`, `metrics`, `plotting`) and read the same electron-phonon *system folders* (`examples/elphon/<System>` layout) through their own adapters.

| Folder | Workflow | Library code |
|---|---|---|
| `skeaf/` | Interpreting pyskeaf / SKEAF extremal-orbit calculations against experimental quantum-oscillation data (dHvA, SdH, torque) given as tables, text or images | `PAOFLOW.agentic.qo_analysis` |
| `thermo_optics/` | Comparing PAOFLOW's temperature-dependent optical response (absorption, free carriers, refractive index, emissivity) with experimental data given as tables, text, vector or scanned PDF figures, slide decks or images: canonical datasets with provenance, model at the measured conditions, metrics with baselines, diagnosis, publication figures | `PAOFLOW.agentic.thermo_optics` |
| `transport/` | Comparing PAOFLOW's electron-phonon transport with experiment: phonon-limited mobilities, carrier densities, conductivity and Seebeck of semiconductors; resistivity, infrared optics and thermal emissivity of metals from α²F_tr; the Boltzmann bridge with impurity and residual channels | `PAOFLOW.agentic.transport` (+ the thermo-optics toolkit) |
| `superconductivity/` | Comparing PAOFLOW's electron-phonon superconductivity with experiment: α²F and its moments, McMillan / Allen–Dynes / Migdal–Eliashberg Tc with the μ* sensitivity, Δ(T), gap ratios, the gap and λ_nk distributions of multi-gap metals, the quasiparticle DOS | `PAOFLOW.agentic.superconductivity` (+ the thermo-optics toolkit) |
| `arpes/` | Comparing PAOFLOW's simulated ARPES with measured spectra: dispersions and kinks (λ by a stated bare-band protocol), self-energies, MDC widths, phonon linewidths versus temperature, band-map images through an axis calibration | `PAOFLOW.agentic.arpes` (+ the thermo-optics toolkit) |
| `nlo/` | Comparing PAOFLOW's second-order optical responses (SHG susceptibility and d coefficients, shift and injection currents, 3D or 2D sheet values) with experiments and published calculations: conventions as data, the symmetry audit of the run as the error bar | `PAOFLOW.agentic.nlo` (+ the thermo-optics toolkit) |

How to start any workflow:
1. Go to its folder.
2. Read its `README.md`.
3. Open the agent there.
4. Paste the kickoff prompt.

Each workflow folder contains:
- `AGENTS.md` / `CLAUDE.md`: agent instructions, loaded automatically when the agent is opened in the folder;
- the playbook and lessons learned;
- a `prompts/` library;
- `templates/`;
- a `projects/` area, ignored by git, for your own studies.
