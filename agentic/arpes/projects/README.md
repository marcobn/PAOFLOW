# Studies

One folder per study, e.g. `projects/Pb_kink/`. Each holds:
- `sources/`, copies or renders of the experimental files (papers, band-map images);
- `data/`, the canonical datasets (`*.csv` with provenance headers) and `model_curves.csv`;
- `checks/`, overlay images that validate every digitisation and every map calibration;
- `tables/`, `figures/`;
- the scripts `digitize_sources.py`, `analysis.py`, `plot_figures.py`, and `REPORT.md`.

Calculation folders (the electron-phonon system folders) stay where they are and are referenced by path, read-only.

Everything here except this README is ignored by git, so your studies never end up in a PAOFLOW commit. A study may also live next to its calculation, outside this folder.
