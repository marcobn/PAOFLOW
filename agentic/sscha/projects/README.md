# Studies

One folder per study, e.g. `projects/TaN_stability/`. Each holds `campaign.py` (the drivers of the scan),
`analysis.py`, `plot_figures.py`, `data/` (experimental datasets with provenance), `tables/`, `figures/`, `PLAN.md`
and `REPORT.md`. The run directories (`output/sscha_T<T>/` next to each `main.sscha.py`) live where the drivers
run (scratch on a cluster) and are referenced by path; their `*.json` and `dyn*` files are copied to permanent
storage after each temperature. Everything here except this README is ignored by git.
