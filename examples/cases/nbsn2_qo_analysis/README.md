# NbSn2: torque rotation [001] → [110] vs pyskeaf

The system on which `PAOFLOW.agentic.qo_analysis` was developed: NbSn2, Fddd, F-centred
orthorhombic (b/a = 1.7356, c/a = 3.3813). The extremal orbits of spin-orbit and
scalar-relativistic pyskeaf runs (bundled, compressed) are compared with a torque-magnetometry
rotation from [001] to [110]: Fermi-surface inventory, field geometry, digitisation of the
experimental FFT map, sweep comparison and branch assignment, field directions and masses, SOC vs
scalar-relativistic, Fermi-level scan, breakdown gaps, weak peaks and a collaborator package.
The pyskeaf outputs come with the case: no QE or PAOFLOW run is needed.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies the
  scripts and the data and lists the runs below in the Runs tab.
- **Command line**: `paoflow-gen --example nbsn2_qo_analysis` copies the case into
  `./nbsn2_qo_analysis`; then run the steps below in that folder.

## Steps

1. Spin-orbit analysis: `python main.py` (about 1.5 minutes on one core). Results go to
   `qo_analysis_soc/` (about 20 MB). Set `RUN_NETWORK = True` in `main.py` for the breakdown
   network (about 15 minutes on 24 cores).
2. Scalar-relativistic run: `python main_sr.py` (a few seconds), with the coarse run as
   convergence reference. Results go to `qo_analysis_sr/`.

## Files

| Path | Content |
|---|---|
| `main.py` | full SOC analysis (inventory, geometry, digitisation, sweep, directions, SOC vs SR, E_F scan, breakdown gaps, weak peaks, collaborator package; set `RUN_NETWORK = True` for the breakdown network) |
| `main_sr.py` | scalar-relativistic run with the coarse run as convergence reference |
| `calc/soc/` | SOC run: `qo_EF_0_freqvsangle_11..18.out.gz` and `Fermi_surf_band_11..18.bxsf.gz` (49³ grid, num_interpolation 320, azimuthal 29.95°, polar 0 → 90° in 90 steps) |
| `calc/sr_81_320/`, `calc/sr_49_200/` | scalar-relativistic frequency tables (81³/320 and 49³/200) |
| `calc/sr_band7_fermi_scan/` | band 7 at E_F = −5, 0, +5 meV (polar 45 → 90°) |
| `data/exp_torque_FFT_vs_angle_slide15.png` | torque FFT-vs-angle map (MagLab 2026, from a slide deck) |
| `data/exp_sweep_peaks.csv` | peaks **digitised** from that image (not raw data) |
| `data/exp_LK_table.csv` | LK values **transcribed** from slide text, plus a VSM [110] set |
| `data/branch_assignments_soc.csv` | final branch labels |
| `data/NbSn2.cell.in` | cell (QE ibrav 10 primitive vectors in units of a) |
| `expected_results.json` | reference numbers, checked by `tests/integration/agentic/qo_analysis` |

## What to look at

- The printed summary of `main.py`: degenerate groups and Fermi-surface pieces, the field
  directions in both conventions ([110] is azimuthal 29.95° in the reciprocal one), 352 digitised
  peaks, the match rate of the sweep against chance, the SOC − SR frequency shifts per band, the
  smallest k-space gaps between sheets and the weak-peak combinations; they match
  `expected_results.json`.
- `qo_analysis_soc/sweep/`: calculated extremal orbits over the torque FFT peaks, with the branch
  labels (`fig_sweep_overlay.png`, `fig_on_exp_map.png`, `summary.txt`);
  `qo_analysis_soc/collaborator_package/`: the data of the study with its README and the
  scripts that redraw the figures.
- `qo_analysis_sr/compare_runs/`: convergence of the scalar-relativistic frequencies (81³/320
  against 49³/200, `fig_run_comparison.png`).

Main conclusions of the original study:
- μ, σ2, γ, δ and β1/β2 near [001], β1 near [110], and the 135 T pockets near [110] are explained, with consistent mass enhancement 1 + λ ≈ 2–3.
- The heavy band-7 orbits beyond ~55° (δ, and κ/δ1/δ2 at [110]) are not explained by convergence, small Fermi-level shifts or SOC.
- Breakdown is expected from the small gaps but not demonstrated.
- Many weak peaks are harmonics or combinations of the main branches (mixing, not breakdown).

The agentic playbook built on this analysis is in `agentic/skeaf/` of the repository.
