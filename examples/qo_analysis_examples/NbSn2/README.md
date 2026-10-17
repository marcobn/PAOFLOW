# NbSn2: torque rotation [001] → [110] vs pyskeaf

The system on which `PAOFLOW.agentic.qo_analysis` was developed: NbSn2, Fddd, F-centred orthorhombic (b/a = 1.7356, c/a = 3.3813).

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

Run with `python main.py` (about 2 minutes without the network). Results go to `qo_analysis_soc/`.

Main conclusions of the original study:
- μ, σ2, γ, δ and β1/β2 near [001], β1 near [110], and the 135 T pockets near [110] are explained, with consistent mass enhancement 1 + λ ≈ 2–3.
- The heavy band-7 orbits beyond ~55° (δ, and κ/δ1/δ2 at [110]) are not explained by convergence, small Fermi-level shifts or SOC.
- Breakdown is expected from the small gaps but not demonstrated.
- Many weak peaks are harmonics or combinations of the main branches (mixing, not breakdown).
