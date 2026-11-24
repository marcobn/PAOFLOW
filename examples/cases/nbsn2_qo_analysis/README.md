# NbSn2: quantum oscillations vs torque magnetometry

Orthorhombic NbSn2 (Fddd): the extremal orbits of spin–orbit and scalar-relativistic pyskeaf runs
are compared with a torque rotation from [001] to [110] with `PAOFLOW.agentic.qo_analysis`:
Fermi-surface inventory, digitised experimental FFT map, branch assignment, field directions,
spin–orbit vs scalar-relativistic, Fermi-level scan, breakdown gaps, weak peaks and a collaborator
package. The pyskeaf outputs come with the case (`calc/`, compressed), so no QE or PAOFLOW run is
needed. The measured peaks in `data/` are digitised from an image of the torque FFT map and the
Lifshitz–Kosevich values transcribed from slide text, as the `#` lines of each table record; they
are not raw data.

## Run it

- **PAOFLOW Studio**: *Examples…* → `nbsn2_qo_analysis`. Studio copies the case, lists its
  scripts and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example nbsn2_qo_analysis` copies the case into
  `./nbsn2_qo_analysis`; run the steps below in that folder.
- **Python**: `main.py` is the full spin–orbit analysis, `main_sr.py` the scalar-relativistic one
  and `qo_rotation.py` the comparison of the how-to; read them to change a call. They need
  `pip install "PAOFLOW[qo_analysis]"`.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Spin-orbit analysis against the torque rotation | `python main.py` | about 1.5 minutes on one core |
| Scalar-relativistic run and its convergence (81^3 vs 49^3) | `python main_sr.py` | about 30 s on one core |
| Rotation compared with the measured peaks (how-to) | `python qo_rotation.py > qo_rotation.log` | a few seconds |

Set `RUN_NETWORK = True` in `main.py` to add the magnetic-breakdown network, about 15 minutes on
24 cores.

## Outputs

- `qo_analysis_soc/<step>/`: tables, figures and a short report of each step of `main.py`, for
  example `sweep/fig_sweep_overlay.png` (calculated orbits over the torque peaks, with the branch
  labels of `data/branch_assignments_soc.csv`) and `sweep/summary.txt` (match rate and chance
  baselines).
- `qo_analysis_soc/collaborator_package/`: the data of the study, with the scripts that redraw its
  figures and a README.
- `qo_analysis_sr/compare_runs/`: the scalar-relativistic frequencies on the 81³ grid against the
  49³ grid (`fig_run_comparison.png`).
- `qo_rotation/`: the same steps for the how-to (`sweep/per_angle_listing.txt`: measured and
  calculated frequencies at each measured angle).
- `expected_results.json`: the reference numbers of `main.py`, checked by
  `tests/integration/agentic/qo_analysis`.

Explained in: [How to compare pyskeaf orbits with measured quantum-oscillation frequencies](https://paoflow.readthedocs.io/en/latest/learn/howtos/quantum-oscillations.html)
