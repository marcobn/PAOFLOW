# Lessons learned (TaC, δ-TaN, HfC, Ta–Hf–C campaigns on Snellius and workstations, November 2026)

Read together with `AGENT_WORKFLOW.md`. Each entry: **symptom → cause → what to do**, with the instance and, where it
exists, the code of `ACTIONS` that detects it.

## Cost and planning

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | The campaign spent six times its plan | Per-cell times were extrapolated from smaller cells; three cells per configuration ran into the 36 h limit before anyone measured one | Run a pilot of the most expensive cell first and re-estimate from its time per iteration and iteration count (stage 3) | Ta₄HfC₅ special-displacement cells: planned 3.5–5 h each, ≈ 10 k SBU in total; actual > 36 h for three of five configurations, ≈ 57 k SBU before the fix |
| 2 | The same cells cost 1/5 after the fix | Inflated k-points, inflated FFT boxes and sloshing multiply (3 × 2–16 × 2) | Fix all three before resubmitting; measure again with a pilot | Reduced-basis rerun: 28 cells in 6.9 k SBU (refs 11 iterations, cells 14–17, ≈ 1 h each on 114–156 cores) |
| 3 | Billed more cores than requested | Shared nodes bill in steps (Snellius fat nodes: 24 cores) | Request multiples of the step; then use the cores you pay for (one 6-core pool per k-point) — `billing_rounding`, `pool_layout` | 60 cores billed as 72; 78 as 96 |
| 4 | A shared budget looked untouched after a day of jobs | The provider's account report updates late | Account from `sacct` (`Campaign.spent`), not from the budget report | `accinfo` unchanged for a day, then −51 k SBU at once |

## Numerical setup

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 5 | A displaced supercell without symmetry runs with 3× the k-points of its grid | pw.x reduces the grid with the *lattice* group and unfolds it with the *crystal* group; a grid not invariant under the lattice group gains off-grid points | `nosym = .true.` for cells without symmetry (and their undisplaced references, so that both sample the same k-points) — `kpoint_inflation` | 4×4×3 grid: 81 k-points instead of 26; 4×3×3: 57 instead of 19 |
| 6 | 80-atom cells with the FFT box of a 1000-atom cell | A supercell built as a multiple of a skewed primitive cell keeps the skew; pw.x sizes the box from the vector lengths | Rewrite the cell in a reduced basis of the same lattice (`reduce_cell_input`, same atoms) — `fft_inflation` | 288³ (rhombohedral cell at 13.9°) → 72×72×288; 192³ → 72×100×192; 5–16× fewer points |
| 7 | scf accuracy jumps from 25 to 9000 Ry, then crawls | Charge sloshing of large metallic cells with plain mixing at β = 0.7 | `mixing_mode = 'local-TF'`, β 0.3, `mixing_ndim = 12`; start displaced cells from the density of their undisplaced reference (`startingpot = 'file'`) — `charge_sloshing`, `plain_mixing_large_cell` | 0.01–1.2 Ry after 11–20 iterations before; 11 (reference) and 14–17 (displaced) iterations after |
| 8 | Flat, wrong optical spectra from one configuration | A k-grid with two points along an axis: PAOFLOW's Fourier interpolation splits the Nyquist term between +R and −R, so the odd part of the hoppings across the cell boundary is lost | At least three points per axis for every interpolated Hamiltonian — `two_point_axis` | 5×5×2 grid: ε(650 nm) 0.31–0.33 and flat; with 4×3×3: 0.42–0.50, like the other configurations |
| 9 | One configuration sampled much worse than the others | A rule "no worse than the original grid" kept a coarse original | A common floor of the shortest superlattice vector (48 bohr) for all runs — `coarse_kgrid` | 3×3×3 at 37 bohr cancelled after minutes and rerun with 4×4×3 (49 bohr) |

## Codes and environment

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 10 | A job hung until its time limit after an error | One MPI rank raised, the others waited in a collective | Run MPI drivers through `run_main` or `python -m mpi4py` (abort on any exception); watch for `output_stalled_after_error` | Four optics jobs, 31 h × 48 cores waiting |
| 11 | `MPI_ERR_ARG` in a gather of large arrays | 32-bit MPI counts (Open MPI 5.0 implements MPI 3.1, no large counts) | A PAOFLOW with chunked collectives, or a smaller interpolation grid — `mpi_count_overflow` | nk × 3 × n<sub>orb</sub>² = 2.4–4.7·10⁹ elements |
| 12 | Davidson stops at the first k-point ("problems computing cholesky") | Ill-conditioned subspace with many bands and random starting wavefunctions | `diagonalization = 'cg'` (QE 7.5 has no ppcg) — `davidson_cholesky` | 320-band nscf of 10-atom cells |
| 13 | A setup stage fails in seconds | The system description requires the q-grid to divide the coupling k-grid, even when no nscf runs | Make the coupling grid a multiple of the q-grid — `qgrid_does_not_divide_kgrid`, checked in preflight | 3×3×2 q against 8×8×8 k; fixed with 9×9×8 |
| 14 | A setup stage looks for a save and markers that do not exist | The stage was written for DFPT phonons; with external force constants the files are never produced | Link the scf save and create the markers in the stage script — `missing_file` | special-displacement setup with transferred force constants |
| 15 | Results of old and new code mixed | A remote clone was updated between campaigns | Record commits in `PLAN.md`; pull only between campaigns; after an update, recompute one finished case and compare | Dielectric weights fix: 2·10⁻⁴ change at 650 nm, so the old results stood |
| 16 | A regenerated setup kept the old setting | The campaign builder keeps existing files (designed to preserve uploaded data) | Move the generated file aside before regenerating; check the regenerated value | `system.py` kept the old coupling grid |
| 17 | Two productions discarded | The nscf of an HDF5 build of QE had not read the scf density | Check the nscf output for the density it read before using it | TaC electron–phonon production |

## Scheduler and monitoring

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 18 | Dependent jobs vanish or wait forever | When a parent fails or is cancelled, Slurm cancels or holds its dependents (`DependencyNeverSatisfied`) | Cancel orphans (act) and resubmit the chain from the failed stage after the fix — `dependency_never_satisfied` | post-processing jobs of cancelled cells |
| 19 | A job ran 36 h and left nothing to restart from | pw.x writes restart files only when it stops at `max_seconds` | Set `max_seconds` below the time limit for long runs; restart with `restart_mode = 'restart'` — `max_seconds` | timed-out cells restarted from scratch |
| 20 | A launcher never launched anything | Its parser split `jobs.tsv` on blanks while the date field holds one | Test every watcher or launcher on a finished run before relying on it; parse tab-separated files on tabs | 12 runs waited a day until a manual launch |
| 21 | An edited watcher kept its old behaviour | A running shell keeps the code it has read | Stop and restart a watcher after editing it | two launchers ran side by side |
| 22 | Watchers lost at a restart | They lived in `/tmp` | Keep watcher scripts in the study folder or use the library watcher; restart them when resuming | campaign watcher after a machine restart |
| 23 | A count of queued jobs was 0 while 42 were queued | `squeue` truncates job names to 8 characters by default | Always ask for full names (`-o '%j'` or the formats of `scheduler.py`) | grep for the campaign name found nothing |
| 24 | A monitor reported a stopped run that was fine | Log greps match unrelated text | Read states through the parsers (`Campaign.status`), not greps | SSCHA monitors (see `agentic/sscha`) |
| 25 | A failed ssh read as "no jobs" | The monitor treated the error as an empty answer | An unreachable machine is unknown (`MachineUnreachable`, `machine_unreachable`) | Lonestar6 control socket lost |

## Storage and reports

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 26 | Results at risk on a cluster | Scratch purge after 14 days | Copy results to permanent storage after every stage; watch the age — `scratch_purge_risk` | Snellius scratch-shared |
| 27 | A two-way sync stopped with conflicts | Both sides changed the same file | Never resolve conflicts silently; ask which side wins | Unison between workstations |
| 28 | Copies on a mirror never shrink | One-way mirrors run without `--delete` by design | Moved or removed folders stay on the mirror; say so when reporting a move | report mirror |
