# Large MPI messages and failing ranks: plan and implementation (November 2026)

PAOFLOW's collective helpers pass element counts and offsets to MPI as C `int`s. A single scatter or gather of more
than 2³¹ − 1 elements fails, and when one rank fails in a collective the others wait for it until the wall-time
limit. This note records the case that exposed both problems, what the fix involves, and the order of the work.

## What happened

In the Ta₄HfC₅ campaign (10-atom cells, 240 orbitals in the extended basis), the room-temperature dielectric function
failed in `PAOFLOW.position_dipole_correction`:

- every rank computes the Löwdin position dipoles on a slice of the interpolated k-grid; the slices are gathered on
  rank 0 with `gather_array` and scattered again with `scatter_full` into the standard k distribution;
- the gathered array has nk × 3 × n<sub>orb</sub>² elements, 24³–30³ × 3 × 240² = 2.4–4.7 × 10⁹, above the int limit;
- Open MPI 5.0.7 on Snellius implements MPI 3.1, without the MPI-4 large-count calls, so mpi4py 4.1 passes the
  counts as `int` and MPI returns `MPI_ERR_ARG` (with MPICH 5.0, as on the Mac Studio, mpi4py uses the large-count
  calls and the gather works);
- the failing rank stopped, the others waited in the collective until the 8 h limit: four jobs, about 31 h × 48 cores
  (≈ 1500 core hours) spent waiting.

The workaround was a smaller interpolation grid (20³; the 80-atom cell grid 6³) and an automatic cap in the Snellius
workflow. With TaC-sized systems (48 orbitals) the limit is far away, which is why it had not shown up before.

## Where the limit applies

- `utils/communication.py`: `scatter_array` and `gather_array` use `Scatterv`/`Gatherv` with counts and
  displacements in elements (`load_sizes`). Everything built on them inherits the limit: `scatter_full`,
  `gather_full` (their `npool` chunks were meant to keep messages small, but only along the first axis and only when
  the caller sets `npool`), and `gather_scatter`. Calls: 11 `gather_array`, 6 `scatter_array`, 40 `gather_full`,
  33 `scatter_full`, 6 `gather_scatter`, 12 `reduce_full`.
- Direct buffer collectives (`Bcast`, `Reduce`, `Allreduce`, ...) in 15 modules (82 calls; most in `pao_sym`,
  `conductor_energy_loop`, `do_Boltz_tensors`, `do_epsilon`, `do_pao_eph_dense_q`). Most move small tensors; the ones
  that scale with nk × n<sub>orb</sub>² need checking.
- Memory is a separate limit: gathering a full nk × 3 × n<sub>orb</sub>² array on rank 0 needs 22–75 GB in the case
  above, even where the counts fit.

## Plan

### 1. A failing rank ends the job (small, first)

`utils/mpi_guard.run_main` (since a7457270) prints the traceback and calls `MPI_Abort` when a driver raises on any
rank. Most `paoflow-gen` drivers use it; the elphon templates that do not (`optics/dielectric.py` and the others
without the guard) should, and so should scripts written by workflows (the Snellius optics and reference-cell
scripts). For scripts outside PAOFLOW, `python -m mpi4py script.py` gives the same behaviour. Effect: a failure costs
seconds instead of the remaining wall time, and job monitors see it at once.

### 2. Collective helpers that do not overflow

- `scatter_array` / `gather_array`: counts and displacements in rows, with a contiguous MPI datatype for one row
  (the product of the trailing dimensions). Only the number of rows has to fit in an `int`. The data layout is the
  same, so results are bitwise identical.
- A per-call size cap (for example 1 GiB): the transfer runs in rounds in which every rank sends a slice of its block
  (counts may be zero). This guards against MPI libraries that mishandle single messages over 2 GiB.
- `reduce_full` and large `Bcast`/`Reduce`/`Allreduce` calls: a helper that loops over slices of the flat buffer.
- The public signatures stay as they are; `npool` keeps working and is no longer needed for message size.

### 3. No gather in `position_dipole_correction`

The gather and scatter only reorder the k-points. A helper `scatter_full_indices(n, npool)` that returns the global
indices `scatter_full` assigns to a rank lets every rank compute its own k-points directly: no communication, no
array on rank 0, the same numbers. The same pattern applies wherever a gather is immediately followed by
`scatter_full`.

### 4. Audit of the direct collectives

Classify the 82 direct calls by the size of their buffers; route those that scale with nk × n<sub>orb</sub>² (or
n<sub>R</sub> × n<sub>orb</sub>²) through the helpers of step 2.

### 5. Tests

- Unit: the row layout, the rounds of step 2, and `scatter_full_indices` against `scatter_full`.
- MPI (as `tests/unit/utils/test_mpi_guard.py`, `mpirun -np 2` and `4`): the helpers with the int limit lowered to a
  few thousand elements by monkeypatching, so the rounds and row types are exercised with small arrays, compared
  bitwise with the current results.
- Regression: an example with `dielectric_tensor` and position dipoles under MPI, identical output.
- Manual, on Snellius (Open MPI 5.0): a gather of 2.2 × 10⁹ `int8` elements (2.2 GB), which fails today.

### 6. Deployment

Commit (by the user), then `git pull` on Snellius only after the running campaign has finished: the Snellius clone
(3940bf5c) is many commits behind, and pulling while jobs are queued would change the code under them. After the
update, the nfft cap of `snellius_workflow/make_campaign.py` can become a memory cap.

## Effort

Steps 1 and 3 about 15–20 minutes each, step 2 about 40 minutes, step 4 about 30 minutes plus fixes, step 5 about
30 minutes, documentation 10 minutes: roughly 2–2.5 hours of agent time, then review and commit.

## Status (implemented 11 November 2026, not yet committed)

- **Step 1**: the twelve elphon templates without the guard end with `run_main(main)` (all but the serial
  `plot_me.py`). The Snellius stage scripts run their MPI drivers as `srun python -m mpi4py <script>`, which aborts
  every rank on an unhandled exception with any PAOFLOW version.
- **Step 2** (`utils/communication.py`): `scatter_array` and `gather_array` count in rows (a contiguous datatype of
  the trailing dimensions) and move at most `MAX_MESSAGE_BYTES` (1 GiB) per call; with more rows than
  `MAX_MPI_COUNT`, every round is staged in a buffer on the root, so even a 1-D array of more than 2³¹ elements goes
  through. `reduce_full`, and the new `bcast_array` and `allreduce_inplace` (both with an optional communicator,
  C- or Fortran-contiguous arrays), work in slices of `MAX_MPI_COUNT` elements. Reductions of arrays below the limit
  make a single call as before, so their summation order, hence their bits, do not change.
- **Step 3**: `scatter_full_indices(n, npool)` gives the indices `scatter_full` assigns to a rank;
  `position_dipole_correction` builds the dipoles of those k-points directly, with no gather on rank 0. The results
  agree with the former route to round-off (≈ 10⁻¹⁴): the vectorised dipole arithmetic depends at that level on how
  many k-points are computed together, so the former route already varied by as much with the number of ranks.
- **Bug found on the way**: `nonlocal_velocity._kgrid_to_cart` read a `(3, 3)` k-grid as `(nk, 3)`, so a batch of
  exactly three k-points in PAOFLOW's `(3, nk)` layout was transposed (for instance an MPI rank holding three
  k-points in `position_dipole_correction`, wrong at order one). A first axis of length 3 is now always `(3, nk)`.
  Large production grids (hundreds of k-points per rank) were not affected; small test runs could be.
- **Step 4** (audit): routed through the helpers are the broadcasts of the full k-grid Hamiltonian in `pao_sym`
  (`Hksp`), of `dHRs`/`d2HRs` in `do_topology` and `do_berry_curvature`, and the in-place reductions of the transport
  Green's functions and lead self-energies, the Fermi-surface coupling, the electron self-energy, and the array
  reductions of `coupled_crystal`, `nonlinear_optics`, `coupled_response` and `phonon_assisted_absorption`. The
  other direct collectives move small tensors (DOS, PDOS, Boltzmann and Hall tensors, Fermi-level sums, λ and ω per
  mode); `DataController.py` already broadcasts in slices.
- **Step 5**: `tests/unit/utils/test_communication.py` (layout and slice planners, the `scatter_full` partition,
  single-rank helpers with tiny limits, and an MPI test on 2 and 3 ranks with 64-byte rounds, staged rounds and
  7-element slices, bit for bit, plus the dipole route comparison), a regression test for the three-k-point
  transposition in `tests/unit/hamiltonian/test_position_dipoles.py`, and an opt-in test of 2.2 × 10⁹ `int8`
  elements (`PAOFLOW_LARGE_MPI_TEST=1`, about 5 GB; passes on the Mac Studio, to be run on Snellius after the update).
