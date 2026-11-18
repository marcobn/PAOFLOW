# Twisted bilayer graphene: moiré bands unfolded onto the graphene zone

The commensurate (n, m) = (3, 2) cell of twisted bilayer graphene: 76 carbon atoms, twist angle 13.17°,
moiré length 10.7 Å, interlayer distance 0.8 × 3.35 Å. A distance-dependent EDTB model of carbon
(`C_EDTB_DD_params.json`) gives the Hamiltonian. The bands of the untwisted (AA-stacked) supercell are
unfolded onto the zone of the (1, 0) bilayer cell; the twisted bands on the same path are coloured by
these spectral weights, next to the bands folded in the moiré zone. `TBG_sparse.py` repeats the
calculation with the sparse solver (shift-invert Lanczos on the bands nearest the Dirac point,
`PAOFLOW.spectrum.sparse_bands.SparseEDTB` and `unfold_bands_sparse`), which never stores a dense
H(k): the route for large moiré cells. No DFT run.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example tbg_unfold` copies the case into `./tbg_unfold` and prints
  the steps; run them in that folder.

## Steps

1. `python TBG.py 3 2`: prints the table of commensurate cells, writes the geometries of
   the (1, 0) and (3, 2) cells (`geom_tbg_*.json`, twisted and AA), draws the atoms of both cells,
   unfolds the AA bands (dense diagonalisation), computes the twisted bands (`Carbon_tbg_3_2/`) and
   writes the figures. The script runs under `mpirun` too (rank 0 writes the files), but at (3, 2)
   four ranks took as long as one.
2. `python TBG_sparse.py 3 2`: the same with the sparse solver, 120 bands nearest the Dirac point,
   k-points spread over joblib workers within the core budget (`LOKY_MAX_CPU_COUNT` caps them).
   Writes `Carbon_tbg/` (AA bands), `Carbon_tbg_3_2_sparse/` (twisted bands) and the `_sparse`
   figures. It runs on its own too (it writes the geometries it needs).

Other cells: pass another (n, m) with n > m ≥ 0, e.g. `4 3` (148 atoms, 9.43°) or `5 4` (244 atoms,
7.34°). The dense step grows with the cube of the number of orbitals; the sparse one is the way to the
small angles.

Runtime (Mac Studio, M2 Ultra): `TBG.py` about 3 minutes, `TBG_sparse.py` about 45 s with 4 workers.

## What to look at

- `tbg_3_2_folded_vs_unfolded.png`: left, the bands folded in the moiré zone; right, the twisted bands
  on the Γ-M-K-Γ path of the bilayer cell, coloured by spectral weight, over the bands of the (1, 0)
  cell (green). The states on the bilayer bands, the Dirac cone at K included, keep a weight near 1
  (amber); the folded replicas (dark blue) carry almost none.
- `tbg_3_2_unfolded.png`: the unfolded AA-supercell bands on the primitive bands.
- `tbg_3_2_atoms_twisted.png`, `tbg_3_2_atoms_untwisted.png`: the two layers and the moiré cell.
- The `_sparse` figures repeat the dense ones in the ±4 eV window around the Dirac point.

Former location: `examples/unfold_examples/TBG`.
