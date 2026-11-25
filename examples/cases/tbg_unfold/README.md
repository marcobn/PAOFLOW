# Twisted bilayer graphene: moiré bands unfolded onto the graphene zone

Twisted bilayer graphene in its commensurate cells, with a distance-dependent environment-dependent
tight-binding (EDTB) model of carbon (`C_EDTB_DD_params.json`; the how-to scripts hold their own copy).
On the (3, 2) cell (76 atoms, 13.17°), PAOFLOW computes the moiré bands folded in the supercell zone
and unfolds the bands of the AA-stacked supercell onto the zone of the bilayer cell, first with a dense
diagonalisation, then with the sparse shift-invert Lanczos solver meant for large moiré cells. No DFT run.

## Run it

- **PAOFLOW Studio**: *Examples…* → `tbg_unfold`. Studio copies the case, lists its scripts and runs
  the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example tbg_unfold` copies the case into `./tbg_unfold`; run the
  steps below in that folder.
- **Python**: each step is a plain PAOFLOW script; read it to change a call. Start the sparse scripts
  with `OMP_NUM_THREADS=1`, because their workers each run their own linear algebra.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Moiré cell, bands and unfolding (dense solver) | `python TBG.py 3 2` | about 3 min (Mac Studio M2 Ultra) |
| The same with the sparse Lanczos solver | `python TBG_sparse.py 3 2` | about 45 s, 4 workers (Mac Studio M2 Ultra) |

`TBG.py` and `TBG_sparse.py` take another cell as (n, m) with n > m ≥ 0, for example `4 3` (148 atoms,
9.43°). The data of the case are the outputs of the script of the twisted-bilayer how-to, which runs
for 6 minutes on 4 cores:
`paoflow-gen --example tbg_unfold --with-data` (in Studio, *Use the precomputed data of the case*)
extracts them from `tbg_unfold-v1.tar.gz` of the release
[case-data](https://github.com/marcobn/PAOFLOW/releases/tag/case-data).

## Outputs

- `geom_tbg_*.json`: the twisted and AA geometries of the (1, 0) and (3, 2) cells.
- `tbg_3_2_folded_vs_unfolded.png`: left, the twisted bands folded in the moiré zone; right, the
  twisted bands on the Γ-M-K-Γ path of the bilayer cell, coloured by the spectral weights of the AA
  supercell, over the bands of the (1, 0) cell.
- `tbg_3_2_unfolded.png`, `tbg_3_2_atoms_twisted.png`, `tbg_3_2_atoms_untwisted.png`: the unfolded AA
  bands, and the atoms of the two (3, 2) cells.
- `Carbon_tbg_3_2/`, `Carbon_tbg/`, `Carbon_tbg_3_2_sparse/` and the `_sparse` figures: the same from
  the sparse solver, in the ±4 eV window around the Dirac point.

Explained in: [How to unfold the bands of large supercells with the sparse solver](https://paoflow.readthedocs.io/en/latest/learn/howtos/unfold-large-supercells.html),
[How to compute the band structure of twisted bilayer graphene](https://paoflow.readthedocs.io/en/latest/learn/howtos/twisted-bilayer.html)
