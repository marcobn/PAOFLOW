# Pt slab: layer-resolved spin Hall conductivity

A four-layer Pt slab: the spin Hall conductivity of the whole slab and of its first layer (projection of the spin current on the atoms of the layer).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example pt_slab_layer_shc` copies the case into `./pt_slab_layer_shc` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Projections on the atomic wavefunctions: `projwfc.x -in proj.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Pt.rel-pz-n-rrkjus_psl.0.1.UPF`.
