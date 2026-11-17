# Bi bilayer nanoribbon: bands projected on the edges

A zigzag Bi(111) bilayer ribbon with spin-orbit coupling: the weight of the six outermost atoms on each side picks out the helical edge states.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example bi_ribbon_edge_states` copies the case into `./bi_ribbon_edge_states` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `minimal` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Bi.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`.
