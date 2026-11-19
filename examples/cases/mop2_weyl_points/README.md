# MoP2: Weyl points

Search of the Brillouin zone for touchings of the highest valence and lowest conduction bands; with Z2Pack the chirality of each point.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example mop2_weyl_points` copies the case into `./mop2_weyl_points` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `minimal` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Mo.rel-pbe-n-nc.UPF`, `P.rel-pbe-n-nc.UPF`.
