# Al: Landauer transmission of the perfect crystal

The number of conducting channels of bulk aluminium along z, T(E), and the DOS of the principal layer.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **landauer** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example al_bulk_landauer` copies the case into `./al_bulk_landauer` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. Projections on the atomic wavefunctions: `projwfc.x -in proj.in`
4. PAOFLOW: `mpirun -np 4 python main.landauer.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Al.pbe-rrkj.UPF`.
