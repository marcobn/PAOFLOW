# Si: dielectric function and joint density of states

Norm-conserving Si with the internal projections: the dielectric function with the complete velocity (position dipoles) and the joint density of states; eps.in runs the epsilon.x of QE for comparison.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example si_optics` copies the case into `./si_optics` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Si_ONCV_PBE_sr.UPF`.
