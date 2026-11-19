# AlP: Hubbard-corrected bands, transport and dielectric function

A fixed on-site correction (-U/2 on the Al and P p orbitals, the non-self-consistent ACBN0 values) opens the gap; then DOS, transport and the optical dielectric tensor.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example alp_hubbard_optics` copies the case into `./alp_hubbard_optics` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `standard` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Al.pbe-mt_fhi.UPF`, `P.pbe-mt_fhi.UPF`.
