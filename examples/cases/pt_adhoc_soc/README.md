# Pt: spin Hall conductivity with ad hoc spin-orbit coupling

A scalar-relativistic Pt calculation; the spin-orbit coupling of the d shell (lambda_d = 0.553 eV) is added to the PAO Hamiltonian, then the spin Hall conductivity is computed as with a fully relativistic run.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example pt_adhoc_soc` copies the case into `./pt_adhoc_soc` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `standard` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Pt.pz-n-rrkjus_psl.0.1_ld1-6.3.UPF`.
