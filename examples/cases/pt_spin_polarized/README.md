# Pt (ultrasoft, nspin = 2): bands, DOS and transport

Spin-polarised platinum with an ultrasoft pseudopotential and Methfessel-Paxton smearing: one set of bands, DOS and tensors per spin channel.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example pt_spin_polarized` copies the case into `./pt_spin_polarized` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. Projections on the atomic wavefunctions: `projwfc.x -in proj.in`
4. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Pt.pz-n-rrkjus_psl.0.1.UPF`.
