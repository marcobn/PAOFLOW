# Tellurium: Rashba-Edelstein response

Left-handed trigonal Te: the spin density induced by a charge current (current-induced spin polarization) near the valence-band top.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example te_rashba_edelstein` copies the case into `./te_rashba_edelstein` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `minimal` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Te.rel-pbe-dn-kjpaw_psl.0.2.2.UPF`.

## Notes from the original example (`examples/qe_examples/example11`)

Calculating Rashba-Edelstein tensor elements for left handed tellurium.
The interpolated k-mesh should be made finer to achieve convergence.
To facilitate memory capacity of typical machines and ensure a reasonable runtime, in this example, the mesh is dramatically reduced.
For accurate results, choose k-meshes with the following specifications.
scf and nscf: 22 x 22 x 16
PAOFLOW interpolation: 200 x 200 x 180

main.py:
* Build the PAO Hamiltonian.
* Interpolate to a finer k-mesh.
* Compute the eigenvalues, gradient, and momenta.
* Compute the spin operator and spin texture in the user-defined energy window.
* Compute the adaptive smearing.
* Calculate the Rashba-Edelstein tensor, considering user-defined parameters like energy window, regularization, and two dimentional parameters.
