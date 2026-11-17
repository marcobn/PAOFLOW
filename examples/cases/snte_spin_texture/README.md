# SnTe monolayer: spin texture, Fermi surface and spin Hall conductivity

A 2D layer with spin-orbit coupling: Fermi surface and spin texture near the valence-band top, and the spin Hall conductivity in sheet units.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example snte_spin_texture` copies the case into `./snte_spin_texture` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. Projections on the atomic wavefunctions: `projwfc.x -in proj.in`
4. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Sn.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`, `Te.rel-pbe-dn-kjpaw_psl.1.0.0.UPF`.

## Notes from the original example (`examples/qe_examples/example08`)

2D SnTe with relativistic PBE pseudopotentials.

* Bands are calculated along a user definied path

* Fermisurface and Spin Texture are calculated through the entire BZ on a densly interpolated mesh (140x140x1).

* Spin Hall Conductivity is computed in 2 dimensional units (h_bar/e)/(Ohm*m)
