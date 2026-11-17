# bcc Fe: anomalous Hall conductivity and Berry curvature

Ferromagnetic iron with spin-orbit coupling (noncollinear): Berry curvature along the band path and the anomalous Hall conductivity sigma_xy(E), with its AC part.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **run** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example fe_anomalous_hall` copies the case into `./fe_anomalous_hall` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. Projections on the atomic wavefunctions: `projwfc.x -in proj.in`
4. PAOFLOW: `mpirun -np 4 python main.py`, then `python plot.py` (or the Plots tab)

Pseudopotentials (from `PSEUDOS/library`): `Fe.rel-pz-spn-rrkjus_psl.0.2.1.UPF`.
