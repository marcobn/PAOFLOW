# Al-H-Al junction: Landauer transmission and eigenchannels

A hydrogen layer between two Al leads: conductor transmission, the four main eigenchannels and the zero-bias current.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials, fills the **landauer** form and lists the runs below in the
  Runs tab.
- **Command line**: `paoflow-gen --example alh_junction_landauer` copies the case into `./alh_junction_landauer` and writes the
  drivers; then run the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in`
3. PAOFLOW: `mpirun -np 4 python main.landauer.py`, then `python plot.landauer.py` (or the Plots tab)

Projections: PAOFLOW's internal projections on the `standard` basis of the pseudopotentials (generated at run time in `BASIS_PS`); pick *projwfc* in the form for QE's projwfc.x, and Studio adds `proj.in` and the projwfc.x run.

Pseudopotentials (from `PSEUDOS/library`): `Al.pbe-rrkj.UPF`, `H.pbe-van_bm.UPF`.

## Notes from the original example (`examples/transport_examples/example02`)

Bulk transport calculation example.

This example demonstrates a bulk transport calculation with Al system.
