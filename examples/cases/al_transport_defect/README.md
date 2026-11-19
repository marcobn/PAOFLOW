# Al with a defect: the low-level Landauer calls

`main.py` builds the Landauer blocks of an Al supercell with a defect in its middle: the first and last
five atomic layers along z are the leads (`left_lead_layers=5`, `right_lead_layers=5`), the rest is the
conductor. It shows the low-level `PAOFLOW.Transport` calls one by one: partition, Hamiltonian blocks,
the optional tuning (on-site shifts, lead convergence, eigenchannels; commented out with their
defaults), energy grid (-7 to 2 eV, 6001 points), lead self-energies, Green functions, transmission and
DOS, with the MPI communicator passed explicitly. For a standard junction the `landauer` workflow of
`paoflow-gen` and Studio writes the same calls (see the case `alh_junction_landauer`).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example al_transport_defect` copies the case into `./al_transport_defect` and prints
  the steps; run them in that folder.

## Steps

1. Your QE runs, not part of the case: pw.x (scf and nscf) for an Al supercell with the defect, long
   along z with at least five clean layers at each end, then projwfc.x with `lwrite_overlaps = .true.`,
   with `prefix = 'al'` and `outdir = 'output/qe'` (the driver reads `output/qe/al.save`).
2. `mpirun -np 4 python main.py`: writes the self-energies, Green functions, transmission and DOS
   to `output/paoflow/` (file names with the postfix `_defect`).

Runtime: set by your supercell and the 6001 energies.

## What to look at

The transmission against energy in `output/paoflow/` (postfix `_defect`). Compare it with the
transmission of the clean supercell, the number of open channels of the leads: the defect scatters
part of it.

Former location: `examples/recipes/transport_defect`.
