# Landauer transmission of a conductor with a defect

`main.py` builds the Landauer blocks of an Al supercell with a defect in its middle: the first and last
five atomic layers are the leads (`left_lead_layers=5`, `right_lead_layers=5`), the rest is the
conductor. It shows the low-level `PAOFLOW.Transport` calls one by one (partition, blocks, optional
tuning, energy grid, self-energies, Green functions, transmission, DOS), with the MPI communicator
passed explicitly.

The QE inputs of the supercell are not part of the recipe: run pw.x, then projwfc.x with
`lwrite_overlaps=.true.`, for your own supercell into `output/qe/al.save`. For a standard junction the
`landauer` workflow of `paoflow-gen` / Studio writes the same calls (see the case
`examples/cases/alh_junction_landauer`).
