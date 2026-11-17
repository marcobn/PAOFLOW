# Sparse backend on a doubled cell

`main_sparse.py` runs `PAOFLOW.SparsePAOFLOW` (the bond-list Hamiltonian of `src/PAOFLOW/sparse/`) on bulk
Si: the PAO Hamiltonian, a doubled cell (`doubling_Hamiltonian`), an energy window, then bands, DOS and
transport without dense `(nk, nawf, nawf)` arrays. The generator does not write it: the sparse backend
is meant for large cells, where the dense drivers of `paoflow-gen` run out of memory.

1. Run the case `examples/cases/si_bands_transport` up to projwfc.x (or copy its `silicon.save`).
2. Copy `main_sparse.py` next to `silicon.save` and run `python main_sparse.py`.

See `.agents/skills/sparse-backend/SKILLS.md` for the memory contract of the backend.
