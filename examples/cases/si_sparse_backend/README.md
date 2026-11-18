# Si: the sparse backend on a doubled cell

`main_sparse.py` runs `PAOFLOW.SparsePAOFLOW`, the bond-list Hamiltonian of `src/PAOFLOW/sparse/`, on
bulk Si (the QE inputs of the case `si_bands_transport`): the PAO Hamiltonian of the primitive cell
(matrix elements below `threshold = 1e-4` eV dropped), the cell doubled once along each lattice vector
(`doubling_Hamiltonian(nx=1, ny=1, nz=1)`: 8 cells, 16 atoms), an energy window, then the bands, the
DOS and PDOS, and the Boltzmann transport tensors, all without dense `(nk, nawf, nawf)` or
`(nawf, nawf, nR)` arrays. The generator does not write it: the sparse backend is meant for large
cells, where the dense drivers of `paoflow-gen` run out of memory. Here it takes under a minute.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials (from `PSEUDOS/library`) and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_sparse_backend` copies the case into `./si_sparse_backend` and prints
  the steps; run them in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in`
2. Non-self-consistent run on the full k grid: `pw.x -in nscf.in` (12×12×12, 18 bands)
3. Atomic projections: `projwfc.x -in proj.in` (`lwrite_overlaps`). `SparsePAOFLOW` reads them
   (`read_atomic_proj_QE`); it has no internal projections.
4. `mpirun -np 4 python main_sparse.py`: writes `output_sparse/`.

Pseudopotential (from `PSEUDOS/library`): `Si.pbe-tm-new-gipaw-v2.1.UPF`.

Runtime: `main_sparse.py` takes 40 s on 4 ranks and 2 minutes serially (Mac Studio, M2 Ultra), with
about 2 GB per rank; the QE runs are quick (2 atoms, 18 Ry).

## What to look at

- `output_sparse/sparse.log`: the bound on the eigenvalue shift the threshold can cause, the
  pre-flight memory projection of the doubling (18 → 144 orbitals) and the energy window (the bands
  kept and the solver chosen for H(k)).
- `output_sparse/bands_0.dat`, `dosdk_0.dat`, the PDOS files and the transport tensors
  (`sigma*`, `Seebeck*`, `kappa*`) of the doubled cell. A serial run (`python main_sparse.py`) must
  reproduce the files of the MPI run.

See `.agents/skills/sparse-backend/SKILLS.md` for the design and the memory contract of the backend.

Former location: `examples/recipes/sparse_backend`.
