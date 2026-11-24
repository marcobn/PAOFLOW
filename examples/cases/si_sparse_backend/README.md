# Si: the sparse backend on a doubled cell

Bulk silicon through `PAOFLOW(sparse=True)`, the route for cells whose Hamiltonian on the k-mesh
does not fit in memory: the PAO Hamiltonian is kept as a list of hoppings truncated at the 36th
neighbour shell, the cell is doubled along the three lattice vectors (16 atoms), and its folded
bands, DOS and Boltzmann transport tensors are computed in one pass over the k-mesh, then checked
against the dense pipeline on the 2-atom cell. Two further scripts compare the truncations of the
Hamiltonian and the choices of the states solved per k-point.

## Run it

- **PAOFLOW Studio**: *Examples…* → `si_sparse_backend`. Studio copies the case, lists its
  scripts and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_sparse_backend` copies the case into
  `./si_sparse_backend`; run the steps below in that folder.
- **Python**: `main.py` is the workflow; `sparse_truncation.py` and `sparse_states.py` are the
  variations of the two how-tos. Read them to change a call.

To skip the QE runs, copy the case with `paoflow-gen --example si_sparse_backend --with-data`
(in Studio, *Use the precomputed data of the case*): it extracts `silicon.save`, with the projections of
`projwfc.x`, from `si_sparse_backend-v1.tar.gz` of the release
[case-data](https://github.com/marcobn/PAOFLOW/releases/tag/case-data).

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Self-consistent run | `pw.x -in scf.in` | quick (2 atoms, 18 Ry) |
| Non-self-consistent run on the full k grid | `pw.x -in nscf.in` | quick |
| Atomic projections (read by the sparse run) | `projwfc.x -in proj.in` | quick |
| Sparse backend: doubled cell, bands, DOS and transport, checked against the dense pipeline | `mpirun -np 4 python main.py` | 2.5 min as one process with 4 threads (busy 20-core node) |
| Truncation of the sparse Hamiltonian: bond_order, rcut, hopping_threshold (how-to) | `python sparse_truncation.py > sparse_truncation.log` | 40 s, one process, 4 threads |
| States solved per k-point: the derived window, nev, interior_window (how-to) | `python sparse_states.py` | 75 s, one process, 4 threads |

## Outputs

- `output/sparse.log`: the truncation and its bound on the eigenvalue shift, the doubling and its
  memory, and the energy window of each pass with the states solved per k-point.
- `output/bands_0.dat`: the folded bands of the 16-atom cell along the fcc path.
- `output/dosdk_0.dat`: the DOS of the 16-atom cell (energy, DOS).
- `output/sigmagauss_0.dat`, `Seebeckgauss_0.dat`, `kappagauss_0.dat`, `PFgauss_0.dat`: the
  transport tensors of the 16-atom cell, in units of the relaxation time.
- `output/silicon_bo36.npz`: the truncated Hamiltonian of the 2-atom cell, which the dense check
  loads.
- `output_dense/`: the DOS and transport tensors of the 2-atom cell on the dense pipeline.
- `output_full/`, `output_bo*/`, `output_rcut30/`, `output_t1e-0*/`: the bands under each
  truncation.
- `output_auto/`, `output_nev/`, `output_interior/`, `output_interior_auto/`: the runs under each
  choice of the states solved.

Explained in: [Large cells on the sparse backend](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial04.html),
[How to choose the truncation of the sparse Hamiltonian](https://paoflow.readthedocs.io/en/latest/learn/howtos/sparse-truncation.html),
[How to choose which states the sparse backend solves](https://paoflow.readthedocs.io/en/latest/learn/howtos/sparse-states.html)
