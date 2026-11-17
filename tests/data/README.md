# Unit-test fixtures

Small, tracked data files the unit tests read. The tests never read `examples/`: an example can
change or move without breaking a test, and CI checks out these files with the code. The
pseudopotentials the tests need live once in `PSEUDOS/library/` (see its README), not here.

Keep the fixtures small. Large QE data belong in the integration-test tarballs
(`tests/integration/*/README.md`) or behind an environment variable (below).

| Folder | Content | Source | Read by |
|---|---|---|---|
| `silicon/` | `scf.in`, `nscf.in`, `proj.in` and `silicon.save/` (`data-file-schema.xml`, `atomic_proj.xml`, the UPF) as pw.x and projwfc.x wrote them | `examples/qe_examples/example01` (bulk Si, 12³ nscf) | gen, gui, sparse, elphon FD tests |
| `silicon/Reference/` | bands, DOS, PDOS and Boltzmann outputs of the example01 `main.py` | example01 `Reference/` | results, output readers |
| `platinum/Reference/` | `FermiSurf_0.bxsf`, `Spin_Berry_z_xy.bxsf` (every 2nd point of the 141 × 141 × 2 grid), `Fermi_surf_band_1/2.bxsf` (every 4th point) | example08 `Reference/` (Pt slab), subsampled | results (Fermi-surface meshes, BXSF slices), pyskeaf workflow |
| `output_formats/` | one file of each two-column output family: `ahcEf_xy.dat`, `epsi_xx.dat`, `epsr_xx.dat`, `shcEf_z_xy.dat` | examples 04, 06, 08 `Reference/` | output readers |
| `siliconSC/` | Slater-Koster parameters and geometry of Si | `examples/unfold_examples/siliconSC` | models (builder conventions) |
| `snte/` | `data-file-schema.xml` of the SnTe monolayer (2D, spin-orbit) | `examples/qe_examples/example17` | gen (smart defaults) |

The subsampled BXSF files keep the header, the band labels and the spanning vectors of the
originals: a general grid includes both ends, so taking every n-th point along the first two axes
of an `(n1, n2, n3)` grid with `(n1 - 1) % n == 0` gives a coarser grid over the same cell.

## Optional large data

| Variable | Points to | Enables |
|---|---|---|
| `PAOFLOW_TEST_SNTE_SAVE` | a `SnTe.save` folder (pw.x run of the example17 inputs, about 1 GB) | `projection/test_spinor_augmentation.py`: spinor metric of the SnTe bands |
| `PAOFLOW_LARGE_MPI_TEST=1` | (flag) | `utils/test_communication.py`: collectives above 2 GB, about 5 GB of memory |

## Adding a fixture

Copy only the files the test reads, trim them when the test allows it, and add a row to the table
above. QE saves and XML files under `tests/data/` are tracked through the exceptions at the end of
the repository `.gitignore`.
