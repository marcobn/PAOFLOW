# Unit-test fixtures

Small, tracked data files the unit tests read. The tests never read `examples/`: an example can
change or move without breaking a test, and CI checks out these files with the code. The
pseudopotentials the tests need live once in `PSEUDOS/library/` (see its README), not here.

Keep the fixtures small. Large QE data belong in the integration-test tarballs
(`tests/integration/*/README.md`) or behind an environment variable (below).

| Folder | Content | Source | Read by |
|---|---|---|---|
| `silicon/` | `scf.in`, `nscf.in`, `proj.in` and `silicon.save/` (`data-file-schema.xml`, `atomic_proj.xml`, the UPF) as pw.x and projwfc.x wrote them | `examples/qe_examples/example01` (bulk Si, 12³ nscf) | gen, gui, sparse, elphon FD tests |
| `silicon/Reference/` | bands, DOS, PDOS and Boltzmann outputs of the example01 `main.py`, as PAOFLOW wrote them in May 2026 (commit bb1b243e): legacy PDOS names (`0_pdosdk_0.dat`, ...), 500 transport energies. Kept for the readers, not as numbers of the current code (see below) | example01 `Reference/` | results, output readers |
| `platinum/Reference/` | `FermiSurf_0.bxsf`, `Spin_Berry_z_xy.bxsf` (every 2nd point of the 141 × 141 × 2 grid), `Fermi_surf_band_1/2.bxsf` (every 4th point) | example08 `Reference/` (Pt slab), subsampled | results (Fermi-surface meshes, BXSF slices), pyskeaf workflow |
| `output_formats/` | one file of each two-column output family: `ahcEf_xy.dat`, `epsi_xx.dat`, `epsr_xx.dat`, `shcEf_z_xy.dat` | examples 04, 06, 08 `Reference/` | output readers |
| `siliconSC/` | Slater-Koster parameters and geometry of Si | `examples/cases/si_supercell_unfold` (was `examples/unfold_examples/siliconSC`): the parameters; the geometry files live only here | models (builder conventions) |
| `snte/` | `data-file-schema.xml` of the SnTe monolayer (2D, spin-orbit) | `examples/qe_examples/example17` | gen (smart defaults) |

`silicon/Reference/dosdk_0.dat` and the `pdosdk` files differ from the current code by up to 1.7 % of the
maximum DOS (bands agree to 1e-8 eV). The change is commit 3a3c3c0d (15 Nov 2026, `resolvable_widths` in
`spectrum/do_dos.py`): an adaptive width below 1e-6 eV, a state with zero velocity on a grid point of high
symmetry (Gamma, X, L of the 24^3 grid), now takes the energy spacing, so the state counts once; before it
contributed only when its energy fell on the grid and was otherwise lost (0.04 % of the states here). The
reader tests do not depend on the values.

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
