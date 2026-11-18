# VASP: notebook of the seven VASP cases

`VASP_examples.ipynb` walks through the seven VASP cases cell by cell, with interactive plots:

- `vasp_si_bands`
- `vasp_pt_spin_hall`
- `vasp_fe_anomalous_hall`
- `vasp_mnf2_afm_bands`
- `vasp_mn3ir_noncollinear_bands`
- `vasp_ferh_bands`
- `vasp_cri3_monolayer_bands`

Each section is the driver of one case split into cells (constructor, projections, projectability,
Hamiltonian, bands, Hall conductivities, figures). The stored outputs predate the augmented PAW metric
of November 2026 (projections in the metric $S = 1 + q|\beta\rangle\langle\beta|$): rerun the cells
for current numbers.

## Run it

The notebook reads the VASP runs from the folders of the cases next to its own
(`../vasp_si_bands/nscf_nspin1/`, ...). Copy the cases side by side under their names (as they sit in
`examples/cases`, or with `paoflow-gen --example <name>` run in one folder), do the VASP runs of each
(see their READMEs), then open the notebook: `jupyter lab VASP_examples.ipynb`. A section runs as soon
as its case has its VASP runs. PAOFLOW Studio shows this step but does not run it (it needs a
browser session of Jupyter).

## Notes on the VASP runs

- PAOFLOW reads `vasprun.xml`, `WAVECAR` and `POTCAR` from the run folder (`savedir`). The POTCAR
  files are licensed and not distributed with PAOFLOW: put one in each run folder, or point
  `PAOFLOW_POTCAR_DIR` at their folder.
- Run the scf (`LCHARG = .TRUE.`) on a coarse k-mesh, copy its `CHGCAR` into the nscf folder and run
  the nscf (`ICHARG = 11`, `LWAVE = .TRUE.`) on the finer mesh, as the INCAR files here do; one scf
  with `LWAVE = .TRUE.` on the fine mesh works too. Use Gamma-centred meshes and `LREAL = .FALSE.`
- The runs used `ISYM = 2`; `ISYM = -1` and `0` work too (with `LSORBIT = T` and in magnetic systems
  they are equivalent: the full grid). PAOFLOW needs spglib with VASP: check the space group it
  prints.
- Magnetic systems need `MAGMOM` in the INCAR: PAOFLOW reads it from `vasprun.xml` to find the
  magnetic symmetry.
- Keep `pao_hamiltonian(expand_wedge=True)` (the default) even with `ISYM = -1`: VASP's k-points may
  differ from PAOFLOW's and need the mapping.
- `projections(basispath=..., configuration={...})` takes the pseudo-orbitals from the POTCAR and the
  other shells of the configuration from the all-electron tables `BASIS/` of the PAOFLOW source
  checkout (the drivers find it next to `src/`), in the augmented PAW metric. Without `basispath`
  the other shells are generated from the POTCAR instead (`docs/internals/internal-basis.md`).
- The drivers plot the bands kept in memory, which MPI splits over the ranks: run them serially
  (the bands files in `output*/` are complete either way).

Former location: `examples/recipes/vasp` (README and notebook).
