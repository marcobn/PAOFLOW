# VASP Mn3Ir: bands of a noncollinear antiferromagnet

Cubic Mn3Ir (1 Ir, 3 Mn) from VASP, with the 120° noncollinear antiferromagnetic order in the
(111) plane and spin-orbit coupling (`LSORBIT = .TRUE.`,
`MAGMOM = 3*0 2 -1 -1 -1 2 -1 -1 -1 2`): PAO Hamiltonian on the POTCAR pseudo-orbitals and the
all-electron tables (Ir `5P 5D 6S 6P 7S`, Mn `3P 3D 4S 4P 4D`; projectability threshold 0.90) and its
band structure (simple cubic path).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example vasp_mn3ir_noncollinear_bands` copies the case into `./vasp_mn3ir_noncollinear_bands` and prints
  the steps; run them in that folder.

Studio and `paoflow-gen` do not run VASP: do the VASP runs below first, with your own
POTCAR files, then the PAOFLOW step.

## Steps

1. VASP: `scf/` (6×6×6, `LCHARG`, `LSORBIT`, `MAGMOM`)
2. VASP: `nscf/` (10×10×10, `ICHARG = 11`, `LWAVE`, 40 bands) with the `CHGCAR` of `scf/`
3. `python main.py`: needs the VASP runs of nscf/ (vasprun.xml, WAVECAR, POTCAR); writes output/ and Mn3Ir_VASP.png

The INCAR, KPOINTS and POSCAR files of each VASP run are in its folder; the POTCAR files are yours.

## What to look at

`Mn3Ir_VASP.png`: the band structure. PAOFLOW finds the magnetic space group from `MAGMOM`:
check the one it prints.

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

The notebook of the case `vasp_overview` runs the same steps cell by cell, with stored figures.

Former location: `examples/recipes/vasp/example05`.
