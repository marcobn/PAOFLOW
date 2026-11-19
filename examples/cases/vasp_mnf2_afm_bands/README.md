# VASP MnF2: bands of a collinear antiferromagnet

Rutile MnF2 (2 Mn, 4 F) from VASP, collinear antiferromagnet (`ISPIN = 2`, `MAGMOM = 5 -5 4*0`):
PAO Hamiltonian on the POTCAR pseudo-orbitals and the all-electron tables (Mn `3P 3D 4S 4P 4D 4F`,
F `2P 3S 3P 3D 4S`; projectability threshold 0.90) and the band structures of both spins
(tetragonal path, `ibrav = 6`).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example vasp_mnf2_afm_bands` copies the case into `./vasp_mnf2_afm_bands` and prints
  the steps; run them in that folder.

Studio and `paoflow-gen` do not run VASP: do the VASP runs below first, with your own
POTCAR files, then the PAOFLOW step.

## Steps

1. VASP: `scf/` (4×4×6, `LCHARG`, `ISPIN = 2`, `MAGMOM`)
2. VASP: `nscf/` (8×8×12, `ICHARG = 11`, `LWAVE`, 36 bands) with the `CHGCAR` of `scf/`
3. `python main.py`: needs the VASP runs of nscf/ (vasprun.xml, WAVECAR, POTCAR); writes output/ and MnF2_VASP.png

The INCAR, KPOINTS and POSCAR files of each VASP run are in its folder; the POTCAR files are yours.

## What to look at

`MnF2_VASP.png`: all bands (left) and the top of the valence bands (right), up in red and down in
blue.

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

Former location: `examples/recipes/vasp/example04`.
