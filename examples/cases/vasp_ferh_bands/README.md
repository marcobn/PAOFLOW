# VASP FeRh: ferromagnetic bands, spin-polarized and with spin-orbit

B2 (CsCl) FeRh from VASP, ferromagnetic: one chain of runs collinear spin-polarized
(`ISPIN = 2`, `MAGMOM = 3 0`), one with spin-orbit coupling and the moment along z
(`LSORBIT = T`, `MAGMOM = 0 0 3 0 0 0`). Each gives a PAO Hamiltonian on the POTCAR pseudo-orbitals
and the all-electron tables (Fe `3D 4S 4P 4D 5S 5P`, Rh `4D 4F 5S 5P 5D`; projectability threshold
0.90) and its band structure (simple cubic path).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example vasp_ferh_bands` copies the case into `./vasp_ferh_bands` and prints
  the steps; run them in that folder.

Studio and `paoflow-gen` do not run VASP: do the VASP runs below first, with your own
POTCAR files, then the PAOFLOW step.

## Steps

1. VASP: `scf_nspin2/` (8×8×8, `LCHARG`, `ISPIN = 2`), then `nscf_nspin2/` (12×12×12, `ICHARG = 11`, `LWAVE`, 16 bands) with its `CHGCAR`
2. VASP: `scf_soc/` (8×8×8, `LCHARG`, `LSORBIT`), then `nscf_soc/` (12×12×12, `ICHARG = 11`, `LWAVE`, 32 bands) with its `CHGCAR`
3. `python main_nspin2.py`: needs the VASP runs of nscf_nspin2/ (vasprun.xml, WAVECAR, POTCAR); writes output_nspin2/ and FeRh_nspin2_VASP.png
4. `python main_soc.py`: needs the VASP runs of nscf_soc/ (vasprun.xml, WAVECAR, POTCAR); writes output_soc/ and FeRh_soc_VASP.png

The INCAR, KPOINTS and POSCAR files of each VASP run are in its folder; the POTCAR files are yours.

## What to look at

`FeRh_nspin2_VASP.png` (up and down bands) and `FeRh_soc_VASP.png` (the spinor bands, both spins
in one set).

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

Former location: `examples/recipes/vasp/example06`.
