# VASP Si: PAO bands, non-spin-polarized and spin-polarized

Bulk silicon from VASP: PAOFLOW projects the PAW wavefunctions on the pseudo-orbitals of the POTCAR
and the all-electron radial tables (`3S 3P 3D 4S 4P 4D`), builds the PAO Hamiltonian and its band
structure (fcc path, 500 points), once from a non-spin-polarized nscf run and once from a
spin-polarized one (`ISPIN = 2`, `MAGMOM = 0 0`).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example vasp_si_bands` copies the case into `./vasp_si_bands` and prints
  the steps; run them in that folder.

Studio and `paoflow-gen` do not run VASP: do the VASP runs below first, with your own
POTCAR files, then the PAOFLOW step.

## Steps

1. VASP: `scf/` (6×6×6, `LCHARG`), non-spin-polarized
2. VASP: `nscf_nspin1/` (12×12×12, `ICHARG = 11`, `LWAVE`, 18 bands) with the `CHGCAR` of `scf/`
3. VASP: `nscf_nspin2/` (the same with `ISPIN = 2`) with the `CHGCAR` of `scf/`
4. `python main_nspin1.py`: needs the VASP runs of nscf_nspin1/ (vasprun.xml, WAVECAR, POTCAR); writes output_nspin1/ and Si_nspin1_VASP.png
5. `python main_nspin2.py`: needs the VASP runs of nscf_nspin2/ (vasprun.xml, WAVECAR, POTCAR); writes output_nspin2/ and Si_nspin2_VASP.png

The INCAR, KPOINTS and POSCAR files of each VASP run are in its folder; the POTCAR files are yours.

## What to look at

`Si_nspin1_VASP.png` and `Si_nspin2_VASP.png`: the two band structures coincide (up and down
spins degenerate). With these tables 15 of the 18 bands pass the projectability threshold 0.95
(`docs/internals/internal-basis.md` compares the presets on this run).

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

Former location: `examples/recipes/vasp/example01`.
