# VASP Pt: spin-orbit bands and spin Hall conductivity

Fcc Pt from VASP with spin-orbit coupling (`LSORBIT = .TRUE.`): PAOFLOW projects on the POTCAR
pseudo-orbitals and the all-electron tables (`5D 6S 6P 7S 7P`), builds the PAO Hamiltonian, the band
structure, then the interpolated Hamiltonian, velocities and adaptive smearing (Methfessel-Paxton)
for the spin Hall conductivity $\sigma^z_{xy}$ between -8 and 4 eV.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example vasp_pt_spin_hall` copies the case into `./vasp_pt_spin_hall` and prints
  the steps; run them in that folder.

Studio and `paoflow-gen` do not run VASP: do the VASP runs below first, with your own
POTCAR files, then the PAOFLOW step.

## Steps

1. VASP: `scf/` (`LCHARG`, `LSORBIT`)
2. VASP: `nscf/` (12×12×12, `ICHARG = 11`, `LWAVE`, 20 bands) with the `CHGCAR` of `scf/`
3. `python main.py`: needs the VASP runs of nscf/ (vasprun.xml, WAVECAR, POTCAR); writes output/ (shcEf_z_xy.dat), Pt_soc_VASP.png and Pt_shc_VASP.png

The INCAR, KPOINTS and POSCAR files of each VASP run are in its folder; the POTCAR files are yours.

## What to look at

`Pt_soc_VASP.png` (bands) and `Pt_shc_VASP.png`: the spin Hall conductivity
`output/shcEf_z_xy.dat` against the Fermi level; compare with the QE case `pt_spin_hall`. With these
tables 12 of the 20 bands pass the projectability threshold 0.95.

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

Former location: `examples/recipes/vasp/example02`.
