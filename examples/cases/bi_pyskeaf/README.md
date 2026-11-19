# Bi: de Haas-van Alphen frequencies with pyskeaf

Rhombohedral bismuth (A7, two atoms, scalar-relativistic PBE): PAOFLOW builds the PAO Hamiltonian
on its internal projections, interpolates it on a 32×32×32 grid, writes the Fermi surface as
single-band BXSF files and runs `pyskeaf` (the Python port of SKEAF) on them: the extremal
Fermi-surface orbits, their de Haas-van Alphen frequencies, cyclotron masses and curvatures for a
magnetic field rotated in the Cartesian xz plane from z, the trigonal axis (φ = 0°), to x, a
binary axis (φ = 90°), in 37 steps.

The BXSF files of that run come with the case (`bxsf/`), so the pyskeaf step runs without QE.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies the
  inputs, the scripts, the shipped BXSF files and the pseudopotential, and lists the runs below in
  the Runs tab.
- **Command line**: `paoflow-gen --example bi_pyskeaf` copies the case into `./bi_pyskeaf`; then
  run the steps below in that folder.

## Steps

1. Self-consistent run (18×18×18 k grid; PAOFLOW reads this save): `pw.x -in Bi2.scf.in`
2. PAO Hamiltonian, Fermi surface and pyskeaf: `mpirun -np 4 python main.py`. It generates the
   pseudo-atomic basis in `BASIS_PS` from the UPF, then writes `output/Fermi_surf_band_<n>.bxsf`,
   `output/qo_EF_0_freqvsangle_<n>.out` and `output/plot_frequencies.png`.
3. pyskeaf alone, on the shipped Fermi surface: `mpirun -np 4 python main_skeaf.py bxsf` (about
   40 s on 4 ranks; no QE run needed). The results go to `bxsf/`. `python main_skeaf.py` without
   an argument reruns pyskeaf on the BXSF files of `output/` (step 2), e.g. after editing the
   `PYSKEAF` settings in `main.py`.

The command-line tool reads the Fortran SKEAF input `config.in` (one band, here file 3 of
`output/`): `pyskeaf config.in --out output` after step 2, or
`pyskeaf config.in --bxsf bxsf/Fermi_surf_band_3.bxsf --out bxsf` (about 25 s on 4 ranks with
`mpirun`); it writes `qo_EF_0_freqvsangle.out` and the short, long and orbit-outline files.

Runtime: steps 1 and 2 take minutes on 4 cores; step 3 under a minute.

## What to look at

- `qo_EF_0_freqvsangle_<n>.out`: one row per extremal orbit and field direction (azimuth θ,
  polar angle φ, frequency in kT, cyclotron mass in mₑ, curvature, electron or hole, number of
  supercell copies of the orbit). The file number `n` is the position of the BXSF file; its
  `BAND:` line gives the band of the run: files 1-5 are bands 14-18.
- Only bands 15 and 16 cross the Fermi level. Band 16 (file 3) gives the three electron pockets:
  their frequencies against φ are in `plot_frequencies.png`, one value near 0.93 kT for the
  field along the trigonal axis (the three pockets are equivalent there), splitting into
  branches between 0.2 and 1.3 kT as the field turns to the binary axis. Band 15 (file 2)
  crosses the Fermi level but gives no closed extremal orbit at `num_interpolation = 60`.
- The pockets of PBE bismuth without spin-orbit coupling are much larger than the measured
  ones (a few tesla): this is a workflow example, not a converged Bi calculation.

`Reference/` (not copied with the case) holds the all-band `FermiSurf_0.bxsf`, the
`Fermi_surf_band_<n>_0.npz` arrays and SKEAF-style `results_*_<n>.out` files from a July 2026
version of pyskeaf; pyskeaf has changed since, and its orbits differ from those of the current
code on the same BXSF files.

Pseudopotential (from `PSEUDOS/library`): `Bi_PseudoDojo05_PBE_sr.upf` (PseudoDojo nc-sr-05 PBE
standard `Bi.upf`).
