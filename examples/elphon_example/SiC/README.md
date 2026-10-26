# 3C-SiC — polar electron-phonon coupling (Fröhlich) without EPW

Cubic silicon carbide is a polar semiconductor: the Born effective charges
(Z* ≈ ±2.7) give the LO-TO splitting of the optical phonons and a long-range
electron-phonon coupling that diverges as 1/|q| (Fröhlich). Neither survives a
plain Fourier interpolation from the coarse DFPT grid, so PAOFLOW treats them
as EPW's `lpolar` does: the analytic dipole model built from ε∞ and Z* is
subtracted on the coarse grid, the short-range remainder is interpolated, and
the model is added back at every dense q, for the coupling and for the force
constants alike (`PAOFLOW.elphon.long_range`, developer note
`docs/internals/long-range-elphon-plan.md`).

This example runs the whole chain from `pw.x` and `ph.x` alone: the coarse
coupling is computed by PAOFLOW from the `dvscf` files, and the nscf is run on
the irreducible wedge (`K_POINTS automatic`) and unfolded by symmetry.

## Directory contents

| file | purpose |
|------|---------|
| `phonon/scf.in` | `pw.x` scf (6³ k, 60 Ry, PBE norm-conserving, a = 8.237 bohr) |
| `phonon/ph.in` | `ph.x` DFPT on the 6³ q-grid (16 irreducible q) with `epsil = .true.` (ε∞, Z*) and `fildvscf` |
| `nscf/nscf.in` | `pw.x` nscf on the 6³ grid, `K_POINTS automatic` (16 irreducible k, 12 bands) |
| `phonons.py` | q2r + matdyn in PAOFLOW with the dipole term: dispersion, DOS, `sic.fc`, figure with and without the term |
| `coupling.py` | the band-resolved coupling at k = Γ along Γ-X with and without the model: the Fröhlich 1/|q| divergence of the LO mode |
| `mobility.py` | SERTA/MRTA electron and hole mobilities with the polar coupling (`--no-long-range` for the plain interpolation; `--quadrupoles quadrupole.fmt` adds the dynamical quadrupoles; `--adaptive-smearing A` for velocity-dependent delta widths, see REPORT.md) |
| `quadrupoles.py` | dynamical quadrupoles from twelve single-q `ph.x` density responses (`--write-inputs`, `--extract` → `quadrupole.fmt`), no ABINIT needed |
| `quadrupole.fmt` | the quadrupoles of this run (Q_Si = 7.49, Q_C = −3.01 e·bohr², EPW's format), so the quadrupole runs can be skipped |
| `REPORT.md` | results of the reference run (phonons vs `matdyn.x`, coupling vs EPW, mobilities) |

**Pseudopotentials:** copy `Si.upf` and `C.upf` (PseudoDojo nc-sr-05 PBE standard,
`PSEUDOS/nc-sr-05_pbe_standard_upf/`) into this directory; both inputs use `pseudo_dir = '../'`.

## Procedure

### 1. Quantum ESPRESSO

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out          # 6^3 q, epsil: ~3 min on 8 cores
cd ../nscf
mkdir -p sic.save && cp ../phonon/sic.save/{charge-density.dat,data-file-schema.xml} sic.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out        # 16 irreducible k of the 6^3 grid, 12 bands
cd ..
paoflow-genbasis-ps --pseudo-dir . --out BASIS_PS --preset standard   # Si and C tables (26 orbitals)
```

`ph.x` writes `sic.dyn0` … `sic.dyn16` and `_ph0/` (the `dvscf` files and patterns);
`sic.dyn1` carries the dielectric tensor and the Born charges that the model needs. The
EPW route of the other examples (`epw.x` with `epbwrite`, `lpolar`) is not needed; it was
used only to validate this one (see the report).

### 2. PAOFLOW

```bash
python phonons.py                                 # output/phonons: dispersion + DOS with the LO-TO term
mpirun -np 8 python coupling.py                   # output/coupling: |g_LO(Γ, q)|^2 along Γ-X, with/without
mpirun -np 8 python mobility.py                   # output/mobility: SERTA/MRTA mobilities, 36^3/36^3 (5 min)
mpirun -np 8 python mobility.py --nk 48 --nq 48   # the X-valley electrons converge slowly: 48^3 and up (25 min)
mpirun -np 8 python mobility.py --no-long-range   # what the plain interpolation gives
mpirun -np 8 python mobility.py --nk 48 --nq 48 --adaptive-smearing 0.7 --smearing-ev 0.001
                                                  # adaptive widths a |v| dq per final state (floor 1 meV)
python quadrupoles.py --write-inputs              # phonon/quadrupoles/q*/ph.in + run.sh (needs the scf save)
(cd phonon/quadrupoles && NP=8 bash run.sh)       # twelve single-q ph.x runs with fildrho, 10 s each
python quadrupoles.py --extract                   # Z* check and Q -> quadrupole.fmt
mpirun -np 8 python coupling.py --quadrupoles quadrupole.fmt   # adds the dipole + quadrupole curve
mpirun -np 8 python mobility.py --nk 60 --nq 60 --quadrupoles quadrupole.fmt   # the complete long-range model
```

The drivers detect the irreducible-wedge nscf and build the long-range model from
`phonon/` (`LongRangeModel.from_coupling`); the phonons used by the dense drivers carry
the same model, which the library enforces.

## What to look at

- `phonons.py` prints ε∞ (7.51), the sum-rule-corrected Z* (±2.74) and the Γ frequencies in
  the limit along the first path segment: TO 790 and LO 952 cm⁻¹ with the model (experiment
  796 and 972), degenerate at 790 with the plain interpolation. `output/phonons/sic.fc` is a
  QE `flfrc` file with the `T` block, and `matdyn.x` reproduces the dispersion to 0.01 cm⁻¹.
- `coupling.py` writes `output/coupling/g_path.dat`: at |q| = 1/12 the LO coupling with the
  model is 15 times the plain interpolation and follows the dipole term alone (1/|q|² in
  |g|²); at the first coarse q-point the two routes coincide exactly. The sum over the four
  valence bands is gauge invariant.
- `mobility.py`: the polar LO scattering is the dominant electron scattering channel (the
  electron mobility is five times larger without it); the X-valley electrons need dense grids
  (see the convergence table of the report), the holes converge faster.

Results and the validation against EPW and `matdyn.x` are in `REPORT.md`.
