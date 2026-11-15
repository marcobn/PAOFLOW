# 3C-SiC — polar electron-phonon coupling (Fröhlich, quadrupoles) without EPW

Cubic silicon carbide is the polar example of the electron-phonon workflows. The Born
effective charges (Z* ≈ ±2.7) give the LO-TO splitting of the optical phonons and a
long-range electron-phonon coupling that diverges as 1/|q| (Fröhlich). Neither survives a
plain Fourier interpolation from the coarse DFPT grid, so PAOFLOW treats them as EPW's
`lpolar` does: the analytic dipole model built from ε∞ and Z* (Verdi and Giustino 2015;
Sjakste *et al.* 2015) is subtracted on the coarse grid, the short-range remainder is
interpolated, and the model is added back at every dense q, for the coupling and for the
force constants alike (`PAOFLOW.elphon.long_range`, developer note
`docs/internals/long-range-elphon-plan.md`). The dynamical quadrupoles (Brunin *et al.* 2020)
extend the model to the next order in q.

The whole chain runs from `pw.x` and `ph.x` alone: the coarse coupling is computed by
PAOFLOW from the `dvscf` files (`coupling_source='dfpt'`), and the nscf is run on the
irreducible wedge (`K_POINTS automatic`) and unfolded by symmetry. The workflow folders give

- `phonons/`: the dispersion and DOS with the LO-TO term (q2r + matdyn in PAOFLOW);
- `coupling/`: the dynamical quadrupoles from twelve single-q `ph.x` runs, and the
  band-resolved `|g(Γ, q)|` along Γ-X with and without the long-range model;
- `transport/`: the SERTA/MRTA electron and hole mobilities with the polar coupling.

## Directory contents

| file | purpose |
|------|---------|
| `system.py` | the configuration (`ElphonSystem`: prefix `sic`, `coupling_source='dfpt'`, 6³ k / 6³ q, `masses_amu = [28.0855, 12.0107]`, `nelec = 8`, `pthr = 0.90`, `long_range=True`, `quadrupoles=None`) and the defaults of `TRANSPORT`, `PHONONS`, `COUPLING` |
| `qe/phonon/scf.in` | `pw.x` scf (6³ k, 60 Ry, PBE norm-conserving, a = 8.237 bohr) |
| `qe/phonon/ph.in` | `ph.x` DFPT on the 6³ q-grid (16 irreducible q) with `epsil = .true.` (ε∞, Z*) and `fildvscf` |
| `qe/nscf/nscf.in` | `pw.x` nscf on the 6³ grid, `K_POINTS automatic` (16 irreducible k, 12 bands) |
| `qe/run_qe.sh`, `qe/kmesh.py` | the QE steps in order; the coarse-mesh editor |
| `phonons/phonons.py` | q2r + matdyn in PAOFLOW with the dipole term: dispersion, DOS, `sic.fc`, figure with and without the term |
| `coupling/long_range_check.py` | the band-resolved coupling at k = Γ along Γ-X with and without the model: the Fröhlich 1/|q| divergence of the LO mode (`--quadrupoles` adds the dipole + quadrupole curve) |
| `coupling/quadrupoles.py` | dynamical quadrupoles from twelve single-q `ph.x` density responses (`--write-inputs`, `--extract` → `coupling/quadrupole.fmt`), no ABINIT needed |
| `coupling/quadrupole.fmt` | the quadrupoles of this run (Q_Si = 7.49, Q_C = −3.01 e·bohr, EPW's format), so the quadrupole runs can be skipped |
| `transport/mobility.py` | SERTA/MRTA electron and hole mobilities with the polar coupling (`--no-long-range` for the plain interpolation; `--quadrupoles ../coupling/quadrupole.fmt` adds the dynamical quadrupoles; `--adaptive-smearing A` for velocity-dependent delta widths, see REPORT.md) |
| `transport/plot_mobility.py` | plots μ(T) and the rates |
| `REPORT.md` | results of the reference run (phonons vs `matdyn.x`, coupling vs EPW, mobilities with and without the quadrupoles) |

**Pseudopotentials:** copy `Si.upf` and `C.upf` (PseudoDojo nc-sr-05 PBE standard,
`PSEUDOS/nc-sr-05_pbe_standard_upf/`) into `qe/`; the inputs use `pseudo_dir = '../'`.

`paoflow-gen` (workflow `elphon`, semiconductor, workflows `phonons, coupling, transport`)
produces this layout from `qe/phonon/scf.in`; `ph.in` gets `epsil = .true.` for every
non-metal.

## Procedure

### 1. Quantum ESPRESSO

```bash
cd qe && NP=8 bash run_qe.sh        # or step by step:
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out          # 6^3 q, epsil: ~3 min on 8 cores
cd ../nscf
mkdir -p sic.save && cp ../phonon/sic.save/{charge-density.dat,data-file-schema.xml} sic.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out        # 16 irreducible k of the 6^3 grid, 12 bands
cd ../..
paoflow-genbasis-ps --pseudo-dir qe --out BASIS_PS --preset standard   # Si and C tables (26 orbitals)
```

`ph.x` writes `sic.dyn0` … `sic.dyn16` and `_ph0/` (the `dvscf` files and patterns);
`sic.dyn1` carries the dielectric tensor and the Born charges that the model needs. The EPW
route of the other examples (`epw.x` with `epbwrite`, `lpolar`) is not needed; it was used
only to validate this one (see the report).

### 2. PAOFLOW

```bash
python phonons/phonons.py                                  # phonons/output: dispersion + DOS with the LO-TO term
mpirun -np 8 python coupling/long_range_check.py           # coupling/output: |g_LO(Γ, q)|^2 along Γ-X, with/without
mpirun -np 8 python transport/mobility.py                  # transport/output: SERTA/MRTA mobilities, 36^3/36^3 (5 min)
mpirun -np 8 python transport/mobility.py --nk 48 --nq 48  # the X-valley electrons converge slowly: 48^3 and up (25 min)
mpirun -np 8 python transport/mobility.py --no-long-range  # what the plain interpolation gives (-> transport/output_plain)
mpirun -np 8 python transport/mobility.py --nk 48 --nq 48 --adaptive-smearing 0.7 --smearing-ev 0.001 --tag output_adaptive
                                                           # adaptive widths a |v| dq per final state (floor 1 meV)
python coupling/quadrupoles.py --write-inputs              # qe/phonon/quadrupoles/q*/ph.in + run.sh (needs the scf save)
(cd qe/phonon/quadrupoles && NP=8 bash run.sh)             # twelve single-q ph.x runs with fildrho, 10 s each
python coupling/quadrupoles.py --extract                   # Z* check and Q -> coupling/quadrupole.fmt
mpirun -np 8 python coupling/long_range_check.py --quadrupoles ../coupling/quadrupole.fmt   # adds the dipole + quadrupole curve
mpirun -np 8 python transport/mobility.py --nk 60 --nq 60 --quadrupoles ../coupling/quadrupole.fmt --tag output_quadrupole
                                                           # the complete long-range model (or set quadrupoles= in system.py)
```

The drivers detect the irreducible-wedge nscf and build the long-range model from
`qe/phonon/` (`LongRangeModel.from_coupling`); the phonons used by the dense drivers carry
the same model, which the library enforces. The coarse vertex g(R_e, R_p) is cached in
`cache/`, with and without the model, and shared by the workflows.

## What to look at

- `coupling/quadrupoles.py` / `--quadrupoles`: the dynamical quadrupoles (Q_Si = 7.49,
  Q_C = −3.01 e·bohr, against 7.41 / −2.63 of Poncé *et al.* 2021) raise the MRTA electron
  mobility by 25–40 % and the hole mobility by 20–30 %; the valence-band coupling at Γ moves
  by only ~1 % (see REPORT.md, sections 5 and 6).
- `phonons/phonons.py` prints ε∞ (7.51), the sum-rule-corrected Z* (±2.74) and the Γ
  frequencies in the limit along the first path segment: TO 790 and LO 952 cm⁻¹ with the
  model (experiment 796 and 972, Feldman *et al.* 1968), degenerate at 790 with the plain
  interpolation. `phonons/output/sic.fc` is a QE `flfrc` file with the `T` block, and
  `matdyn.x` reproduces the dispersion to 0.01 cm⁻¹.
- `coupling/long_range_check.py` writes `coupling/output/g_path.dat`: at |q| = 1/12 the LO
  coupling with the model is 15 times the plain interpolation and follows the dipole term
  alone (1/|q|² in |g|²); at the first coarse q-point the two routes coincide exactly. The
  sum over the four valence bands is gauge invariant.
- `transport/mobility.py`: the polar LO scattering is the dominant electron scattering
  channel (the electron mobility is five times larger without it); the X-valley electrons
  need dense grids (see the convergence table of the report), the holes converge faster.
  Without free-carrier screening the SERTA electron rate of the acoustic (piezoelectric-type)
  long-range coupling does not converge with the grid; MRTA does (REPORT.md, section 6).

Results and the validation against EPW and `matdyn.x` are in `REPORT.md`.

## References

- C. Verdi and F. Giustino, *Fröhlich electron-phonon vertex from first principles*, Phys. Rev. Lett. **115**, 176401 (2015); J. Sjakste, N. Vast, M. Calandra and F. Mauri, *Wannier interpolation of the electron-phonon matrix elements in polar semiconductors: Polar-optical coupling in GaAs*, Phys. Rev. B **92**, 054307 (2015) — the long-range dipole vertex subtracted and restored in the interpolation.
- H. Fröhlich, *Electrons in lattice fields*, Adv. Phys. **3**, 325 (1954).
- G. Brunin, H. P. C. Miranda, M. Giantomassi, M. Royo, M. Stengel, M. J. Verstraete, X. Gonze and G.-M. Rignanese, Phys. Rev. Lett. **125**, 136601 (2020); Phys. Rev. B **102**, 094308 (2020) — dynamical quadrupoles in the electron-phonon coupling.
- M. Royo and M. Stengel, Phys. Rev. X **9**, 021050 (2019) — the dynamical quadrupole tensor from the long-wavelength expansion of the density response.
- S. Poncé, F. Macheda, E. R. Margine, N. Marzari, N. Bonini and F. Giustino, *First-principles predictions of Hall and drift mobilities in semiconductors*, Phys. Rev. Research **3**, 043022 (2021) — quadrupoles and mobilities of 3C-SiC among others (Table IV).
- S. Poncé, E. R. Margine and F. Giustino, Phys. Rev. B **97**, 121201(R) (2018); S. Poncé, W. Li, S. Reichardt and F. Giustino, Rep. Prog. Phys. **83**, 036501 (2020) — SERTA/MRTA mobilities.
- X. Gonze and C. Lee, *Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from density-functional perturbation theory*, Phys. Rev. B **55**, 10355 (1997) — the non-analytic (LO-TO) term of the force constants, as in `q2r.x`/`matdyn.x`.
- D. W. Feldman, J. H. Parker, W. J. Choyke and L. Patrick, *Phonon dispersion curves by Raman scattering in SiC, polytypes 3C, 4H, 6H, 15R, and 21R*, Phys. Rev. **173**, 787 (1968) — measured ω_TO, ω_LO; Landolt-Börnstein, Group III, Vol. 41A1b (3C-SiC): ε∞ = 6.52, ε0 = 9.72.
- M. Roschke and F. Schwierz, *Electron mobility models for 4H, 6H, and 3C SiC*, IEEE Trans. Electron Devices **48**, 1442 (2001) — measured mobilities of 3C-SiC.
- J. R. Yates, X. Wang, D. Vanderbilt and I. Souza, Phys. Rev. B **75**, 195121 (2007) — adaptive smearing.
