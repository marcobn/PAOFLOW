# Monolayer GeS: shift current and SHG of a 2D polar crystal

Structure: Pmn2₁ monolayer GeS, the puckered (phosphorene-like) monochalcogenide of Wang et al.
2017 and Rangel et al. 2017. The in-plane cell is fixed at the PBE literature values
(a = 4.40 Å armchair along x, b = 3.68 Å zigzag along y) and the positions were relaxed with PBE
(`relax.in`; PseudoDojo norm-conserving pseudopotentials, 60 Ry, 12×12×1 k, 18 Å cell with
`assume_isolated='2D'`); `scf.in` and `nscf.in` carry the relaxed positions, so the relaxation
need not be repeated.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the inputs and the pseudopotentials and lists the runs below in the Runs tab.
- **Command line**: `paoflow-gen --example ges_nlo` copies the case into `./ges_nlo`; then run
  the steps below in that folder.

## Steps

1. Self-consistent run: `pw.x -in scf.in` (`mpirun -np 8 pw.x -nk 4 -in scf.in > scf.out`)
2. Non-self-consistent run on the full 18×18×1 grid (`nosym`, `noinv`, nbnd 40):
   `pw.x -in nscf.in` (`mpirun -np 8 pw.x -nk 2 -in nscf.in > nscf.out`)
3. PAO basis tables from the pseudopotentials (seconds):
   `paoflow-genbasis-ps --pseudo Ge_PseudoDojo05_PBE_sr.upf --out BASIS_PS --preset standard`
4. `paoflow-genbasis-ps --pseudo S_PseudoDojo05_PBE_sr.upf --out BASIS_PS --preset standard`
5. PAOFLOW: `mpirun -np 8 python main.py` (120×120 k: a few minutes on 8 ranks; 200×200 took
   13 min).

`main.py` builds the PAO Hamiltonian with the internal projections on the `BASIS_PS` tables (62
orbitals, 26 bands with projectability > 0.95: all 20 valence + 6 conduction; the repository
all-electron `BASIS` tables give only 0.90 for the Ge 3d bands with these pseudopotentials) and
calls `PAOFLOW.nonlinear_optics(nk=(120, 120, 1), twoD=True, ...)`, which writes the sheet
responses: `shift_<abc>.dat` (nm µA/V²), `shg{r,i,a}_<abc>.dat` (pm²/V), the injection currents.

## What to look at

Symmetry (point group mm2, polar axis x): σ^{xxx}, σ^{xyy}, σ^{yxy} are allowed;
σ^{yyy} = σ^{xxy} = 0 (mirror y → −y); the T-odd files (`shiftm`, `injectm`) vanish. Targets:
the shift current σ^{xxx}, σ^{xyy} of Wang et al. 2017 Fig. 1(c) and Rangel et al. 2017 (peaks
of a few hundred nm µA/V² at the PBE/GW gap), and the SHG of Wang et al. Converge `NK`
(120 → 300) and `ETA` before quoting numbers.

The nscf uses `nosym`/`noinv` so that the full 18×18×1 grid is computed directly (a
symmetry-reduced nscf expanded by PAOFLOW is equivalent). The norm-conserving pseudopotentials
make the projections exact (ultrasoft / PAW ones are projected in the augmented metric since
October 2026; before that they broke the point symmetry at a few k-points, GaAs: 2–5 % between
equivalent components, see `REPORT.md` section 3.2 in the case `gaas_nlo`); here the
mirror-forbidden components vanish to 1e-9.

Reference run (October 2026, 18×18×1 nosym nscf, 200×200 dense grid, eta 0.05 eV, no scissor,
PBE gap 1.68 eV): sheet shift current peaks σ^{xxx} = 46, σ^{xyy} = 48, σ^{yxy} = 26 nm µA/V²
(at 2.8, 2.8 and 2.05 eV; divided by a 0.6 nm layer thickness ≈ 77 and 80 µA/V² for xxx and xyy, the order of magnitude
of Rangel 2017 / Wang 2017); the forbidden yyy and xxy components and the T-odd files vanish
(< 1e-8). 13 min on 8 ranks. Fig. 3 of the validation report (`make_report_figures.py` of the case
`gaas_nlo`) is drawn from `output/` when the two cases sit in the same parent folder.

Pseudopotentials (from `PSEUDOS/library`): `Ge_PseudoDojo05_PBE_sr.upf`,
`S_PseudoDojo05_PBE_sr.upf` (the `Ge.upf` and `S.upf` of the PseudoDojo nc-sr-05 PBE standard set,
`PSEUDOS/nc-sr-05_pbe_standard_upf`).
