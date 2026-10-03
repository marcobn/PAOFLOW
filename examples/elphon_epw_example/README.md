# Pb electron–phonon (Eliashberg) — PAOFLOW on EPW's coarse coupling

This example computes the isotropic Eliashberg properties of fcc lead (α²F, λ,
ω_log, Tc) with PAOFLOW's PAO interpolation. The coarse coupling comes from
**EPW** (`epbwrite`). EPW evaluates both ψ_k and ψ_{k+q} from the nscf save, so
its matrix elements share the band gauge of PAOFLOW's projections on that save,
which the interpolation needs (see `docs/internals/Electron-Phonon-Coupling.md`).

The QE steps are those of the EPW tutorial 04 (Pb superconductivity,
<https://docs.epw-code.org/tutorials/tutorial_04/index.html>): 6³ coarse k and
q grids, with ph.x on the 16 irreducible q only. EPW is run **without Wannier
functions** (`wannierize = .false.`): it serves only to compute the coarse
matrix elements, and PAOFLOW does all the interpolation.

---

## Directory contents

| file | purpose |
|------|---------|
| `phonon/scf.in` | `pw.x` scf (8³ k, with symmetry) |
| `phonon/ph.in` | `ph.x` DFPT on the 6³ q-grid (irreducible q, `fildvscf`) |
| `epw/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 16`) |
| `epw/epw.in` | `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, `exclude_bands = 1:5` |
| `epw/write_ukk.py` | writes `pb.ukk`, the band bookkeeping EPW reads when `wannierize = .false.` |
| `epw/epw_wannier.in` | the tutorial's Wannierized EPW input (alternative; same `.epb` content) |
| `main.py` | PAOFLOW analysis: PAO electronic structure + Eliashberg on EPW's coupling |

**Pseudopotential:** copy `Pb.upf` (ONCVPSP, from the EPW tutorial material) into
this directory. Both `phonon/` and `epw/` use `pseudo_dir = '../'`.

---

## Procedure

### 1. Quantum ESPRESSO and EPW

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
python3 /path/to/q-e/EPW/bin/pp.py          # prefix 'pb' -> collects dvscf, patterns, dyn into save/

cd ../epw
mpirun -np 8 pw.x -in nscf.in > nscf.out
python3 write_ukk.py                             # pb.ukk for wannierize = .false.
mpirun -np 8 epw.x -nk 8 -in epw.in > epw.out   # EPW: one process per pool (-np = -nk)
```

`epw.x` writes `pb.epb1 … pb.epbN` (the coarse Bloch coupling, one file per
pool), which together with `pb.ukk` is all PAOFLOW reads. EPW then continues
into its Wannier stage with identity rotations; that output is not used.
`../phonon/save/pb.phsave/patterns.1.xml` must exist (created by `pp.py`),
otherwise `epw.x` stops with "cannot open file for reading or writing".
Tested with QE/EPW 7.5 (EPW 6.0). The coarse-coupling cost scales with the
number of pools and with the kept bands, so use as many pools as k-points allow.

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo Pb.upf --out BASIS_PS
```

`main.py` uses the `standard` configuration: 5d 6s 6p plus 6d 7s 7p, 18 orbitals.

### 3. PAOFLOW analysis

```bash
mpirun -np 8 python main.py               # dense k 48^3, dense q 24^3 (about 2 min on 8 cores)
mpirun -np 8 python main.py --nk 24 --nq 24
mpirun -np 8 python main.py --coarse-q    # q kept on the coarse 6^3 grid
```

Output in `output/`: `alpha2F.dat` (ω in meV, α²F) and `eliashberg.npz` (all
result arrays).

---

## Reference results (Pb, μ* = 0.1, σ = 0.05 eV)

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes |
|---|---|---|---|---|
| PAOFLOW, k 48³ / q 24³ (this example: EPW `wannierize = .false.`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| PAOFLOW, k 24³ / q 24³ (this example) | 1.066 | 5.19 meV | 4.61 K | 4.94 K |
| PAOFLOW, k 48³ / q 24³, on the Wannierized EPW run (`epw_wannier.in`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| EPW itself, k 48³ / q 24³ (Wannierized run) | 1.151 | 4.44 meV | 4.38 K | 4.76 K |
| EPW tutorial 04 | 1.158 | 4.40 meV | 4.37 K | 4.75 K |

The remaining difference from EPW follows the density of states at E_F of the
two band interpolations: 0.251 /eV/spin for the 18-orbital PAO bands against
0.235 for EPW's 4 Wannier functions. `--coarse-q` (λ ≈ 1.6) is not converged in
q and is shown only as the cheaper workflow.

---

## Notes

- **Commensurate grids:** the coarse q-grid must divide the coarse k-grid (here
  6³ / 6³), and `NK_DENSE` must be a multiple of `NQ_DENSE`.
- **Explicit k list:** EPW requires the full grid as `K_POINTS crystal` in
  [0, 1). `PAOFLOW.gen.epw_inputs.kpoints_card((6, 6, 6))` writes exactly the
  list in `epw/nscf.in`. PAOFLOW recovers the grid from the list and must be run
  with `pao_hamiltonian(expand_wedge=False)`, as in `main.py`.
- **Band exclusion without Wannier functions:** with `wannierize = .false.`
  EPW takes the excluded bands from `pb.ukk` (`write_placeholder_ukk(...,
  exclude_bands=..., nelec=...)`); `bands_skipped` in `epw.in` only documents
  the choice. `PAOFLOW.gen.epw_inputs.epw_input` writes the matching `epw.in`,
  and `paoflow-gen` (electron-phonon workflow) generates both.
- **Wannierized EPW runs** work too (`epw/epw_wannier.in`), as long as the
  outer disentanglement window contains every band (no `dis_win_min/max`).
