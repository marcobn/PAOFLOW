# MgB₂ electron–phonon (Eliashberg) — PAOFLOW on EPW's coarse coupling

This example computes the isotropic Eliashberg properties of MgB₂ (α²F, λ,
ω_log, Tc) with PAOFLOW's PAO interpolation of EPW's coarse coupling, then the
temperature-dependent gap from the isotropic Migdal–Eliashberg equations. The
workflow is the same as `../Pb`. MgB₂ adds three things:

- **A cell with repeated species:** two species, three atoms (Mg, B, B).
- **Phonons up to 100 meV:** this needs a larger Matsubara cutoff than Pb.
- **A two-gap superconductor:** the isotropic equations average the σ and π gaps,
  which is why this case prepares the anisotropic pipeline.

The QE steps follow the MgB₂ exercise of EPW tutorial 04
(<https://docs.epw-code.org/tutorials/tutorial_04/index.html>):
- coarse grids of 6³ k and 3³ q
- ph.x on the 6 irreducible q only
- EPW run **without Wannier functions** (`wannierize = .false.`): it computes the
  coarse matrix elements, and PAOFLOW does all the interpolation

---

## Directory contents

| file | purpose |
|------|---------|
| `phonon/scf.in` | `pw.x` scf (8³ k, with symmetry) |
| `phonon/ph.in` | `ph.x` DFPT on the 3³ q-grid (6 irreducible q, `fildvscf`) |
| `epw/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 24`) |
| `epw/epw.in` | `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, all 24 bands kept |
| `epw/write_ukk.py` | writes `mgb2.ukk` (band bookkeeping EPW reads when `wannierize = .false.`) and empty `mgb2.bvec`/`mgb2.mmn` stubs |
| `main.py` | PAOFLOW analysis: PAO electronic structure + Eliashberg on EPW's coupling |
| `me.py` | isotropic Migdal–Eliashberg on the α²F of `main.py`: gap vs T, real axis, linearised-kernel Tc |
| `plot_me.py` | plots the `me.py` results |

**Pseudopotentials:** copy the ONCVPSP (PBE, norm-conserving) `Mg.upf` and
`B.upf` of the reference run into this directory. Both `phonon/` and `epw/` use
`pseudo_dir = '../'`. `Mg.upf` includes the 2s 2p semicore states, hence 16
valence electrons for the cell (10 + 2 × 3).

`paoflow-gen` (electron-phonon workflow, EPW source) produces this layout from
`scf.in`, as `main.elphon.py`, `me.elphon.py` and `plot.elphon.py`.

---

## Procedure

### 1. Quantum ESPRESSO and EPW

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
python3 /path/to/q-e/EPW/bin/pp.py           # prefix 'mgb2' -> collects dvscf, patterns, dyn into save/

cd ../epw
mkdir -p mgb2.save                                # the nscf starts from the scf charge density
cp ../phonon/mgb2.save/{charge-density.dat,data-file-schema.xml} mgb2.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out
python3 write_ukk.py                              # mgb2.ukk for wannierize = .false.
mpirun -np 20 epw.x -nk 20 -in epw.in > epw.out  # EPW: one process per pool (-np = -nk)
```

- **Outputs PAOFLOW reads:** `epw.x` writes `mgb2.epb1 … mgb2.epbN`, one file per
  pool, which together with `mgb2.ukk` is all PAOFLOW needs. The reference run
  used 20 pools; EPW took 5.6 min of wall time (QE/EPW 7.5, EPW 6.0).
- **Practical notes:** see `../Pb/README.md` for the empty wannier90 stubs, the
  `patterns.1.xml` written by `pp.py`, and the choice of pools.

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo-dir . --out BASIS_PS
```

`main.py` uses the `standard` configuration: 40 PAO orbitals for the cell.

### 3. PAOFLOW analysis

```bash
mpirun -np 8 python main.py               # dense k 24^3, dense q 24^3 (about 11 min on 8 cores)
mpirun -np 8 python main.py --coarse-q    # q kept on the coarse 3^3 grid
```

The dense 24³ q-grid folds to 793 irreducible q. Output goes to `output/`:
`alpha2F.dat` (ω in meV, α²F) and `eliashberg.npz` (all result arrays).

### 4. Superconducting gap (Migdal–Eliashberg)

```bash
python me.py                    # serial, a few seconds
python plot_me.py               # figure -> output/me/migdal_eliashberg.png
```

`me.py` uses:
- `muc = 0.1`, `npade = 90`
- `wscut = 0.5` eV, the tutorial's MgB₂ value (Pb uses 0.1 eV; the phonons here
  reach 100 meV)
- temperatures 5–30 K in 1 K steps
- `degaussq = 0.15` meV (0.5 meV for the linearised kernel)

It writes `output/me/` in EPW's formats (`mgb2.imag_iso_<T>`,
`mgb2.pade_iso_<T>`, `mgb2.acon_iso_<T>`, `mgb2.qdos_iso_<T>`, `gap_vs_T.dat`,
`max_eigenvalue.dat`). The real-axis panels of `plot_me.py` extend to 150 meV.

---

## Reference results (MgB₂, μ* = 0.1, σ = 0.05 eV)

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes |
|---|---|---|---|---|
| PAOFLOW, k 24³ / q 24³ (this example) | 0.608 | 61.0 meV | 16.8 K | 17.3 K |
| EPW tutorial 04 (isotropic, k 40³ / q 20³) | 0.573 | | 14.2 K | 14.6 K |

The fine grids differ (PAOFLOW 24³ k, EPW 40³ k), so the table is not a
convergence comparison.

Migdal–Eliashberg (`me.py`) on the PAOFLOW α²F:

| T | Δ(iω₀) | Z(iω₀) | gap edge, Padé / acon |
|---|---|---|---|
| 5 K (185 Matsubara points, as in the tutorial) | 3.106 meV | 1.591 | 3.115 / 3.115 meV |
| 15 K | 2.415 meV | 1.593 | 2.427 / 2.429 meV |
| 19 K | 1.154 meV | 1.596 | 1.158 / 1.160 meV |

- **Tc:** 19.93 K from the linearised kernel, 20.02 K from Δ²(T) → 0.
- **Isotropic vs anisotropic:** the isotropic gap (about 3.1 meV) falls between
  the two gaps of the anisotropic solution, about 7–8 meV for σ and 2–3 meV for π
  in EPW tutorial 04. The anisotropic Tc there is about 35 K (39–40 K with the
  full-bandwidth solver). The isotropic Tc of about 20 K is the expected
  underestimate.

---

## Notes

- **Masses per species, expanded per atom:** `MASSES_AMU = [24.305, 10.811]`
  follows `ATOMIC_SPECIES` (EPW's `amass`).
  - `main.py` expands it to one mass per atom with
    `atom_masses(MASSES_AMU, nscf['species'], nscf['atom_names'])`, which gives
    `[24.305, 10.811, 10.811]`.
  - Passing the per-species list directly makes `read_epw_epb` fail with a
    record-size mismatch, because it then expects 2 atoms instead of 3.
- **Commensurate grids:** the coarse q-grid (3³) must divide the coarse k-grid
  (6³), and `NK_DENSE` must be a multiple of `NQ_DENSE`.
- **Bands vs orbitals:** the nscf has `nbnd = 24`, fewer than the 40 PAO
  orbitals. The PAO Hamiltonian keeps the bands with projectability above
  `pthr = 0.95`. Raising `nbnd` (and the bands kept by EPW) is a convergence
  check for the bands near E_F; EPW's cost grows with the square of the kept
  bands.
- **Convergence:** `--nk 48 --nq 24` checks the electron grid; the coarse 3³ q
  grid of the tutorial bounds the dense-q interpolation.
