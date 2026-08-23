# Pb electron–phonon (Eliashberg) — PAO dense-q route

This directory computes the isotropic Eliashberg properties of fcc lead
(α²F, λ, ω_log, Tc) with PAOFLOW's **PAO dense-q** method: Quantum ESPRESSO's
coarse-grid DFPT electron–phonon coupling is rotated into the pseudo-atomic-orbital
(PAO) gauge and Wigner–Seitz interpolated — **both** the electron k-grid and the
phonon q-grid — onto dense meshes. The driver is `main.elphon.py`.

This run uses the **AHC** coupling source (`SOURCE='ahc'`, norm-conserving
pseudopotential), a 9³ SCF/coupling k-grid, and a 6³ DFPT q-grid.

---

## Directory contents

### Quantum ESPRESSO inputs (you provide / run first)
| file | purpose |
|------|---------|
| `pb.scf.in`  | `pw.x` self-consistent ground state |
| `pb.nscf.in` | `pw.x` non-self-consistent run on the **full** k-grid (`nosym=.true.`, `noinv=.true.`, `nbnd > nawf`) |
| `pb.ph.in`   | `ph.x` DFPT phonons on the 6³ q-grid (writes `lead.dyn*`, `dvscf`) |
| `pb.elph.in` | `ph.x` with `electron_phonon='ahc'` → AHC coupling dumps in `ahc_dir/` |
| `pb_s.UPF`   | norm-conserving pseudopotential |
| `submit_qe.sbatch` | example batch script for the QE steps |

`lead.ph.in` / `lead.ahc.in` are equivalent templates auto-written by
`main.elphon.py inputs` (see below).

### PAOFLOW inputs
| file / dir | purpose |
|------------|---------|
| `main.elphon.py` | the PAO dense-q driver (edit the CONFIG block at the top) |
| `plot.elphon.py` | plots α²F(ω) and cumulative λ(ω) from the output |
| `BASIS_PS/`      | PAOFLOW pseudo-atomic-orbital basis for Pb |

### Produced by the run (do not edit)
| file / dir | produced by |
|------------|-------------|
| `lead.save/` | `pw.x` nscf (contains `data-file-schema.xml`, wavefunctions, **symmetries**) |
| `lead.dyn0 … lead.dyn216` | `ph.x` DFPT (dynamical matrices, full 6³ = 216 q) |
| `ahc_dir/` | `ph.x` AHC (`ahc_gkk_iq*.bin` coupling) |
| `output/` | PAOFLOW results: `alpha2F.dat`, `eliashberg.npz` |

---

## Key parameters (`main.elphon.py` CONFIG)

```
SOURCE   = 'ahc'         # norm-conserving AHC coupling
KGRID    = (9, 9, 9)     # nscf / coupling k-grid  (== pw.x K_POINTS)
QGRID    = (6, 6, 6)     # DFPT phonon q-grid      (== ph.x nq1/nq2/nq3)
NBND     = 22            # bands in nscf / ahc_nbnd (> nawf)
MASSES_AMU = [207.2]     # atomic masses (amu)
NELEC    = 14            # valence electrons
NK_DENSE = 42            # dense electron k-grid
NQ_DENSE = 42            # dense phonon q-grid
SIGMA_RY = 0.02          # Fermi-surface smearing (Ry)
MU_STAR  = 0.10          # Coulomb pseudopotential (Tc)
PTHR     = 0.95          # projectability threshold
```

**Grid consistency (must hold):**
- nscf must be the **full** Γ-centred k-grid with `nosym=.true., noinv=.true.` and
  `nbnd > nawf` — the PAO vertex needs every `k+q` to exist in the k-list.
- The `ph.x` q-grid, the `lead.dyn*` files, and `QGRID` must all be the **same**
  coarse grid (here 6³ → 216 q-points, one `lead.dyn<iq>` per q).
- **`NK_DENSE` must be an integer multiple of `NQ_DENSE`** (the k+q index shift on
  the dense grid). Equal values (42 = 42) satisfy this.
- Crystal symmetry is read automatically from `lead.save` and used to fold the
  dense q-grid to its irreducible wedge (no action needed).

---

## Step-by-step procedure

### 1. Quantum ESPRESSO (coarse grids)
Run in this directory (or via `submit_qe.sbatch`), in order:

```bash
pw.x  < pb.scf.in   > pb.scf.out      # 1. SCF ground state
pw.x  < pb.nscf.in  > pb.nscf.out     # 2. NSCF, full k-grid (nosym, noinv, nbnd>nawf) -> lead.save/
ph.x  < pb.ph.in    > pb.ph.out       # 3. DFPT phonons on 6^3 q-grid -> lead.dyn*, dvscf
ph.x  < pb.elph.in  > pb.elph.out     # 4. AHC coupling (electron_phonon='ahc') -> ahc_dir/
```

Steps 3 and 4 must use the **same** `outdir`, `fildyn`, and `fildvscf`.

> Tip: `python main.elphon.py inputs` writes ready-to-run `lead.ph.in` and
> `lead.ahc.in` templates for steps 3–4 if you prefer to regenerate them.

### 2. PAOFLOW analysis (dense interpolation)
Once `lead.save/`, `ahc_dir/`, and all `lead.dyn*` are present:

```bash
# edit the CONFIG block in main.elphon.py if needed, then:
python main.elphon.py analyse
```

On an HPC node, parallelise over the (symmetry-reduced) q-points with MPI +
BLAS threads, e.g. on a 128-core node:

```bash
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=$OMP_NUM_THREADS   # or OPENBLAS_NUM_THREADS
ibrun -np 64 python main.elphon.py analyse
```

Near the top of the output you should see a line such as
`dense-q symmetry: N irreducible / M full q` confirming the symmetry reduction is
active. Keep `-np` ≤ the number of irreducible q-points.

### 3. Plot
```bash
python plot.elphon.py        # reads output/eliashberg.npz
```

---

## Output

- `output/alpha2F.dat` — two columns `ω(meV)  α²F(ω)`; the header lists
  λ, ω_log, Tc (McMillan and Allen–Dynes) and μ*.
- `output/eliashberg.npz` — full result arrays (α²F, λ, per-mode λ_qν, ω_qν, …).

**Reference values for Pb** (converged dense grids): λ ≈ 1.2–1.5, ω_log ≈ 60 K,
Tc ≈ 6–7 K (μ* = 0.1).

---

## Notes & troubleshooting

- **Coarse-grid convergence:** a 3³ DFPT q-grid is too coarse and inflates λ near Γ;
  use 6³ or finer. λ should converge from above as `NQ_DENSE` increases.
- **`dyn files cover only X/Y`:** `QGRID` in `main.elphon.py` does not match the
  actual `lead.dyn*` grid — set it to the DFPT `nq1/nq2/nq3`.
- **Memory:** the large real-space vertex `g(R_e,R_p)` is held in one shared copy
  per node, so `-np` can be raised without multiplying its footprint.
- **Pseudopotentials:** `SOURCE='ahc'` requires norm-conserving pseudopotentials.
  For ultrasoft/PAW, use the patched-QE `el_ph_mat` dump (`SOURCE='elphmat'`).
