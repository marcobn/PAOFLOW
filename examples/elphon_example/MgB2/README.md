# MgB₂ electron–phonon (Eliashberg) — PAOFLOW on EPW's coarse coupling

This example computes the Eliashberg properties of MgB₂ (α²F, λ, ω_log, Tc)
with PAOFLOW's PAO interpolation of EPW's coarse coupling. It then computes the
temperature-dependent gaps from the isotropic and the **anisotropic**
Migdal–Eliashberg equations. The workflow is the same as `../Pb`. MgB₂ adds
three things:

- **A cell with three atoms (Mg, B, B):** every Wigner–Seitz sum of the
  interpolation is resolved by orbital/atom pair, which keeps the dense bands,
  phonons and couplings symmetric.
- **Phonons up to 100 meV:** this needs a larger Matsubara cutoff than Pb.
- **A two-gap superconductor:** the σ and π Fermi sheets couple differently.
  The isotropic equations average them into one gap; the anisotropic equations
  resolve both.

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
| `main.py` | PAOFLOW analysis: PAO electronic structure, Eliashberg on EPW's coupling, Fermi-surface coupling |
| `me.py` | isotropic Migdal–Eliashberg on the α²F of `main.py`: gap vs T, real axis, linearised-kernel Tc |
| `plot_me.py` | plots the `me.py` results |
| `me_aniso.py` | anisotropic Migdal–Eliashberg on the Fermi-surface coupling of `main.py`: Δ_nk(T), gap distributions, quasiparticle DOS, Tc |
| `plot_me_aniso.py` | plots the `me_aniso.py` results (with the isotropic gap for comparison) |

**Pseudopotentials:** copy the ONCVPSP (PBE, norm-conserving) `Mg.upf` and
`B.upf` of the reference run into this directory. Both `phonon/` and `epw/` use
`pseudo_dir = '../'`. `Mg.upf` includes the 2s 2p semicore states, hence 16
valence electrons for the cell (10 + 2 × 3).

`paoflow-gen` (electron-phonon workflow, EPW source) produces this layout from
`scf.in`, as `main.elphon.py`, `me.elphon.py`, `me_aniso.elphon.py` and
`plot.elphon.py`.

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
mpirun -np 20 python main.py                   # dense k 24^3, dense q 24^3 (8 min on 20 cores)
mpirun -np 20 python main.py --nk 40 --nq 20   # the fine grids of EPW tutorial 04
mpirun -np 20 python main.py --coarse-q        # q kept on the coarse 3^3 grid
```

The dense 24³ q-grid folds to 793 irreducible q. Output goes to `output/`:
`alpha2F.dat` (ω in meV, α²F), `eliashberg.npz` (all result arrays) and, unless
`--iso-only` or `--coarse-q` is given, `fs_coupling.npz`.

`fs_coupling.npz` holds the state-resolved coupling λ(nk, mk′, ω) of the
Fermi-surface states:
- the states are the bands within `FSTHICK_EV = 0.2` eV of E_F (EPW `fsthick`),
  folded to the irreducible wedge of the dense k-grid (125 states at 24³);
- the coupling is accumulated in the same irreducible-q loop, so EPW is not run
  again;
- its Fermi-surface average reproduces λ of `eliashberg.npz` (to 10⁻⁸).

`main.py` passes the PAO orbital centres (`pao_orbital_positions`) to the
interpolation, so every Wigner–Seitz sum is resolved by orbital/atom pair. With
the single-site sums of a one-atom cell, MgB₂'s dense bands differed by up to
83 meV between symmetry-equivalent k, and λ came out as 0.608 instead of 0.579.

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

### 5. Anisotropic gaps (Fermi-surface-restricted Migdal–Eliashberg)

```bash
python me_aniso.py --linear-temps 5 60 12   # serial, seconds
python plot_me_aniso.py                     # figure -> output/me_aniso/migdal_eliashberg_aniso.png
```

`me_aniso.py` uses the anisotropic FSR settings of EPW tutorial 04 (`laniso`,
`limag`, `lpade`): `muc = 0.1`, `wscut = 0.5` eV, `npade = 90`, and
temperatures 5–45 K (`nstemp = 9`). It solves for Z_nk(iω_j) and Δ_nk(iω_j) for
every Fermi-surface state, continues each state to the real axis with Padé, and
with `--linear-temps` adds the linearised-kernel Tc.

It writes `output/me_aniso/` in EPW's formats:
- `mgb2.lambda_FS`, `mgb2.lambda_k_pairs`: λ_nk per state, and its distribution
- `mgb2.imag_aniso_<T>`: Z_nk and Δ_nk on the Matsubara axis
- `mgb2.imag_aniso_gap0_<T>`, `mgb2.pade_aniso_gap0_<T>`: gap distributions
- `mgb2.imag_aniso_gap_FS_<T>`: Δ_nk(iω₀) per state, for Fermi-surface plots
- `mgb2.pade_aniso_<T>`, `mgb2.qdos_<T>`
- `gap_vs_T_aniso.dat`, `max_eigenvalue_aniso.dat`, `migdal_eliashberg_aniso.npz`

---

## Reference results (MgB₂, μ* = 0.1, σ = 0.05 eV)

Eliashberg properties, with pair-resolved Wigner–Seitz interpolation:

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes | N_F (states/eV/spin) |
|---|---|---|---|---|---|
| PAOFLOW, k 24³ / q 24³ (default) | 0.579 | 60.1 meV | 14.3 K | 14.7 K | 0.422 |
| PAOFLOW, k 32³ / q 32³ | 0.523 | 62.4 meV | 10.5 K | 10.7 K | 0.330 |
| PAOFLOW, k 40³ / q 20³ (tutorial grids) | 0.578 | 62.3 meV | 14.7 K | 15.1 K | 0.368 |
| EPW tutorial 04 (k 40³ / q 20³) | 0.573 | 61.7 meV | 14.2 K | 14.6 K | |

On the tutorial grids PAOFLOW reproduces EPW to 1% in λ and ω_log. With the
single-site interpolation the 24³ λ was 0.608.

**σ = 0.05 eV needs at least 40³ k.** N_F settles at 0.357 from 48³ k on.
It is 18% high at 24³, 8% low at 32³ and 3% high at 40³, and λ follows it.

Migdal–Eliashberg at 5 K and Tc (`me.py`, `me_aniso.py`; Tc from the
linearised kernel):

| grids | isotropic Δ(iω₀) | isotropic Tc | anisotropic Δ_nk(iω₀): π … σ | anisotropic Tc |
|---|---|---|---|---|
| k 24³ / q 24³ | 2.64 meV | 17.0 K | 1.9 … 4.9 meV | 24.0 K |
| k 32³ / q 32³ | 1.95 meV | 12.9 K | 1.1 … 4.9 meV | 25.9 K |
| k 40³ / q 20³ | 2.73 meV | 17.6 K | 1.4 … 7.7 meV | 40.8 K (25% of the Fermi surface; see below) |

- **The anisotropic solution has two gaps.** At 5 K the π states hold about
  2 meV and the σ states about 5–8 meV. Δ_nk tracks λ_nk, which runs from 0.33
  (π) to 0.99 (σ). The anisotropic Tc is about twice the isotropic one.
- **Use equal dense grids (`--nk N --nq N`).** With a coarser q-grid, k+q reaches
  only a sublattice of the k-grid, and the anisotropic equations split into
  independent sublattice problems. At 40³ k / 20³ q, 75% of the Fermi surface
  loses its gap at about 30 K, while the rest keeps it to 40.8 K.
  `main.py` and `me_aniso.py` warn in this case.
- **Convergence.** The anisotropic gaps and Tc of this example are not
  converged in the k-grid; the σ sheets are small. Production values need at
  least `--nk 40 --nq 40` (about 2.5 h on 20 cores) or a larger smearing.
  EPW tutorial 04 shows its anisotropic gaps only as figures (Figs. 3 and 6),
  without numerical values.

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
  (6³), and `NK_DENSE` must be a multiple of `NQ_DENSE` (equal for the
  anisotropic equations).
- **Bands vs orbitals:** the nscf has `nbnd = 24`, fewer than the 40 PAO
  orbitals. The PAO Hamiltonian keeps the bands with projectability above
  `pthr = 0.95`. Raising `nbnd` (and the bands kept by EPW) is a convergence
  check for the bands near E_F; EPW's cost grows with the square of the kept
  bands.
- **Convergence:** `--nk 48 --nq 24` checks the electron grid; the coarse 3³ q
  grid of the tutorial bounds the dense-q interpolation.
