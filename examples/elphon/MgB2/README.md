# MgB₂ — two-gap superconductivity (anisotropic Migdal–Eliashberg) and the σ-band kink

MgB₂ is the two-gap example of the electron-phonon workflows. From one set of QE runs (`qe/`:
scf, `ph.x` on a 3³ q-grid, nscf on the 6³ grid) PAOFLOW computes the coarse coupling itself
(`coupling_source='dfpt'` in `system.py`) and interpolates it, the electrons and the phonons
to dense grids in the PAO basis. The workflow folders then give

- `superconductivity/`: the Eliashberg properties (α²F, λ, ω_log, Tc), the Fermi-surface pair
  coupling and the temperature-dependent gaps from the isotropic and the **anisotropic**
  Migdal–Eliashberg equations, as in the MgB₂ exercise of EPW tutorial 04
  (<https://docs.epw-code.org/tutorials/tutorial_04/index.html>);
- `arpes/`: the photoemission band map across the σ-band crossing with the anisotropic
  self-energy of every Fermi-surface state: the E₂g kink;
- `phonons/`: the phonon dispersion and DOS.

The workflow is the same as `../Pb`. MgB₂ adds three things:

- **A cell with three atoms (Mg, B, B):** every Wigner–Seitz sum of the interpolation is
  resolved by orbital/atom pair, which keeps the dense bands, phonons and couplings
  symmetric.
- **Phonons up to 100 meV:** this needs a larger Matsubara cutoff than Pb.
- **A two-gap superconductor:** the σ and π Fermi sheets couple differently (Liu, Mazin and
  Kortus 2001; Choi *et al.* 2002). The isotropic equations average them into one gap; the
  anisotropic equations resolve both.

The QE steps follow the tutorial: coarse grids of 6³ k and 3³ q, `ph.x` on the 6
irreducible q only. The EPW route (`epw.x` without Wannier functions, `epbwrite`) remains
available as `coupling_source='epw'`.

---

## Directory contents

| file | purpose |
|------|---------|
| `system.py` | the configuration (`ElphonSystem`: prefix `mgb2`, `coupling_source='dfpt'`, 6³ k / 3³ q, `masses_amu = [24.305, 10.811]` (Mg, B), `nelec = 16`, `pthr = 0.95`) and the defaults of every workflow (`SUPERCONDUCTIVITY` with `fs_coupling=True`, `ARPES`, `PHONONS`) |
| `qe/phonon/scf.in` | `pw.x` scf (8³ k, with symmetry) |
| `qe/phonon/ph.in` | `ph.x` DFPT on the 3³ q-grid (6 irreducible q, `fildvscf`) |
| `qe/nscf/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 24`); `python ../kmesh.py 6 6 6` switches to the irreducible wedge |
| `qe/nscf/epw.in`, `qe/nscf/write_ukk.py` | EPW route: `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, all 24 bands kept; `mgb2.ukk` and the empty `mgb2.bvec`/`mgb2.mmn` stubs |
| `qe/run_qe.sh`, `qe/kmesh.py` | the QE steps in order; the coarse-mesh editor |
| `phonons/phonons.py` | q2r + matdyn in PAOFLOW: dispersion, DOS, `mgb2.fc` |
| `superconductivity/eliashberg.py` | PAO electronic structure, Eliashberg on the coupling, Fermi-surface coupling (`--fs-coupling`, the default here) |
| `superconductivity/me.py` | isotropic Migdal–Eliashberg on the α²F: gap vs T, real axis, linearised-kernel Tc |
| `superconductivity/me_aniso.py` | anisotropic Migdal–Eliashberg on the Fermi-surface coupling: Δ_nk(T), gap distributions, quasiparticle DOS, Tc |
| `superconductivity/plot_me.py` | plots the `me.py` and `me_aniso.py` results (with the isotropic gap for comparison) |
| `arpes/arpes.py` | ARPES band map across the σ-band crossing on Γ-M with the anisotropic self-energy of every Fermi-surface state (from `fs_coupling.npz`) and the isotropic one: the E₂g kink |

**Pseudopotentials:** copy the ONCVPSP (PBE, norm-conserving) `Mg.upf` and `B.upf` of the
reference run into `qe/`. The inputs use `pseudo_dir = '../'`. `Mg.upf` includes the 2s 2p
semicore states, hence 16 valence electrons for the cell (10 + 2 × 3).

`paoflow-gen` (workflow `elphon`, metal, workflows `superconductivity, arpes`, Fermi-surface
coupling on) produces this layout from `qe/phonon/scf.in`; the drivers are the same files.

---

## Procedure

### 1. Quantum ESPRESSO

```bash
cd qe && NP=8 bash run_qe.sh        # or step by step:
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
cd ../nscf
mkdir -p mgb2.save                                # the nscf starts from the scf charge density
cp ../phonon/mgb2.save/{charge-density.dat,data-file-schema.xml} mgb2.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out
```

PAOFLOW computes the coarse coupling itself from the `ph.x` outputs (`qe/phonon/_ph0/`,
`qe/phonon/mgb2.dyn*`, or the `pp.py` `save/`) and the nscf wavefunctions of
`qe/nscf/mgb2.save`; the native vertex reproduces EPW's `.epb` matrix elements to the
rounding level (`docs/internals/dfpt-coupling-port.md`). With the native source the nscf may
also be run on the irreducible wedge (`K_POINTS automatic 6 6 6 0 0 0`, irreducible k only):
the drivers detect it and PAOFLOW rotates the stored Bloch functions to the full grid by
symmetry.

**EPW route** (`coupling_source='epw'`): `python3 <q-e>/EPW/bin/pp.py` in `qe/phonon`,
then in `qe/nscf` `python3 write_ukk.py` and `mpirun -np 20 epw.x -nk 20 -in epw.in` (one
process per pool). `epw.x` writes `mgb2.epb1 … mgb2.epbN`, which together with `mgb2.ukk` is
all PAOFLOW needs; the reference run used 20 pools, EPW took 5.6 min (QE/EPW 7.5, EPW 6.0).
See `../Pb/README.md` for the empty wannier90 stubs and the `patterns.1.xml` of `pp.py`.

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo-dir qe --out BASIS_PS
```

The drivers use the `standard` configuration: 40 PAO orbitals for the cell.

### 3. Eliashberg function and Fermi-surface coupling (`superconductivity/`)

```bash
cd superconductivity
mpirun -np 20 python eliashberg.py                   # dense k 24^3, dense q 24^3, fs_coupling (8 min on 20 cores)
mpirun -np 20 python eliashberg.py --nk 40 --nq 20   # the fine grids of EPW tutorial 04
mpirun -np 20 python eliashberg.py --coarse-q        # q kept on the coarse 3^3 grid
mpirun -np 20 python eliashberg.py --adaptive-smearing 0.7 --sigma-ev 0.001
                                                     # adaptive Fermi-surface delta widths a |v_nk| dk (floor 1 meV)
```

The dense 24³ q-grid folds to 793 irreducible q. Output goes to `superconductivity/output/`:
`alpha2F.dat` (ω in meV, α²F), `eliashberg.npz` (all result arrays) and, unless
`--no-fs-coupling` or `--coarse-q` is given, `fs_coupling.npz`.

`fs_coupling.npz` holds the state-resolved coupling λ(nk, mk′, ω) of the Fermi-surface
states: the bands within `fsthick_ev = 0.2` eV of E_F (EPW `fsthick`), folded to the
irreducible wedge of the dense k-grid (125 states at 24³); the coupling is accumulated in the
same irreducible-q loop, so nothing is rerun; its Fermi-surface average reproduces λ of
`eliashberg.npz` (to 10⁻⁸).

The drivers pass the PAO orbital centres (`pao_orbital_positions`) to the interpolation, so
every Wigner–Seitz sum is resolved by orbital/atom pair. With the single-site sums of a
one-atom cell, MgB₂'s dense bands differed by up to 83 meV between symmetry-equivalent k,
and λ came out as 0.608 instead of 0.579.

### 4. Superconducting gap (Migdal–Eliashberg)

```bash
python me.py                    # serial, a few seconds
python plot_me.py               # figures -> output/me/migdal_eliashberg.png, output/me_aniso/...
```

`me.py` uses (`SUPERCONDUCTIVITY` of `system.py`): `muc = 0.1`, `npade = 90`, `wscut = 0.1`
eV, `degaussq = 0.15` meV (0.5 meV for the linearised kernel) and the temperatures of the Pb
example; pass `--wscut 0.5 --temps 5 6 ... 30` for the tutorial's MgB₂ values (the phonons
here reach 100 meV). It writes `output/me/` in EPW's formats (`mgb2.imag_iso_<T>`,
`mgb2.pade_iso_<T>`, `mgb2.acon_iso_<T>`, `mgb2.qdos_iso_<T>`, `gap_vs_T.dat`,
`max_eigenvalue.dat`). The real-axis panels of `plot_me.py` extend to 150 meV
(`real_axis_max_mev`).

### 5. Anisotropic gaps (Fermi-surface-restricted Migdal–Eliashberg)

```bash
python me_aniso.py --linear-temps 5 60 12   # serial, seconds
python plot_me.py                           # -> output/me_aniso/migdal_eliashberg_aniso.png
```

`me_aniso.py` uses the anisotropic FSR settings of EPW tutorial 04 (`laniso`, `limag`,
`lpade`): `muc = 0.1`, `wscut = 0.5` eV (`wscut_aniso`), `npade = 90`, and temperatures
5–45 K (`me_aniso_temps`, `nstemp = 9`). It solves for Z_nk(iω_j) and Δ_nk(iω_j) for every
Fermi-surface state, continues each state to the real axis with Padé, and with
`--linear-temps` adds the linearised-kernel Tc. Use equal dense grids in `eliashberg.py`
(`--nk N --nq N`): with a coarser q-grid the equations split into independent sublattice
problems.

It writes `output/me_aniso/` in EPW's formats: `mgb2.lambda_FS`, `mgb2.lambda_k_pairs`
(λ_nk per state, and its distribution), `mgb2.imag_aniso_<T>` (Z_nk and Δ_nk on the Matsubara
axis), `mgb2.imag_aniso_gap0_<T>`, `mgb2.pade_aniso_gap0_<T>` (gap distributions),
`mgb2.imag_aniso_gap_FS_<T>` (Δ_nk(iω₀) per state, for Fermi-surface plots),
`mgb2.pade_aniso_<T>`, `mgb2.qdos_<T>`, `gap_vs_T_aniso.dat`, `max_eigenvalue_aniso.dat`,
`migdal_eliashberg_aniso.npz`.

### 6. ARPES and the σ-band kink (`arpes/`)

```bash
cd arpes
python arpes.py                        # needs ../superconductivity/output/fs_coupling.npz (seconds)
python arpes.py --direction 0.3333 0.3333 0 --temps 20     # Gamma-K instead of Gamma-M
```

From the Fermi-surface pair coupling Λ_ab(ω) of the anisotropic run, every irreducible state
gets its own Eliashberg function α²F_a(ω) = (ω/2) Σ_b Λ_ab(ω) and its Engelsberg–Schrieffer
self-energy Σ_a(ω, T) (`PAOFLOW.elphon.self_energy.eliashberg_self_energy`); the ARPES
simulation of a segment of Γ–M in the k_z = 0 plane across the σ-band crossing (band 6 at
0.128 of Γ–M; hν = 28 eV, s + p light, 20 K, 5 meV resolution; `ARPES` of `system.py`) takes
the self-energy of the nearest irreducible state of each band and k-point
(`--self-energy anisotropic isotropic`).

![MgB2 kink](figures/kink.png)

With the 40³/40³ coupling of this run (λ = 0.58 isotropic, N_F = 0.37 states/eV/spin): the
σ state at the crossing has λ_nk = 0.81 and a plateau −Im Σ = 87 meV above the E₂g energy,
the π states λ_nk ≈ 0.40 and 39 meV; the slope of Re Σ at E_F reproduces λ_nk to 1 %. In the
band map the σ band flattens below E_F and broadens abruptly at ≈ 70 meV, the E₂g kink of
the ARPES experiments (Uchiyama *et al.* 2002), much weaker with the isotropic self-energy.
The σ-band λ is below the ≈ 1.0–1.2 of converged calculations (Choi *et al.* 2002) because
of the coarse DFPT q-grid of this example. Outputs: `arpes/output/{sigma_anisotropic_<T>K.dat,
arpes_bare.npz, arpes_anisotropic_<T>K.npz, arpes_isotropic_<T>K.npz, arpes.png}`.

---

## Reference results (MgB₂, μ* = 0.1, σ = 0.05 eV)

Eliashberg properties, with pair-resolved Wigner–Seitz interpolation:

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes | N_F (states/eV/spin) |
|---|---|---|---|---|---|
| PAOFLOW, k 24³ / q 24³ (default) | 0.579 | 60.1 meV | 14.3 K | 14.7 K | 0.422 |
| PAOFLOW, k 32³ / q 32³ | 0.523 | 62.4 meV | 10.5 K | 10.7 K | 0.330 |
| PAOFLOW, k 40³ / q 20³ (tutorial grids) | 0.578 | 62.3 meV | 14.7 K | 15.1 K | 0.368 |
| EPW tutorial 04 (k 40³ / q 20³) | 0.573 | 61.7 meV | 14.2 K | 14.6 K | |

On the tutorial grids PAOFLOW reproduces EPW to 1% in λ and ω_log. With the single-site
interpolation the 24³ λ was 0.608.

**σ = 0.05 eV needs at least 40³ k.** N_F settles at 0.357 from 48³ k on. It is 18% high at
24³, 8% low at 32³ and 3% high at 40³, and λ follows it.

Migdal–Eliashberg at 5 K and Tc (`me.py`, `me_aniso.py`; Tc from the linearised kernel):

| grids | isotropic Δ(iω₀) | isotropic Tc | anisotropic Δ_nk(iω₀): π … σ | anisotropic Tc |
|---|---|---|---|---|
| k 24³ / q 24³ | 2.64 meV | 17.0 K | 1.9 … 4.9 meV | 24.0 K |
| k 32³ / q 32³ | 1.95 meV | 12.9 K | 1.1 … 4.9 meV | 25.9 K |
| k 40³ / q 20³ | 2.73 meV | 17.6 K | 1.4 … 7.7 meV | 40.8 K (25% of the Fermi surface; see below) |

- **The anisotropic solution has two gaps.** At 5 K the π states hold about 2 meV and the σ
  states about 5–8 meV. Δ_nk tracks λ_nk, which runs from 0.33 (π) to 0.99 (σ). The
  anisotropic Tc is about twice the isotropic one (measured: Tc = 39 K, gaps ≈ 2 and 7 meV;
  Choi *et al.* 2002).
- **Use equal dense grids (`--nk N --nq N`).** With a coarser q-grid, k+q reaches only a
  sublattice of the k-grid, and the anisotropic equations split into independent sublattice
  problems. At 40³ k / 20³ q, 75% of the Fermi surface loses its gap at about 30 K, while the
  rest keeps it to 40.8 K. `eliashberg.py` and `me_aniso.py` warn in this case.
- **Convergence.** The anisotropic gaps and Tc of this example are not converged in the
  k-grid; the σ sheets are small. Production values need at least `--nk 40 --nq 40` (about
  2.5 h on 20 cores) or a larger smearing. EPW tutorial 04 shows its anisotropic gaps only
  as figures (Figs. 3 and 6), without numerical values.

---

## Notes

- **Masses per species, expanded per atom:** `masses_amu = [24.305, 10.811]` follows
  `ATOMIC_SPECIES` (EPW's `amass`); `ElphonSystem.masses` expands it to one mass per atom
  (`[24.305, 10.811, 10.811]`). Passing the per-species list directly to the readers would
  make them expect 2 atoms instead of 3.
- **Commensurate grids:** the coarse q-grid (3³) must divide the coarse k-grid (6³), and the
  dense `nk` must be a multiple of `nq` (equal for the anisotropic equations).
- **Bands vs orbitals:** the nscf has `nbnd = 24`, fewer than the 40 PAO orbitals. The PAO
  Hamiltonian keeps the bands with projectability above `pthr = 0.95`. Raising `nbnd` is a
  convergence check for the bands near E_F.
- **Convergence:** `--nk 48 --nq 24` checks the electron grid; the coarse 3³ q grid of the
  tutorial bounds the dense-q interpolation.

## References

- E. R. Margine and F. Giustino, *Anisotropic Migdal-Eliashberg theory using Wannier functions*, Phys. Rev. B **87**, 024505 (2013) — the anisotropic equations and their EPW implementation (MgB₂ as the test case).
- H. J. Choi, D. Roundy, H. Sun, M. L. Cohen and S. G. Louie, *The origin of the anomalous superconducting properties of MgB₂*, Nature **418**, 758 (2002) — two-gap superconductivity of MgB₂ from the anisotropic Eliashberg equations.
- A. Y. Liu, I. I. Mazin and J. Kortus, *Beyond Eliashberg superconductivity in MgB₂: anharmonicity, two-phonon scattering, and multiple gaps*, Phys. Rev. Lett. **87**, 087005 (2001); Y. Kong, O. V. Dolgov, O. Jepsen and O. K. Andersen, *Electron-phonon interaction in the normal and superconducting states of MgB₂*, Phys. Rev. B **64**, 020501(R) (2001).
- H. Uchiyama, K. M. Shen, S. Lee, A. Damascelli, D. H. Lu, D. L. Feng, Z.-X. Shen and S. Tajima, *Electronic structure of MgB₂ from angle-resolved photoemission spectroscopy*, Phys. Rev. Lett. **88**, 157002 (2002) — the σ-band kink in ARPES.
- G. M. Eliashberg, Sov. Phys. JETP **11**, 696 (1960); P. B. Allen and B. Mitrović, Solid State Phys. **37**, 1 (1982); W. L. McMillan, Phys. Rev. **167**, 331 (1968); P. B. Allen and R. C. Dynes, Phys. Rev. B **12**, 905 (1975).
- S. Poncé, E. R. Margine, C. Verdi and F. Giustino, Comput. Phys. Commun. **209**, 116 (2016); H. Lee *et al.*, npj Comput. Mater. **9**, 156 (2023) — EPW and its tutorial 04.
- S. Engelsberg and J. R. Schrieffer, Phys. Rev. **131**, 993 (1963) — the self-energy of the kink; R. P. Day, B. Zwartsenberg, I. S. Elfimov and A. Damascelli, npj Quantum Mater. **4**, 54 (2019) — the photoemission matrix elements.
- J. R. Yates, X. Wang, D. Vanderbilt and I. Souza, Phys. Rev. B **75**, 195121 (2007) — adaptive smearing.
