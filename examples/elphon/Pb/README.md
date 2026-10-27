# Pb — Eliashberg superconductivity, metal transport and the ARPES kink

Lead is the metal example of the electron-phonon workflows. From one set of QE runs (`qe/`:
scf, `ph.x` on the 6³ q-grid, nscf on the 6³ grid) PAOFLOW computes the coarse coupling
itself (`coupling_source='dfpt'` in `system.py`) and interpolates it, the electrons and the
phonons to dense grids in the PAO basis. The workflow folders then give

- `superconductivity/`: the isotropic Eliashberg properties (α²F, λ, ω_log, Tc) and the
  temperature-dependent gap from the isotropic Migdal–Eliashberg equations, as in EPW
  tutorial 04 (<https://docs.epw-code.org/tutorials/tutorial_04/index.html>);
- `transport/`: the resistivity ρ(T), the extended-Drude infrared optics and the thermal
  emissivity of the metal from the transport Eliashberg function α²F_tr;
- `optics/`: the interband dielectric function on the extended PAO basis (optional input
  of the emissivity);
- `arpes/`: the photoemission band map across the Fermi crossing with the electron-phonon
  self-energy: the kink;
- `phonons/`: the phonon dispersion and DOS (q2r + matdyn in PAOFLOW).

The QE steps are those of the EPW tutorial (6³ coarse k and q grids, `ph.x` on the 16
irreducible q only). The coupling source validated against EPW was EPW itself, run
**without Wannier functions** (`wannierize = .false.`, `epbwrite`): that route remains
available (`coupling_source='epw'`) and the native vertex reproduces its `.epb` matrix
elements to the rounding level.

---

## Directory contents

| file | purpose |
|------|---------|
| `system.py` | the configuration (`ElphonSystem`: prefix `pb`, `coupling_source='dfpt'`, 6³ k / 6³ q, `masses_amu = [207.2]`, `nelec = 14` (5d10 6s2 6p2), `pthr = 0.95`) and the defaults of every workflow (`SUPERCONDUCTIVITY`, `OPTICS`, `TRANSPORT`, `ARPES`, `PHONONS`) |
| `qe/phonon/scf.in` | `pw.x` scf (8³ k, with symmetry) |
| `qe/phonon/ph.in` | `ph.x` DFPT on the 6³ q-grid (irreducible q, `fildvscf`) |
| `qe/nscf/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 16`); `python ../kmesh.py 6 6 6` switches to the irreducible wedge (`K_POINTS automatic`), which the native coupling unfolds by symmetry |
| `qe/nscf/epw.in`, `qe/nscf/write_ukk.py` | EPW route: `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, `exclude_bands = 1:5`; `pb.ukk` (band bookkeeping EPW reads when `wannierize = .false.`) and empty `pb.bvec`/`pb.mmn` stubs |
| `qe/nscf/epw_wannier.in` | the tutorial's Wannierized EPW input (alternative; same `.epb` content) |
| `qe/optics/nscf.in` | optional: nscf with `nbnd = 40` for the extended-basis interband dielectric function |
| `qe/run_qe.sh`, `qe/kmesh.py` | the QE steps in order; the coarse-mesh editor |
| `phonons/phonons.py` | q2r + matdyn in PAOFLOW: dispersion, DOS, `pb.fc` |
| `superconductivity/eliashberg.py` | PAO electronic structure + Eliashberg on the coupling (`--transport` adds α²F_tr, `--fs-coupling` the Fermi-surface pair coupling) |
| `superconductivity/me.py` | isotropic Migdal–Eliashberg on the α²F: gap vs T, real axis, linearised-kernel Tc |
| `superconductivity/me_aniso.py` | anisotropic Migdal–Eliashberg on the Fermi-surface coupling (`--fs-coupling` run with `--nk N --nq N`) |
| `superconductivity/plot_me.py` | plots the `me.py` (and `me_aniso.py`) results |
| `transport/resistivity.py` | metal resistivity ρ(T), extended-Drude optics and thermal emissivity from α²F_tr |
| `transport/plot_resistivity.py` | plots α²F vs α²F_tr, ρ(T) (optional measured overlay) and the emissivity |
| `optics/dielectric.py` | interband dielectric function on the extended basis (`dielectric_tensor(intraband=False)`, `--interband-only` is the default here) |
| `arpes/arpes.py` | ARPES band map along Γ-X across k_F with the isotropic self-energy from α²F (the kink), optionally the band-resolved Fan–Migdal one (`--self-energy isotropic fan-migdal`) |

**Pseudopotential:** copy `Pb.upf` (ONCVPSP, from the EPW tutorial material) into `qe/`.
The inputs of `qe/phonon/`, `qe/nscf/` and `qe/optics/` use `pseudo_dir = '../'`.

`paoflow-gen` (workflow `elphon`, metal, workflows `superconductivity, transport, arpes`)
produces this layout from `qe/phonon/scf.in`; the drivers are the same files.

---

## Procedure

### 1. Quantum ESPRESSO

```bash
cd qe && NP=8 bash run_qe.sh        # or step by step:
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
cd ../nscf
mkdir -p pb.save                                 # the nscf starts from the scf charge density
cp ../phonon/pb.save/{charge-density.dat,data-file-schema.xml} pb.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out
```

PAOFLOW computes the coarse coupling itself from the `ph.x` outputs (`qe/phonon/_ph0/`,
`qe/phonon/pb.dyn*`, or the `pp.py` `save/`) and the nscf wavefunctions of
`qe/nscf/pb.save`; the native vertex reproduces EPW's `.epb` matrix elements to the
rounding level (`docs/internals/dfpt-coupling-port.md`). With the native source the nscf may
also be run on the irreducible wedge (`python ../kmesh.py 6 6 6`: `K_POINTS automatic
6 6 6 0 0 0`, 16 irreducible k instead of 216): the drivers detect it and PAOFLOW rotates
the stored Bloch functions to the full grid by symmetry.

**EPW route** (`coupling_source='epw'` in `system.py`; required for ultrasoft or PAW
pseudopotentials):

```bash
cd qe/phonon && python3 /path/to/q-e/EPW/bin/pp.py   # prefix 'pb' -> collects dvscf, patterns, dyn into save/
cd ../nscf                                           # the explicit full k list (python ../kmesh.py 6 6 6 --explicit)
python3 write_ukk.py                                 # pb.ukk for wannierize = .false.
mpirun -np 8 epw.x -nk 8 -in epw.in > epw.out        # EPW: one process per pool (-np = -nk)
```

`epw.x` writes `pb.epb1 … pb.epbN` (the coarse Bloch coupling, one file per pool), which
together with `pb.ukk` is all PAOFLOW reads. After printing "The .epb files have been
correctly written", EPW enters its Wannier stage, which reads wannier90's `pb.bvec`/`pb.mmn`;
`write_ukk.py` writes empty stubs of both so that stage finishes cleanly.
`../phonon/save/pb.phsave/patterns.1.xml` must exist (created by `pp.py`). Tested with
QE/EPW 7.5 (EPW 6.0).

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo qe/Pb.upf --out BASIS_PS
```

The drivers use the `standard` configuration: 5d 6s 6p plus 6d 7s 7p, 18 orbitals.

### 3. Eliashberg function (`superconductivity/`)

```bash
cd superconductivity
mpirun -np 8 python eliashberg.py               # dense k 48^3, dense q 24^3 (about 2 min on 8 cores)
mpirun -np 8 python eliashberg.py --nk 24 --nq 24
mpirun -np 8 python eliashberg.py --coarse-q    # q kept on the coarse 6^3 grid
mpirun -np 8 python eliashberg.py --adaptive-smearing 0.7 --sigma-ev 0.001
                                                # adaptive Fermi-surface delta widths a |v_nk| dk (floor 1 meV)
```

Output in `superconductivity/output/`: `alpha2F.dat` (ω in meV, α²F) and `eliashberg.npz`
(all result arrays). The coarse vertex g(R_e, R_p) is cached in `cache/` and reused by
every workflow.

### 4. Superconducting gap (Migdal–Eliashberg)

```bash
python me.py                            # serial, a few seconds
python plot_me.py                       # figure -> output/me/migdal_eliashberg.png
python me.py --a2f ../qe/nscf/pb.a2f    # optional: the same solver on EPW's own alpha^2F
```

`me.py` uses the parameters of the tutorial's Eliashberg steps (`SUPERCONDUCTIVITY` of
`system.py`): `muc = 0.1`, `wscut = 0.1` eV, `npade = 90`, the 23 temperatures 0.3–6.0 K,
`degaussq = 0.15` meV (0.5 meV and 25 temperatures 0.25–6.25 K for the linearised kernel).
It writes `output/me/` in EPW's formats: `pb.imag_iso_<T>` (gap and Z on the imaginary
axis), `pb.pade_iso_<T>` and `pb.acon_iso_<T>` (real axis, from Padé and from the analytic
continuation), `pb.qdos_iso_<T>` (quasiparticle DOS), `gap_vs_T.dat` (Δ(T) from the lowest
Matsubara frequency and the two real-axis gap edges) and `max_eigenvalue.dat` (largest
eigenvalue of the linearised kernel, which is 1 at Tc).

### 5. Resistivity, infrared optics and thermal emissivity (`transport/`)

```bash
cd superconductivity && mpirun -np 8 python eliashberg.py --transport   # adds alpha^2F_tr, lambda_tr, plasma frequency (2 min)
cd ../transport
python resistivity.py                               # rho(T), extended-Drude eps(omega, T), emissivity
python plot_resistivity.py [--experiment rho.dat]   # -> output/{metal_transport,emissivity}.png
# optional interband part on the extended basis (separate nscf with nbnd = 40):
mkdir -p ../qe/optics/pb.save && cp ../qe/phonon/pb.save/{charge-density.dat,data-file-schema.xml} ../qe/optics/pb.save/
(cd ../qe/optics && mpirun -np 4 pw.x -nk 4 -in nscf.in > nscf.out)
python ../optics/dielectric.py                      # -> qe/optics/output (intraband=False)
python resistivity.py --interband-dielectric ../qe/optics/output
```

**What `--transport` adds.** It weights the e-ph double-delta sum by the velocity change,
giving the transport Eliashberg function α²F_tr and λ_tr (`output/alpha2F_tr.dat`; Allen
1978). It also gives the Fermi-surface plasma frequency.

**What `resistivity.py` does.** It combines the e-ph channel (Allen's DC rate, and the
frequency-dependent rate with mass enhancement of the extended Drude model, Allen 1971,
for the optics) with an optional residual resistivity (`--rho0`) and Fermi-liquid e-e
scattering (`--b-ee`), by Matthiessen's rule, and writes `transport/output/`:
`resistivity.dat`, `eps_<T>K.dat`, `emish_<T>K.dat`, `emist.dat` and `emissivity.npz`. The
same channels also drive PAOFLOW's Boltzmann transport:
`pf.transport(scattering_channels=[TransportSpectralChannel.from_npz(...)])`.

**Pb results** (48³/24³, σ = 0.05 eV, band group velocities):
- λ = 1.20, λ_tr = 0.94, ħω_p = 8.34 eV.
- Resistivity in µΩ cm, linear above about 100 K:

  | T (K) | 20 | 50 | 100 | 200 | 300 | 400 | 600 |
  |---|---|---|---|---|---|---|---|
  | ρ | 0.52 | 2.4 | 5.2 | 10.8 | 16.2 | 21.7 | 32.6 |

  Experiment gives about 21 µΩ cm at 300 K.
- An earlier Boltzmann run on PAOFLOW's 48³ grid with the same channel, at the same
  chemical potential, gave 19.3 µΩ cm at 300 K. That run used the former non-local velocity
  term (removed in October 2026), which also altered the band velocities.
- Total hemispherical emissivity of a clean surface: 0.012 (100 K), 0.037 (300 K), 0.071
  (600 K).
- The interband part changes the totals by less than 2%. Its main feature is the
  parallel-band absorption near 2 eV.
- At room temperature the extended and simple Drude models nearly coincide, because k_BT
  is already comparable to the phonon energies. The Holstein enhancement of the infrared
  absorption appears for T well below the Debye temperature.

**Convergence.**
- **k-grid.** The Fermi-surface quantities need at least 48³ k at σ = 0.05 eV: at 24³, ħω_p
  is 6.9 eV and λ_tr is 1.23.
- **q-grid.** Refining q from 24³ to 48³ (k = 48³) changes λ from 1.20 to 1.14, λ_tr from
  0.94 to 0.87 and ρ(300 K) by about 6%. ω_p does not depend on q.
- **Equal grids.** Equal k- and q-grids are not required for these Fermi-surface averages
  (the anisotropic Migdal–Eliashberg equations do need them), but the q-grid must still be
  converged.
- **Smearing.** `--adaptive-smearing A` replaces the fixed Fermi-surface width by
  `A |v_nk| dk` per state (floored at `--sigma-ev`; see "Adaptive smearing of the deltas" in
  `docs/internals/Electron-Phonon-Coupling.md`). With the native DFPT coupling and q = 24³
  (8 cores, October 2026):

  | k-grid | σ = 0.05 eV | σ = 0.02 eV | adaptive a = 0.7 | a = 0.35 | a = 0.2 |
  |---|---|---|---|---|---|
  | 24³ | λ = 1.018 | 1.373 | 1.068 | | |
  | 48³ | 1.176 | | 1.069 | 1.054 | 1.100 |
  | 72³ | 1.029 | | 1.069 | | |

  The fixed-width λ scatters by ±8 % between the grids (its 50 meV is far below the level
  spacing |v_F| dk, 0.3–1 eV at these grids), while the adaptive value is flat from 24³ to
  72³ and moves by 2 % between a = 0.2 and 0.7. The widths of the states near E_F are 1 meV
  to 0.25 eV at 72³ (median 8 meV with a = 0.7). Tc follows λ: 4.5 K (Allen–Dynes, μ* = 0.1)
  with the adaptive widths against 4.2–5.2 K for the fixed ones.

### 6. ARPES and the phonon kink (`arpes/`)

```bash
cd arpes
mpirun -np 8 python arpes.py                                          # isotropic Sigma from ../superconductivity/output/eliashberg.npz (seconds)
mpirun -np 8 python arpes.py --self-energy isotropic fan-migdal --nq 18 --adaptive-smearing 0.7   # + the state-resolved form (13 min on 8 cores)
```

`arpes.py` (`PAOFLOW.arpes`, see `docs/internals/arpes.md`) simulates the photoemission band
map of a short segment of Γ–X across the Fermi crossing of band 6 (at 0.575 of Γ–X, He I
light with s + p polarisation, 10 K, 2 meV energy resolution; `ARPES` of `system.py`) with
the photoemission matrix elements of the PAO orbitals, bare and dressed with the
electron-phonon self-energy:

![Pb kink](figures/kink.png)

- **Isotropic self-energy** from α²F of `eliashberg.py` (λ = 1.03, the 72³/24³ run): the
  Engelsberg–Schrieffer form at 10 K has Im Σ rising to a plateau of 8.7 meV beyond the
  phonon energies (≈ 11 meV) and the dispersive Re Σ with extrema at ±7 meV; its slope at
  E_F gives λ = 1.11 and Z = 0.48. In the band map the dispersion flattens by 1/(1 + λ)
  within the phonon energies and the line broadens beyond them: the kink.
- **State-resolved Fan–Migdal self-energy** (`--self-energy fan-migdal`, 18³ q): with a fixed
  width of 5 meV it is noisy, because the level spacing |v_F| Δq of the q-grid is ≈ 0.3 eV;
  with the adaptive widths (`--adaptive-smearing 0.7`) it is smooth but structureless,
  Im Σ ≈ −7.3 meV and Re Σ ≈ −6.5 meV over the window, since the widths exceed the phonon
  energies. The plateau value agrees with the isotropic 8.7 meV to the anisotropy of this
  k-point. The kink of a metal is therefore resolved by the Eliashberg form, the
  state-resolved form being the tool for semiconductors (see the Si example) and for wide
  energy windows.
- Outputs: `arpes/output/{sigma_isotropic_<T>K.dat, arpes_bare.npz, arpes_isotropic_<T>K.npz,
  [self_energy_fan_migdal.npz, arpes_fan_migdal_<T>K.npz], arpes.png}`.

---

## Reference results (Pb, μ* = 0.1, σ = 0.05 eV)

| | λ | ω_log | Tc McMillan | Tc Allen–Dynes |
|---|---|---|---|---|
| PAOFLOW, k 48³ / q 24³ (this example: EPW `wannierize = .false.`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| PAOFLOW, k 24³ / q 24³ (this example) | 1.066 | 5.19 meV | 4.61 K | 4.94 K |
| PAOFLOW, k 48³ / q 24³, on the Wannierized EPW run (`epw_wannier.in`) | 1.221 | 4.50 meV | 4.77 K | 5.21 K |
| EPW itself, k 48³ / q 24³ (Wannierized run) | 1.151 | 4.44 meV | 4.38 K | 4.76 K |
| EPW tutorial 04 | 1.158 | 4.40 meV | 4.37 K | 4.75 K |

Migdal–Eliashberg (`me.py`) on the k 48³ / q 24³ α²F, against the tutorial (EPW's α²F,
λ = 1.158):

| | Δ(iω₀), 0.3 K | Z(iω₀), 0.3 K | gap edge (Padé / acon), 0.3 K | Tc, linearised kernel | Tc, Δ²(T) → 0 |
|---|---|---|---|---|---|
| PAOFLOW, k 48³ / q 24³ | 1.016 meV | 2.110 | 1.037 / 1.037 meV | 5.69 K | 5.71 K |
| PAOFLOW, k 24³ / q 24³ | 0.937 meV | 1.997 | 0.952 / 0.952 meV | 5.42 K | 5.44 K |
| EPW tutorial 04 | 0.923 meV | 2.071 | | ≈ 5.25 K | |

The larger gap and Tc follow the larger λ and ω_log of the PAO α²F: (Z₀ − 1)/λ = 0.925 in
both PAOFLOW and EPW. The remaining difference from EPW follows the density of states at
E_F of the two band interpolations: 0.251 /eV/spin for the 18-orbital PAO bands against
0.235 for EPW's 4 Wannier functions. `--coarse-q` (λ ≈ 1.6) is not converged in q and is
shown only as the cheaper workflow. Measured: Tc = 7.2 K, λ ≈ 1.55 (tunnelling; McMillan
and Rowell 1965); the PBE scalar-relativistic coupling of this example underestimates it,
and spin-orbit coupling raises λ (Heid *et al.* 2010).

---

## Notes

- **Commensurate grids:** the coarse q-grid must divide the coarse k-grid (here 6³ / 6³),
  and the dense `nk` must be a multiple of `nq`.
- **Explicit k list:** EPW requires the full grid as `K_POINTS crystal` in [0, 1).
  `PAOFLOW.gen.epw_inputs.kpoints_card((6, 6, 6))` writes exactly the list in
  `qe/nscf/nscf.in`. PAOFLOW recovers the grid from the list; with
  `coupling_source='dfpt'` the nscf may use `K_POINTS automatic 6 6 6 0 0 0` instead
  (detected by the drivers).
- **Band exclusion without Wannier functions:** with `wannierize = .false.` EPW takes the
  excluded bands from `pb.ukk` (`write_placeholder_ukk(..., exclude_bands=..., nelec=...)`);
  `bands_skipped` in `epw.in` only documents the choice. `PAOFLOW.gen.epw_inputs.epw_input`
  writes the matching `epw.in`, and `paoflow-gen elphon` (EPW source) generates both.
- **Wannierized EPW runs** work too (`qe/nscf/epw_wannier.in`), as long as the outer
  disentanglement window contains every band (no `dis_win_min/max`).
- **Spin-orbit coupling:** with a fully relativistic pseudopotential set
  `spin_degeneracy=1` in `system.py` (or leave `None`: it is read from the save).

## References

- W. L. McMillan, *Transition temperature of strong-coupled superconductors*, Phys. Rev. **167**, 331 (1968); P. B. Allen and R. C. Dynes, *Transition temperature of strong-coupled superconductors reanalyzed*, Phys. Rev. B **12**, 905 (1975) — Tc formulas.
- G. M. Eliashberg, *Interactions between electrons and lattice vibrations in a superconductor*, Sov. Phys. JETP **11**, 696 (1960); P. B. Allen and B. Mitrović, *Theory of superconducting Tc*, Solid State Phys. **37**, 1 (1982) — the Migdal–Eliashberg equations.
- E. R. Margine and F. Giustino, *Anisotropic Migdal-Eliashberg theory using Wannier functions*, Phys. Rev. B **87**, 024505 (2013) — the EPW implementation (isotropic and anisotropic) the solvers follow; H. J. Vidberg and J. W. Serene, J. Low Temp. Phys. **29**, 179 (1977) — Padé continuation.
- S. Poncé, E. R. Margine, C. Verdi and F. Giustino, Comput. Phys. Commun. **209**, 116 (2016); H. Lee *et al.*, npj Comput. Mater. **9**, 156 (2023) — EPW and its tutorial 04.
- P. B. Allen, *New method for solving Boltzmann's equation for electrons in metals*, Phys. Rev. B **17**, 3725 (1978) — the transport Eliashberg function α²F_tr and the resistivity.
- P. B. Allen, *Electron-phonon effects in the infrared properties of metals*, Phys. Rev. B **3**, 305 (1971) — the extended Drude model (frequency-dependent rate and mass enhancement); G. Grimvall, *The Electron-Phonon Interaction in Metals* (North-Holland, Amsterdam, 1981).
- S. Engelsberg and J. R. Schrieffer, *Coupled electron-phonon system*, Phys. Rev. **131**, 993 (1963) — the isotropic self-energy of the kink.
- A. Damascelli, Z. Hussain and Z.-X. Shen, *Angle-resolved photoemission studies of the cuprate superconductors*, Rev. Mod. Phys. **75**, 473 (2003) — ARPES and the electron-phonon kink; R. P. Day, B. Zwartsenberg, I. S. Elfimov and A. Damascelli, npj Quantum Mater. **4**, 54 (2019) — the photoemission matrix elements.
- W. L. McMillan and J. M. Rowell, *Lead phonon spectrum calculated from superconducting density of states*, Phys. Rev. Lett. **14**, 108 (1965) — the measured α²F of Pb; R. Heid, K.-P. Bohnen, I. Yu. Sklyadneva and E. V. Chulkov, *Effect of spin-orbit coupling on the electron-phonon interaction of the superconductors Pb and Tl*, Phys. Rev. B **81**, 174527 (2010).
- J. R. Yates, X. Wang, D. Vanderbilt and I. Souza, Phys. Rev. B **75**, 195121 (2007) — adaptive smearing.
