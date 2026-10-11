# Si phonon-assisted optical absorption — PAOFLOW on EPW's coarse coupling

This example computes the phonon-assisted (indirect) optical absorption of
silicon, the second-order photon + phonon process that dominates between the
indirect and the direct gap. PAOFLOW interpolates every ingredient in the PAO
basis: EPW's coarse electron-phonon coupling, the electrons and their velocity
matrix elements, and the phonons. It then evaluates the formula of EPW's
`lindabs` (`indabs.f90`; Noffsinger et al., Phys. Rev. Lett. **108**, 167402
(2012)). The direct (vertical) absorption is computed alongside.

The QE/EPW steps are those of `../Pb` and `../MgB2`: the coarse coupling from an
EPW run **without Wannier functions** (`wannierize = .false.`). The fine-grid
settings follow the `indabs` exercise of EPW tutorial 06
(<https://docs.epw-code.org/tutorials/tutorial_06/index.html>): 12³ k, 6³ q,
photon energies 0.05–3 eV, 300 K, `degaussw = 0.05` eV, `fsthick = 4` eV, and a
mid-gap Fermi level.

---

## Directory contents

| file | purpose |
|------|---------|
| `phonon/scf.in` | `pw.x` scf (18³ k, 38 Ry, PBE norm-conserving) |
| `phonon/ph.in` | `ph.x` DFPT on the 3³ q-grid (4 irreducible q, `fildvscf`) |
| `epw/nscf.in` | `pw.x` nscf on the full 6³ grid as an explicit `K_POINTS crystal` list (`nbnd = 21`) |
| `epw/epw.in` | `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, all 21 bands kept |
| `epw/write_ukk.py` | writes `Si2.ukk` (band bookkeeping for `wannierize = .false.`) and empty `Si2.bvec`/`Si2.mmn` stubs |
| `main.py` | PAOFLOW analysis: PAO electronic structure, then phonon-assisted + direct absorption |
| `optics/nscf.in` | `pw.x` nscf with 60 bands (8³ k) for the extended-basis direct dielectric function |
| `optics_eps.py` | PAOFLOW `dielectric_tensor` on the extended basis with the non-local velocity → `optics/output/epsr_*/epsi_*.dat` |
| `plot.py` | plots Im ε(ω) and α(ω), overlaying EPW's `epsilon2_indabs_<T>K.dat` when present; after `--emissivity` it also plots the thermal emissivity |

**Pseudopotential:** copy the PseudoDojo `Si.upf` (nc-sr-05, PBE standard;
`PSEUDOS/nc-sr-05_pbe_standard_upf/` in this repository) into this directory.
Both `phonon/` and `epw/` use `pseudo_dir = '../'`.

`paoflow-gen` (electron-phonon workflow, EPW source, property "phonon-assisted
optical absorption") produces this layout as `main.elphon.py` and
`plot.elphon.py`.

---

## Procedure

### 1. Quantum ESPRESSO and EPW

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out
python3 /path/to/q-e/EPW/bin/pp.py           # prefix 'Si2' -> collects dvscf, patterns, dyn into save/

cd ../epw
mkdir -p Si2.save
cp ../phonon/Si2.save/{charge-density.dat,data-file-schema.xml} Si2.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out
python3 write_ukk.py
mpirun -np 32 epw.x -nk 32 -in epw.in > epw.out   # one process per pool
```

The reference run used QE 7.5 / EPW 6.0 with 32 pools; EPW took 19 s.

### 2. PAOFLOW basis

```bash
paoflow-genbasis-ps --pseudo Si.upf --out BASIS_PS
```

`main.py` uses the `standard` configuration: 26 PAO orbitals for the cell.

### 3. PAOFLOW analysis

```bash
mpirun -np 8 python main.py                        # 12^3 k, 6^3 q, 300 K (13 s on 8 cores)
mpirun -np 8 python main.py --temps 10 300 600     # several temperatures in one run
mpirun -np 8 python main.py --nk 24 --nq 12        # denser grids (35 s on 8 cores)
mpirun -np 8 python main.py --no-nonlocal-velocity # PAO dH/dk velocity only
python plot.py --temp 300                          # figure -> output/absorption.png
```

Output goes to `output/`, one set of files per temperature in EPW's layout:
- `epsilon2_indabs_<T>K.dat`, `epsilon2_indabs_lorenz<T>K.dat`: photon energy
  and the phonon-assisted Im ε for EPW's nine intermediate-state broadenings η
  (0.001–0.5 eV), with a Gaussian or a Lorentzian energy-conserving delta
- `epsilon2_dirabs_<T>K.dat`: the direct Im ε along x, y, z and their average
- `alpha_<T>K.dat`: absorption coefficient (cm⁻¹, direct + phonon-assisted) per η
- `absorption.npz`: every result array, read by `plot.py`

### 4. Thermal emissivity of a wafer

The direct dielectric function is best taken from PAOFLOW's established
optical route: `dielectric_tensor` on the **extended** basis with the
non-local velocity, as in paoflow-gen's optical workflow. The extended basis
has 44 orbitals for Si (3s4s5s, 3p4p5p, 3d4d), so it needs a separate nscf with
more bands. EPW is not rerun.

```bash
mkdir -p optics/Si2.save && cp phonon/Si2.save/{charge-density.dat,data-file-schema.xml} optics/Si2.save/
# optics/nscf.in: nscf with nbnd = 60 on an 8x8x8 grid
(cd optics && mpirun -np 8 pw.x -nk 8 -in nscf.in > nscf.out)
python optics_eps.py   # extended basis, nonlocal velocity, interpolated 24^3, dielectric_tensor
mpirun -np 8 python main.py --emissivity --direct-dielectric optics/output \
    --nk 24 --nq 12 --omega-min 0.01 --omega-step 0.01 --degauss-ev 0.01 \
    --temps 300 400 500 600 700 900 1100 1300 1500       # 2 min on 8 cores
python plot.py                                           # -> output/emissivity.png
```

`optics_eps.py` runs:
- `projections(configuration='extended')`, `projectability`, `pao_hamiltonian`
- `interpolated_hamiltonian(24, 24, 24)`, `pao_eigh`
- `gradient_and_momenta(nonlocal_velocity=True)`
- `dielectric_tensor(emin=0.01, emax=10, ne=1000, d_tensor='diag', delta=0.1)`

Without `--direct-dielectric`, the direct term is computed from this run's
(standard) PAO basis by Kramers–Kronig up to `KK_OMEGA_MAX_EV = 10` eV.

By Kirchhoff's law the emissivity of a body at uniform temperature equals its
absorptance. `--emissivity` computes it for a free-standing slab of
`THICKNESS_UM = 500` µm (`--thickness-um`) at every temperature:

- **Fermi level:** the charge-neutral E_F(T) replaces the mid-gap value, so the
  density of thermally excited carriers, and their phonon-assisted
  free-carrier absorption, is included. These carriers dominate the absorption
  below the gap, where the blackbody spectrum of 300–1500 K lies.
- **Dielectric function (with `--direct-dielectric`):**
  - Re ε is the extended-basis one, plus the Kramers–Kronig shift from the
    phonon-assisted term.
  - Im ε is the extended-basis one above the direct gap. Below the gap it is
    this run's direct term (Gaussian, so no broadening tail) plus the
    phonon-assisted and free-carrier term.
  - The Drude–Lorentz tail of `dielectric_tensor` (ε₂ ≈ 0.15 at 0.5 eV for
    `delta = 0.1`) would otherwise make the wafer opaque below the gap.
- **Dielectric function (Kramers–Kronig fallback):** without the external
  file, Re ε comes from Kramers–Kronig of the direct term over all PAO bands up
  to 10 eV. The standard-basis spectrum above that violates the f-sum rule,
  and the printed ratio flags it.
- **Emissivity:** the Fresnel slab absorptance gives the spectral directional
  and hemispherical emissivity, and the Planck-weighted total at the same
  temperature. The opaque half-space, 1 − R, is reported alongside.
- **Outputs:**
  - `eps_<T>K.dat` (ω, Re ε, Im ε)
  - `emish_<T>K.dat` (ω, slab, opaque)
  - `emis_th<deg>_<T>K.dat`
  - `emist.dat` (T, total slab, total opaque, fraction of the blackbody
    spectrum inside the ω grid)
  - `emissivity.npz`

Total hemispherical emissivity of a 500 µm wafer (PBE, non-local velocity, k 24³ / q 12³):

| T (K) | 300 | 400 | 500 | 600 | 700 | 900 | 1100 | 1500 |
|---|---|---|---|---|---|---|---|---|
| slab, extended-basis direct ε | 0.001 | 0.06 | 0.32 | 0.55 | 0.61 | 0.62 | 0.62 | 0.62 |
| opaque half-space, extended-basis direct ε | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 | 0.62 |
| slab, Kramers–Kronig fallback | 0.002 | 0.06 | 0.35 | 0.59 | 0.63 | 0.64 | 0.64 | 0.63 |

The two routes differ only through ε₁ and hence R. The extended-basis
`dielectric_tensor` gives ε₁(0) = 17.3 (f-sum ratio 0.88); the standard-basis
Kramers–Kronig up to 10 eV gives 14.8. Both are above the DFPT ε∞ = 12.97,
which includes local fields.

- **The wafer goes from transparent to opaque.** At 300 K the wafer is
  transparent below the gap and nearly non-emissive. Thermally excited
  carriers make it opaque, and a 1 − R emitter, above about 700 K. The opaque
  half-space hardly depends on T, which is why the finite thickness is needed.
- **The transition comes too early.** With the PBE gap (0.61 eV, against
  1.12 eV in experiment) the intrinsic carriers turn on at too low a
  temperature. Lightly doped Si wafers turn opaque near 900–1000 K (Sato,
  Jpn. J. Appl. Phys. 6, 339 (1967)).
- **Free-carrier reflection at 1500 K.** At 1500 K the very high carrier
  density reflects the low photon energies (free-carrier plasma), lowering the
  spectral emissivity below about 0.2 eV.
- **Convergence.** The free carriers sit near the band edges. At 500 K the
  slab total is 0.11 at 12³/6³, 0.35 at 24³/12³ and 0.37 at 32³/16³, so use at
  least 24³/12³.

---

## Method

- **One gauge for g and v.** The coupling g(k, q) and the velocity matrix
  elements v(k), v(k+q) are rotated with the same dense-grid eigenvectors.
  The two time orderings of the second-order amplitude interfere, so this is
  required, not optional.
- **Velocity operator.** The velocity is the PAO analogue of EPW's
  `vme = 'wannier'`: ∂H/∂k plus the intersite position term
  i H_ij (τ_j − τ_i).
  - With `NONLOCAL_VELOCITY = True` (the default; norm-conserving
    pseudopotentials only) it also includes the non-local pseudopotential
    term, as in PAOFLOW's `dielectric_tensor`.
  - For Si this raises Im ε by about 3.5% (indirect) and 6% (direct).
- **Fermi level and gap.** The Fermi level sits mid-gap; set `FERMI_ENERGY_EV`
  to override it.
  - The band energies are the PBE ones, so the absorption edge sits at the PBE
    indirect gap (0.61 eV on the dense grid).
  - EPW tutorial 06 uses GW eigenvalues (`eig_read`), whose edge is near the
    experimental 1.1 eV.
- **Symmetry folding.** The dense q-grid is folded to its irreducible wedge.
  Only the polarisation average is exact under this folding, so it is reported
  in all three indirect components.

---

## Reference results (Si, PBE, 300 K, η = 0.05 eV, `degaussw` = 0.05 eV)

Gaps of the dense PAO spectrum: indirect 0.608 eV, direct (at Γ) 2.548 eV.

Phonon-assisted Im ε and α (direct + indirect, n_r = 3.4), without the non-local
velocity term:

| ω (eV) | Im ε 12³/6³ | 16³/8³ | 24³/12³ | α (cm⁻¹) 24³/12³ |
|---|---|---|---|---|
| 1.0 | 0.053 | 0.049 | 0.049 | 729 |
| 1.4 | 0.118 | 0.143 | 0.152 | 3.2 × 10³ |
| 1.8 | 0.292 | 0.354 | 0.325 | 8.7 × 10³ |
| 2.2 | 1.02 | 0.94 | 0.93 | 3.0 × 10⁴ |

- **Convergence.** From about 0.4 eV above the indirect gap, the spectrum is
  converged to about 10% at 24³/12³. The first 0.3 eV above the edge needs
  denser grids, as noted in the tutorial.
- **Broadening and temperature.** Between η = 0.001 and 0.1 eV, Im ε changes
  by about 1% or less below the direct gap. The absorption grows with temperature, as phonon
  absorption adds to emission. At 10 K only phonon emission remains, which
  shifts the edge up by about ω_TO.
- **Direct term cross-check.** The direct term agrees with PAOFLOW's
  `dielectric_tensor` on the same PAO Hamiltonian: peak heights to 5–9% and
  ∫ω Im ε dω to 5%, with the remainder from the different broadening forms.
- **Comparison with EPW.** A quantitative comparison with EPW's `lindabs`
  on the same data has not been done yet. It needs a Wannierized EPW
  calculation, because the `wannierize = .false.` run does not give a
  localized `epmatwp`. `plot.py` overlays EPW's `epsilon2_indabs_<T>K.dat`
  once it exists.

---

## Notes

- **Above the direct gap.** The second-order amplitude diverges at the direct
  transitions (S₁, S₂ denominators). EPW's `indabs` shares this divergence.
  Use the phonon-assisted spectrum below the direct gap and the direct spectrum
  above it.
- **Sub-gap tail.** The tail at high temperature is phonon-assisted
  free-carrier absorption by carriers thermally excited across the (small)
  PBE gap.
- **Semiconductors and insulators only.** For metals `thermal_emissivity`
  raises, because the intraband (Drude) term that sets their infrared
  reflectivity is not included yet.
- **Emissivity model.** The slab is incoherent: there is no thin-film
  interference. It has no multiphonon lattice absorption (the two-phonon bands
  of Si at 0.07–0.17 eV), no doping, and no band-gap change with temperature.
- **Polar materials.** Si is non-polar, so no long-range (Fröhlich) term is
  needed. PAOFLOW refuses polar materials (Z* ≠ 0) unless
  `allow_missing_long_range=True`.
