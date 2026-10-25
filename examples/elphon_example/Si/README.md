# Si phonon-assisted optical absorption — PAOFLOW's native coupling

This example computes the phonon-assisted (indirect) optical absorption of
silicon, the second-order photon + phonon process that dominates between the
indirect and the direct gap. PAOFLOW interpolates every ingredient in the PAO
basis: the coarse electron-phonon coupling, the electrons and their velocity
matrix elements, and the phonons. It then evaluates the formula of EPW's
`lindabs` (`indabs.f90`; Noffsinger et al., Phys. Rev. Lett. **108**, 167402
(2012)). The direct (vertical) absorption is computed alongside.

The coarse coupling is computed by PAOFLOW itself from the `ph.x` outputs and
the nscf wavefunctions (`COUPLING_SOURCE = 'dfpt'` in the drivers): only `pw.x`
and `ph.x` are run, no EPW. The nscf is run on the irreducible wedge
(`K_POINTS automatic`), and PAOFLOW unfolds its Bloch functions to the full
grid by symmetry. The EPW route (`epw.x` with `epbwrite`, `wannierize =
.false.`, as in `../Pb` and `../MgB2`) remains available as the alternative
source and is described at the end of section 1. The fine-grid
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
| `phonon/q2r_matdyn.py` | `q2r.x` + `matdyn.x` in PAOFLOW: force constants (`output/Si2.fc`), dispersion and DOS from the `Si2.dyn*` files, plotted with `GPAO.plot_phonons` |
| `nscf/nscf.in` | `pw.x` nscf on the 6³ coarse grid, `K_POINTS automatic` (16 irreducible k, `nbnd = 21`) |
| `nscf/kmesh.py` | sets the coarse k-mesh: `python kmesh.py NK1 NK2 NK3` rewrites the `K_POINTS` card of `nscf.in` (`--explicit` for the full list EPW needs), `nk1-3` of `epw.in`, `write_ukk.py` and `COARSE_GRID` of `main.py`/`mobility.py` (the q-grid must divide it) |
| `nscf/epw.in` | EPW route only: `epw.x` with `epbwrite = .true.`, `wannierize = .false.`, all 21 bands kept |
| `nscf/write_ukk.py` | EPW route only: writes `Si2.ukk` (band bookkeeping for `wannierize = .false.`) and empty `Si2.bvec`/`Si2.mmn` stubs |
| `main.py` | PAOFLOW analysis: PAO electronic structure, then phonon-assisted + direct absorption |
| `optics/nscf.in` | `pw.x` nscf with 60 bands (8³ k) for the extended-basis direct dielectric function |
| `optics_eps.py` | PAOFLOW `dielectric_tensor` on the extended basis with the position dipoles in the velocity → `optics/output/epsr_*/epsi_*.dat` |
| `emissivity_from_absorption.py` | recomputes the thermal emissivity of a finished run from `absorption.npz` with another direct dielectric function or `--direct-scissor` (writes to a new directory) |
| `make_hubbard_nscf.py` | from the converged `HUBBARD` card of an eACBN0 run: the nscf inputs for eACBN0 electrons (`acbn0/nscf/`, the EPW k list) and for the eACBN0 optics run (`acbn0/optics/`) |
| `plot.py` | plots Im ε(ω) and α(ω), overlaying EPW's `epsilon2_indabs_<T>K.dat` when present; after `--emissivity` it also plots the thermal emissivity |
| `mobility.py` | phonon-limited carrier mobility: state-resolved SERTA/MRTA rates near the band edges, electron and hole mobilities, `scattering_rates.npz` for `pf.transport` |
| `plot_mobility.py` | plots μ(T) and 1/τ against the energy from the band edge → `output/mobility/mobility.png` |

**Pseudopotential:** copy the PseudoDojo `Si.upf` (nc-sr-05, PBE standard;
`PSEUDOS/nc-sr-05_pbe_standard_upf/` in this repository) into this directory.
Both `phonon/` and `nscf/` use `pseudo_dir = '../'`.

`paoflow-gen` (electron-phonon workflow, EPW source, property "phonon-assisted
optical absorption") produces this layout as `main.elphon.py` and
`plot.elphon.py`, with the nscf and EPW inputs under its default directory
`epw/` (option `epw_dir`). The property "carrier transport (mobility,
SERTA/MRTA)" produces `mobility.py` and `plot_mobility.py` under the same names.

---

## Procedure

### 1. Quantum ESPRESSO

```bash
cd phonon
mpirun -np 8 pw.x -in scf.in > scf.out
mpirun -np 8 ph.x -in ph.in  > ph.out        # 3^3 q-grid, 4 irreducible q, fildvscf -> _ph0/

python q2r_matdyn.py                         # optional: phonon dispersion + DOS (q2r + matdyn) -> output/si_phonons.png

cd ../nscf
mkdir -p Si2.save
cp ../phonon/Si2.save/{charge-density.dat,data-file-schema.xml} Si2.save/
mpirun -np 8 pw.x -in nscf.in > nscf.out     # 6^3 grid, K_POINTS automatic: 16 irreducible k
```

That is all the native coupling needs: PAOFLOW reads the `ph.x` outputs
(`phonon/_ph0/` and `phonon/Si2.dyn*`, or a `pp.py` `save/`) and the nscf
wavefunctions of `nscf/Si2.save`, and rotates the 16 stored Bloch functions to the
216 points of the grid by symmetry (the drivers detect an irreducible-wedge
save). Another coarse mesh: `python kmesh.py 8 8 8` (any multiple of the 3³
q-grid), then rerun the nscf. The native vertex reproduces EPW's `.epb` matrix
elements to the rounding level (`docs/internals/dfpt-coupling-port.md`).

**EPW route (alternative, required for ultrasoft or PAW pseudopotentials).**
Set `COUPLING_SOURCE = 'epw'` in the drivers and run EPW without Wannier
functions on the explicit full k list:

```bash
cd phonon && python3 /path/to/q-e/EPW/bin/pp.py   # prefix 'Si2' -> collects dvscf, patterns, dyn into save/
cd ../nscf
python kmesh.py 6 6 6 --explicit                   # K_POINTS crystal list (216 points), epw.in, write_ukk.py
mpirun -np 8 pw.x -in nscf.in > nscf.out
python3 write_ukk.py
mpirun -np 32 epw.x -nk 32 -in epw.in > epw.out    # one process per pool; writes Si2.epb*
```

The reference EPW run used QE 7.5 / EPW 6.0 with 32 pools; EPW took 19 s.

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
mpirun -np 8 python main.py --no-position-dipoles # PAO dH/dk velocity only
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
position dipoles in the velocity (`gradient_and_momenta(position_dipoles=True)`;
until October 2026 these runs used the non-local term instead, see the
velocity-operator note below), as in paoflow-gen's optical workflow. The extended basis
has 44 orbitals for Si (3s4s5s, 3p4p5p, 3d4d), so it needs a separate nscf with
more bands. EPW is not rerun.

```bash
mkdir -p optics/Si2.save && cp phonon/Si2.save/{charge-density.dat,data-file-schema.xml} optics/Si2.save/
# optics/nscf.in: nscf with nbnd = 60 on an 8x8x8 grid
(cd optics && mpirun -np 8 pw.x -nk 8 -in nscf.in > nscf.out)
python optics_eps.py   # extended basis, position dipoles, interpolated 24^3, dielectric_tensor
mpirun -np 8 python main.py --emissivity --direct-dielectric optics/output \
    --nk 24 --nq 12 --omega-min 0.01 --omega-step 0.01 --degauss-ev 0.01 \
    --temps 300 400 500 600 700 900 1100 1300 1500       # 2 min on 8 cores
python plot.py                                           # -> output/emissivity.png
```

`optics_eps.py` runs:
- `projections(configuration='extended')`, `projectability`, `pao_hamiltonian`
- `interpolated_hamiltonian(24, 24, 24)`, `pao_eigh`
- `gradient_and_momenta(position_dipoles=True)`
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

**Band-gap correction (scissor).** ph.x supports Hubbard U only with atomic
projectors, which barely changes the Si gap. The e-ph calculation therefore
stays PBE, and a scissor fixes the band energies:

```bash
# experimental gap, temperature dependent (Varshni, Si: E0 = 1.17 eV, alpha = 4.73e-4 eV/K, beta = 636 K)
mpirun -np 8 python main.py --emissivity --direct-dielectric optics/output \
    --target-gap-ev 1.17 --varshni 4.73e-4 636 --nk 24 --nq 12 \
    --omega-min 0.01 --omega-step 0.01 --degauss-ev 0.01 --temps 300 500 700 900 1100 1300 1500
# or the eACBN0 gap: --gap-reference-save <eACBN0 run>/<prefix>.save  (its PAO gap on the dense grid)
```

- **What changes:** the conduction bands are shifted rigidly. The interband
  velocities are renormalized (Levine–Allan; `--scissor-velocity ks` keeps the
  Kohn–Sham ones, as EPW's `scissor` does). The extended-basis direct ε gets
  the same shift.
- **Cost:** a temperature-dependent gap repeats the second-order kernel per
  temperature, 3.5 min instead of 2 for 8 temperatures.

With the Varshni gap the wafer turns opaque near 700–900 K, as in experiment:

| T (K) | 300 | 500 | 700 | 900 | 1100 | 1500 |
|---|---|---|---|---|---|---|
| slab, Varshni scissor, extended-basis ε | 0.000 | 0.003 | 0.25 | 0.61 | 0.63 | 0.62 |
| opaque half-space | 0.64 | 0.64 | 0.63 | 0.63 | 0.63 | 0.62 |

At 300 K, with a fixed 1.12 eV scissor, the phonon-assisted absorption
coefficient is close to the experimental curve. These experimental values are
approximate:

| Photon energy | 1.55 eV | 1.75 eV | 2.05 eV |
|---|---|---|---|
| computed α | 660 cm⁻¹ | 1500 cm⁻¹ | 3800 cm⁻¹ |
| experiment | ≈ 850 cm⁻¹ | ≈ 1900 cm⁻¹ | ≈ 4100 cm⁻¹ |

Without the scissor (PBE gap), the totals below come out at too low a
temperature.

Total hemispherical emissivity of a 500 µm wafer (PBE gap, former non-local velocity term as it was before October 2026, k 24³ / q 12³):

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
- **The transition comes too early without the scissor.** With the PBE gap
  (0.61 eV, against 1.12 eV in experiment) the intrinsic carriers turn on at
  too low a temperature; see the scissor table above. Lightly doped Si wafers turn opaque near 900–1000 K (Sato,
  Jpn. J. Appl. Phys. 6, 339 (1967)).
- **Free-carrier reflection at 1500 K.** At 1500 K the very high carrier
  density reflects the low photon energies (free-carrier plasma), lowering the
  spectral emissivity below about 0.2 eV.
- **Convergence.** The free carriers sit near the band edges. At 500 K the
  slab total is 0.11 at 12³/6³, 0.35 at 24³/12³ and 0.37 at 32³/16³, so use at
  least 24³/12³.

**eACBN0 electrons instead of a scissor.** The scissor keeps the PBE
eigenvectors and renormalises the velocities. With an eACBN0 (DFT+U+V,
ortho-atomic) run of this crystal — `examples/acbn0_examples/Si`, run on the
same `Si.upf` and lattice (the loop gives U_p ≈ 3.0 eV, V_pp ≈ 1.7 eV and a gap
of ≈ 1.2 eV) — the band energies **and eigenvectors** can come from its PAO
Hamiltonian, while the coupling in the orbital basis stays the PBE one. Neither
`ph.x` nor `epw.x` accepts U+V or ortho-atomic projectors, so this mixing is
done by PAOFLOW (`--electrons-save`). It needs two pw.x nscf runs with the
converged `HUBBARD` card, on the EPW k list (21 bands, the Hamiltonian) and on an
automatic grid with 60 bands (the extended-basis dielectric function), both
started from the eACBN0 save (charge density **and** `occup.txt`):

```bash
python make_hubbard_nscf.py <eACBN0 workdir>/scf.in --nk 6 6 6 --nbnd 21   # -> acbn0/nscf/nscf.in, acbn0/optics/nscf.in
for run in nscf optics; do
  cp <eACBN0 outdir>/Si2.save/{charge-density.dat,data-file-schema.xml,occup.txt} acbn0/$run/Si2.save/; done
(cd acbn0/nscf && mpirun -np 8 pw.x -in nscf.in > nscf.out)
(cd acbn0/optics && mpirun -np 8 pw.x -nk 8 -in nscf.in > nscf.out)
python optics_eps.py --savedir acbn0/optics/Si2.save --outdir acbn0/optics/output
mpirun -np 8 python main.py --electrons-save acbn0/nscf/Si2.save --emissivity \
    --direct-dielectric acbn0/optics/output --varshni 4.73e-4 636 --nk 24 --nq 12 \
    --omega-min 0.01 --omega-step 0.01 --degauss-ev 0.01 --temps 300 500 700 900 1100 1300 1500
```

- The two saves must share the PAO basis (pseudopotential, `BASIS_PS`,
  `'standard'`, `PTHR`, structure and atom order); `main.py` checks it and
  stops otherwise. The eACBN0 nscf may also be on another grid.
- No 0 K scissor: `--varshni` alone applies `-alpha T^2/(T + beta)` on top of
  the eACBN0 gap. `--gap-reference-save` is not needed any more (and is refused
  together with `--electrons-save`).
- `--direct-dielectric` must point to the eACBN0 optics run: `main.py` warns if
  the onset of the external ε₂ and the Hamiltonian's direct gap disagree.
- With a scissor on the eACBN0 bands (`--target-gap-ev 1.17`, the eACBN0
  indirect gap being 0.22 eV too high while its direct gap is right), add
  `--direct-scissor thermal`: the external ε then receives only the
  temperature-dependent part of the scissor. For a finished run,
  `python emissivity_from_absorption.py output --direct-dielectric <optics output>
  --direct-scissor thermal --outdir output_thermal` recomputes the emissivity
  in seconds.
- Not included: the Hubbard contribution to the coupling and to the velocity
  commutator. Compare the f-sum ratio printed by the emissivity step with the
  PBE one (0.87–0.88) to size the latter.
- The DFPT coupling is overscreened (ε = 12.97 against 11.94 measured), with
  either electrons. `--screening-eps 11.94` applies the correction of Poncé,
  Margine and Giustino (2018): g is multiplied by ε_DFT(|q|)/ε(|q|) of Resta's
  model. It raises the phonon-assisted ε₂ by 3–5 % and α below the direct edge
  by 4 %. Give it the measured constant, not the eACBN0 ε₁(0) of
  `optics_eps.py`, which is an independent-particle value on another footing.
- `absorption.npz` records `electrons_save`, `electrons_dft` and the Hubbard
  parameters (`electrons_hubbard`).

### 5. Carrier mobility (SERTA / MRTA)

The same EPW coupling gives the phonon-limited electron and hole mobilities.
- **Rates.** `mobility.py` computes the state-resolved rates 1/τ_nk(T) in the
  self-energy (SERTA) and momentum-relaxation (MRTA) approximations, for the
  states within 0.3 eV of the band edges.
- **Transport.** It then evaluates the k-resolved conductivity, carrier
  densities, mobilities and Seebeck coefficient on the same dense grid.

```bash
mpirun -np 20 python mobility.py                       # 48^3 k = q, 200-500 K, intrinsic (3.5 min)
mpirun -np 20 python mobility.py --nk 72 --nq 72       # 14 min (3^3 DFPT q)
mpirun -np 20 python mobility.py --doping 1e18         # n type (cm^-3; negative = p type)
python plot_mobility.py                                # -> output/mobility/mobility.png
```

**Benchmark.** The reference is Poncé, Margine and Giustino, Phys. Rev. B
**97**, 121201(R) (2018), at the same level of theory: PBE, PBE lattice
(5.47 Å), no spin–orbit and no GW or screening corrections. That calculation
gives μ_e = 1343 and μ_h = 722 cm²/Vs, converged with about 85k irreducible k
and 200k irreducible q. Experiment gives about 1400 and 470–500.

At 48³ k = q, σ = 10 meV and 300 K, MRTA (cm²/Vs):

| coupling (DFPT q-grid) | H(R) of the band structure | μ_e | μ_h |
|---|---|---|---|
| 3³ (this example) | EPW 6³ nscf | 1763 | 424 |
| 3³ | 18³ scf, same PAO basis | 1376 | 407 |
| 6³ | EPW 6³ nscf | 1492 | 440 |
| 6³ | 18³ scf | 1167 | 423 |

What matters, in order:
- **Band velocities.** They are the group velocities ∂ε/∂k. Earlier versions
  added the non-local pseudopotential correction to them, which made μ_e 12%
  too high. That correction applies to interband optical dipoles only.
- **Band structure.** Interpolating from EPW's 6³ nscf makes the transverse
  electron mass 10% too light and the heavy holes 7–10% too light, compared
  with QE on the same k-points. The coupling is in the fixed PAO orbital gauge,
  so the band structure can come from a denser PAO run in the same basis. The
  18³ scf brings m_t and the hole masses within 3% of QE (m_l is 13% heavy).
- **Coupling grid.** Going from a 3³ to a 6³ DFPT q-grid lowers μ_e by
  15–20%; μ_h changes by less than 5%.
- **k/q sampling.** Uniform grids converge slowly: with the earlier velocities
  μ_h rose 418 → 482 → 504 from 48³ to 72³, and μ_e rose about 6%. The paper
  uses quasi-random grids densified at the band extrema.
- **Not the cause.** The smearing: at 48³, σ = 5, 10 and 20 meV change the
  results by less than 10%. Nor the relaxation-time approximation: MRTA ≈
  SERTA within 3%, and the paper's IBTE differs from SERTA by +6% for electrons
  and −1% for holes.

**Screening correction (Poncé 2018).** `screening_eps=11.94` (`--screening-eps`)
rescales g by ε_DFT(|q|)/ε(|q|) of Resta's model, with ε_DFT = 12.97 from the
`.epb` files (`elphon/coupling_screening.py`). On the 6³-q coupling with H(R)
from the EPW 12³ nscf (48³ k = q, σ = 10 meV, 300 K, MRTA):

| | μ_e | μ_h |
|---|---|---|
| DFPT screening | 1292 | 432 |
| ε → 11.94 | 1164 | 371 |
| change | −9.9 % | −14.2 % |
| Poncé *et al.* 2018 | −8.8 % | −12.4 % |

SERTA changes by the same amounts. The correction belongs on top of a
converged PBE baseline, as in the paper; it moves the holes further below the
PBE reference.

With all corrections, the electrons are within the remaining grid convergence
of the reference. The holes are about 40% low, and that is still open.

`--doping 1e18` gives n_e = 1.0·10¹⁸ cm⁻³ and the intrinsic μ_e, as expected
in the non-degenerate regime. The calculation is phonon-limited only; add the
ionized-impurity scattering through the Boltzmann bridge.

**Boltzmann bridge.** `output/mobility/scattering_rates.npz` holds the
transport-weighted τ_eff(E, T) for PAOFLOW's Boltzmann module. The channel is
aligned to the band edges of the run that uses it:

```python
from PAOFLOW.boltzmann.scattering_channels import EnergyResolvedRate
eph = EnergyResolvedRate.from_npz('output/mobility/scattering_rates.npz', 'mrta')
pf.transport(tmin=300, tmax=300, nt=1, emin=-0.3, emax=1.0, ne=261,
             scattering_channels=[eph, 'impurity'], tau_dict={...})
```

At 36³, through `pf.transport` on PAOFLOW's band structure of the 18³ scf save,
the bridge reproduces the k-resolved mobilities to 15–17%. That is the cost of
an energy-only τ.

---

`mobility.py --electrons-save acbn0/nscf/Si2.save [--varshni 4.73e-4 636]`
takes the gap, the masses and the carrier statistics from the eACBN0 PAO
Hamiltonian (see section 4); the rates keep the PBE coupling. The energies of
`scattering_rates.npz` are then on the scale of that save.

## Method

- **One gauge for g and v.** The coupling g(k, q) and the velocity matrix
  elements v(k), v(k+q) are rotated with the same dense-grid eigenvectors.
  The two time orderings of the second-order amplitude interfere, so this is
  required, not optional.
- **Velocity operator.** The velocity is the PAO analogue of EPW's
  `vme = 'wannier'`: ∂H/∂k plus the intersite position term
  i H_ij (τ_j − τ_i).
  - With `POSITION_DIPOLES = True` (the default) it also includes the
    position dipoles of the PAO orbitals, i[H, d(k)] with the
    Löwdin-orthogonalised two-centre dipoles (`hamiltonian/position_dipoles.py`),
    the analogue of EPW's Berry-connection term. Checked against the exact
    plane-wave velocities of the Kohn–Sham states at the coarse k-points: the
    plain ∂H/∂k overestimates the valence–conduction |v|² by 1.4–2.3, with the
    dipoles it is within 10%; for Si this lowers Im ε by 6% (indirect) and
    7.5% (direct, 2.5–5 eV), and the direct term lands at 0.98 of EPW's.
    `POSITION_DIPOLE_LEVEL = 'intra-atomic'` keeps the on-site blocks only
    (the textbook approximation; not enough for the diffuse standard basis).
  - The non-local pseudopotential term i[V_NL, r] (`NONLOCAL_VELOCITY`, until
    October 2026) was removed: the velocity i[H, r] of the PAO Hamiltonian
    already contains V_NL, so with the dipoles it is complete (0.99–1.01 of
    the exact Im ε of Si) and the term on top double counted (0.87;
    `docs/internals/nonlocal-velocity.md`). Passing `nonlocal_velocity=` to
    the drivers now raises.
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

Phonon-assisted Im ε and α (direct + indirect, constant n_r = 3.4 here; with
`--direct-dielectric` the drivers use n(ω) of that run), with the plain velocity
(no position dipoles):

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
- **Comparison with EPW.** Done on the inputs of EPW tutorial 06 (Si, same
  nscf, same `.epb` coupling, 12³/6³ dense grids, 300 K; the Wannierised
  `epw1` must be rerun with `dis_win_max` above the top nscf band, since the
  reader refuses window-compacted `.epb` files). Both codes implement the same
  CHBB formula; the phonon-assisted Im ε is 0.79 (3³ q) / 0.91 (6³ q coarse
  grids) of EPW's between 0.8 and 2.5 eV and the direct term 0.98 (with the
  position dipoles), the coupling elements agree to ±10%, and the residual is
  the band interpolation: EPW's 8 sp3 Wannier functions on a 6³ grid put the
  conduction minimum 84 meV too high with twice the longitudinal mass (QE and
  PAO agree), which also doubles EPW's electron scattering rates; a 12³ coarse
  grid cures it. `plot.py` overlays EPW's `epsilon2_indabs_<T>K.dat`.

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
