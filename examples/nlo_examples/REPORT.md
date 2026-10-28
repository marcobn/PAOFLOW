# Non-linear optics from the PAO Hamiltonian: validation report

*October 27 2026. Shift current, injection currents and second-harmonic generation (SHG) of
`PAOFLOW.nonlinear_optics` on three systems: gapped graphene (tight-binding model), GaAs
(bulk, cubic) and monolayer GeS (2D, polar). Every number comes from the output folders named
below; the figures are produced by `make_report_figures.py` from the same files. Run folders:
`Workspace/PAOFLOW/validation/nonlinear_optics_{GaAs,GeS}` (not bundled) and the examples here.*

> **Caveats.** The responses are the independent-particle ones of the PAO Hamiltonian (no
> excitons, no local fields, no phonon assistance); the scissor shifts the conduction energies
> while the matrix elements keep the DFT eigenvectors; the broadening η is a constant 0.05 eV; the
> degeneracy tolerance is 10 meV (pairs closer than that are dropped). The GaAs run uses the
> PAO Hamiltonian expanded from a *symmetry-reduced* nscf, which is symmetric only to about
> 1 meV in real space; the symmetry audit below is its error bar. Literature values are quoted
> from the sources named in §5; when read from published figures they are approximate ranges.

## 1. Bottom line

| system | quantity | PAOFLOW | literature | status |
|---|---|---|---|---|
| gapped graphene (TB) | D3h relations σ^{yyy} = −σ^{yxx} = −σ^{xxy}, σ^{xxx} = 0; same for χ | exact (relative deviations 1e-15, forbidden 4e-16 of the peak) | analytic | pass |
| GaAs | \|χ_xyz\| at the lowest photon energy (scissor 0.86 eV, 40³) | 110 pm/V | 166–180 pm/V measured at 0.12 eV; 207–245 pm/V TDDFT + scissor (0.7–0.9 eV) | 35–50 % low (see §3.2) |
| GaAs | \|χ_xyz\| at 1.5 eV, peak | 341 pm/V; 610 pm/V at 2.63 eV | hundreds of pm/V between 1.5 and 3 eV (E1 region) | order of magnitude and shape plausible |
| GaAs | σ_xyz shift current | −30 µA/V² at 3.9 eV, −8 at 2.0 eV | tens of µA/V² (Nastos–Sipe 2006, Ibañez-Azpiroz 2018) | order of magnitude |
| GaAs | symmetry audit | χ_yzx/χ_xyz within 5 % (1–2 eV); forbidden χ_xxx 0.01 % at 0 eV, 11 % at the 2.5 eV resonance; T-odd files 1e-5 of the natural scale | — | error bar 5–11 % |
| monolayer GeS | sheet shift current peaks σ^{xxx}, σ^{xyy}, σ^{yxy} | 46, 48, 26 nm µA/V² (2.84, 2.83, 2.05 eV) | 100–150 µA/V² quoted per layer thickness (Rangel 2017, later works) → ≈ 75 µA/V² here for 0.6 nm | order of magnitude |
| monolayer GeS | mirror-forbidden σ^{yyy}, σ^{xxy}; T-odd files | < 3e-8 nm µA/V²; identically 0 | symmetry | pass (nosym nscf) |

## 2. Methods

The implementation ([docs/internals/nonlinear-optics.md](../../docs/internals/nonlinear-optics.md))
evaluates, at every k-point of a dense grid interpolated from `HRs` with the pair-resolved
Wigner–Seitz cells, the band energies, the interband position matrix r = −i v/ω, its generalized
derivative from the sum rule over all PAO bands (including the ⟨n|∂²H|m⟩ term of a tight-binding
basis) and the velocity differences; the δ-function responses (shift and injection currents) are
accumulated directly on the photon grid, the SHG of Chen *et al.* (2022) through transition-energy
histograms. Unit tests pin the sum rule against covariant finite differences (1e-5), the Berry
curvature against the Chern number of the Haldane model, the inversion and translation
behaviour of the Rice–Mele chain, the D3h selection rules, the T-odd terms, the scissor, the
histogram scheme and the Kleinman symmetry of the static χ.

| system | electrons | dense grid | η | scissor | basis | wall time |
|---|---|---|---|---|---|---|
| gapped graphene | `{'label': 'graphene', 't': 2.7, 'delta': 1.0}` (gap 1 eV at K) | 300×300 | 0.05 | 0 | 2 orbitals | 2.4 s serial |
| GaAs | QE example10 (PAW, 60/600 Ry, 16³ nscf, nbnd 50, symmetry-reduced) | 40³ (60³ in progress) | 0.05 | 0.86 eV (PBE gap 0.534 → 1.39) | 62 PAO orbitals (internal BASIS), 18 bands > 0.95 | 15 min, 8 ranks |
| monolayer GeS | PBE, PseudoDojo NC, positions relaxed at a = 4.40, b = 3.68 Å, 18 Å cell, nosym 18×18×1 nscf, nbnd 40, gap 1.68 eV (direct 1.91) | 200×200 | 0.05 | 0 | 62 orbitals (`BASIS_PS` standard), 26 bands > 0.95 | 13 min, 8 ranks |

## 3. Results

### 3.1 Gapped graphene (tight-binding model)

![graphene](figures/fig1_graphene.png)

*Fig. 1. (a) Sheet shift current of the gapped honeycomb model; (b) SHG susceptibility of the sheet.*

The honeycomb lattice with a sublattice potential has point group D3h with a₁ along x. The
computed tensors obey the selection rules to machine precision: σ^{yyy} = 14.74 nm µA/V² at
1.07 eV, σ^{yxx} = σ^{xxy} = σ^{xyx} = −14.74, σ^{xxx} = −5e-15; the same relations hold for
χ (|χ^{yyy}| = 8.85e5 pm²/V at 0.51 eV, the two-photon resonance of the 1 eV gap). The circular
injection current vanishes (D3h is not gyrotropic: 6e-7 of the allowed scale) and the T-odd
files are identically zero. This is the test of the engine on a model with exact symmetry; the
same relations are enforced in the unit tests at 18² k.

### 3.2 GaAs

![gaas](figures/fig2_gaas.png)

*Fig. 2. (a) |χ_xyz|, Re χ_xyz, Im χ_xyz of GaAs (40³, scissor 0.86 eV, η 0.05 eV) with χ_yzx
on top of it, the measured low-frequency values (166–180 pm/V at 0.12 eV) and the static range
of a TDDFT + scissor calculation; (b) the shift current σ_xyz, σ_yzx and the forbidden σ_xxx.*

| photon energy (eV) | 0 | 1.0 | 1.5 | 2.0 | 3.0 | peak |
|---|---|---|---|---|---|---|
| \|χ_xyz\| (pm/V) | 110 | 152 | 341 | 254 | 348 | 610 at 2.63 |
| \|χ_yzx\| (pm/V) | 112 | 144 | 348 | 256 | 348 | 613 at 2.63 |
| \|χ_xxx\| (forbidden, pm/V) | 0.01 | 0.3 | 1.2 | 1.2 | 2.0 | 67 at 2.51 |
| σ_xyz (µA/V²) | 0 | 0 | −0.10 | −7.9 | −16.0 | −29.6 at 3.90 |
| σ_yzx (µA/V²) | 0 | 0 | −0.10 | −1.4 | −16.2 | −29.6 at 3.90 |
| σ_xxx (forbidden, µA/V²) | 0 | 0 | 0 | −0.05 | −0.05 | −0.42 |

Files: `Workspace/PAOFLOW/validation/nonlinear_optics_GaAs/paoflow/output_nk40_tol10meV/`.

**Static susceptibility.** The lowest-energy |χ_xyz| = 110 pm/V is below the measured
166–180 pm/V (0.12 eV) and the 207–245 pm/V of a TDDFT + scissor calculation (static limit,
scissor 0.7–0.9 eV). Independent-particle LDA/PBE + scissor values in the literature fall in the
same 100–250 pm/V bracket; the result is at its low end. Two known ingredients pull it up:
the k-convergence (40³ is not converged for sharp features: the Re/Im spectra are still spiky;
60³ is running) and the missing local-field and excitonic effects. The scissor itself lowers χ
(the 1/ω factors), as in all scissor calculations.

**Spectra.** |χ_xyz| rises through the two-photon onset at half the gap (0.7 eV), reaches
341 pm/V at 1.5 eV and peaks at 610 pm/V at 2.63 eV; the E1 region (3 eV) gives 348 pm/V.
Published spectra (Rashkeev 1998, Nastos 2005, Wang 2017 Fig. 4, the measurements of Bergfeld
and Daum 2003) show |χ| of several hundred pm/V between 1.5 and 3 eV with the strongest
structure near the E1 transitions; the shape is reproduced at the level of the broadening.

**Shift current.** σ_xyz starts at the scissored gap (1.4 eV), stays within −8 µA/V² up to
2 eV and reaches −30 µA/V² at 3.9 eV; the order of magnitude agrees with the LDA + scissor
calculations of Nastos and Sipe (2006) and Ibañez-Azpiroz *et al.* (2018), which also place the
largest values at 3–4 eV. Converged values need the 60³ grid (in progress) and a comparison of
the sign convention (J = σ E E*, see the docs).

**Symmetry audit.** χ_yzx tracks χ_xyz within 5 % between 1 and 2 eV and within 2 % elsewhere;
the forbidden χ_xxx is 0.01 % of χ_xyz at 0 eV but 11 % at the 2.5 eV resonance; σ_xxx is 1.4 %
of the peak σ_xyz; the T-odd shift and injection files and the circular injection are 1e-5 of
their natural scale. The real-space images of the PAO Hamiltonian expanded from the
symmetry-reduced nscf are symmetric only to 8e-4 eV (eigenvalues at off-grid k spread by up to
12 meV over the 24 T_d images), which the 1/ω factors amplify; a no-symmetry nscf (4096 k) is
running to remove this error bar. With the former 0.1 meV degeneracy tolerance the leakage was
33 % and the shift current had a spurious −218 µA/V² peak at 1.61 eV: that is why the default
is now 10 meV.

### 3.3 Monolayer GeS

![ges](figures/fig3_ges.png)

*Fig. 3. (a) Sheet shift current of monolayer GeS (nosym nscf, BASIS_PS, 200², η 0.05 eV, no
scissor), the mirror-forbidden σ^{yyy} on the axis, and the literature range of 100–150 µA/V²
converted with a 0.6 nm layer; (b) the sheet SHG susceptibility.*

| component | shift current peak (nm µA/V²) | at 2.0 eV | at 3.0 eV | \|χ\| peak (pm²/V) |
|---|---|---|---|---|
| xxx | 46.0 at 2.84 eV | −5.8 | 36.4 | 8.7e5 at 3.06 eV |
| xyy | 47.6 at 2.83 eV | 1.4 | 24.2 | 9.8e5 at 2.49 eV |
| yxy | 26.0 at 2.05 eV | 19.4 | 16.0 | 2.2e6 at 1.87 eV |
| yyy, xxy (forbidden) | < 3e-8 | | | < 0.02 |

Files: `Workspace/PAOFLOW/validation/nonlinear_optics_GeS/paoflow/output_nk200_ps/`.

The polar monolayer (mm2, polar axis x) has three independent in-plane shift-current components;
the mirror-forbidden ones vanish to 1e-9 of the allowed scale and the T-odd files are identically
zero, because the Hamiltonian comes from a no-symmetry nscf (time reversal and the mirror are
then exact). The shift current starts at the direct gap (1.9 eV) and peaks at 46–48 nm µA/V²
near 2.85 eV; divided by a 0.6 nm layer thickness this is ≈ 75 µA/V², the order of magnitude of
the 100–150 µA/V² quoted for monolayer GeS (Rangel *et al.* 2017 and later works, with their own
thickness conventions and a GW-level gap). The SHG sheet susceptibility peaks at 2.2e6 pm²/V
(yxy, 1.87 eV); published χ^{(2)} of group-IV monochalcogenide monolayers are of 1e5–1e6 pm²/V
depending on the thickness convention. A quantitative comparison needs the same gap (scissor to
the GW value) and normalisation.

A first attempt with the repository `BASIS` tables gave a projectability of 0.90 for the Ge 3d
bands, only 14 bands above 0.95 and a gapless PAO Hamiltonian; the tables generated from the
pseudopotentials (`paoflow-genbasis-ps`, preset standard) give 26 bands above 0.95 (all 20
valence and 6 conduction) and the correct gap. The in-plane cell was fixed at the PBE literature
values because a free 2D `vc-relax` converged to a cell 8 % too small.

## 4. Convergence and open items

- **GaAs k-grid.** 40³ gives spiky Re/Im χ at η = 0.05 eV; the 60³ run (in progress, ~1 h on
  8 ranks) will be added to Fig. 2 and this table. The static value and the peak positions are
  the quantities to compare.
- **GaAs symmetry.** The no-symmetry nscf (in progress) removes the 5 % spread between
  equivalent components; the examples use nosym from the start.
- **Scissor for GeS** to the GW gap, and the thickness convention of the literature, before
  quoting percentages.
- **Beyond the tight-binding position operator** (Löwdin position dipoles in r and r_{;a}),
  the U(N)-covariant treatment of degenerate bands, and symmetry reduction of the k-grid are
  the planned follow-ups.

## 5. Literature used

- GaAs low-frequency χ_xyz: 180 ± 10 pm/V (Levine and Bethea), 166 pm/V (Roberts), 172 pm/V at
  0.118 eV (Eyres *et al.*), as summarised in the ab initio comparisons of Luppi, Hübener and
  Véniard, Phys. Rev. B **82**, 235201 (2010) and J. Chem. Phys. **132**, 241104 (2010); their
  TDDFT + scissor static |χ_xyz| = 245, 224, 207 pm/V for scissors of 0.7, 0.8, 0.9 eV.
- GaAs SHG spectra: S. Bergfeld and W. Daum, Phys. Rev. Lett. **90**, 036801 (2003) (measured
  |χ_xyz(ω)| for two-photon energies 2–5 eV); S. N. Rashkeev, W. R. L. Lambrecht and B. Segall,
  Phys. Rev. B **57**, 3905 (1998); F. Nastos, B. Olejnik, K. Schwarz and J. E. Sipe, Phys. Rev. B
  **72**, 045223 (2005); C. Wang *et al.*, Phys. Rev. B **96**, 115147 (2017), Fig. 4.
- GaAs shift current: F. Nastos and J. E. Sipe, Phys. Rev. B **74**, 035201 (2006);
  J. Ibañez-Azpiroz, S. S. Tsirkin and I. Souza, Phys. Rev. B **97**, 245143 (2018).
- Monolayer GeS: T. Rangel *et al.*, Phys. Rev. Lett. **119**, 067402 (2017) (≈ 100 µA/V²);
  C. Wang *et al.* (2017), Fig. 1(c); later works quoting 140–150 µA/V²; H. Chen *et al.*, Phys.
  Rev. B **105**, 075123 (2022) for the SHG formulation.

## 6. Reproduce

```bash
cd examples/nlo_examples/Graphene && python graphene_nonlinear.py
cd ../GaAs   && ln -s ../../qe_examples/example10/GaAs.save . && mpirun -np 8 python main.py
cd ../GeS    && (QE scf/nscf, paoflow-genbasis-ps as in README) && mpirun -np 8 python main.py
cd .. && python make_report_figures.py
```
