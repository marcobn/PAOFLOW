# Non-linear optics from the PAO Hamiltonian: validation report

*October 27 2026. Shift current, injection currents and second-harmonic generation (SHG) of
`PAOFLOW.nonlinear_optics` on three systems: gapped graphene (tight-binding model), GaAs
(bulk, cubic) and monolayer GeS (2D, polar). Every number comes from the output folders named
below; the figures are produced by `make_report_figures.py` from the same files. Run folders:
`Workspace/PAOFLOW/validation/nonlinear_optics_{GaAs,GeS}` (not bundled) and the examples here.*

> **Caveats.** The responses are the independent-particle ones of the PAO Hamiltonian (no
> excitons, no local fields, no phonon assistance); the scissor shifts the conduction energies
> while the matrix elements keep the DFT eigenvectors; the broadening η is a constant 0.05 eV; the
> degeneracy tolerance is 10 meV (pairs closer than that are dropped). The GaAs runs use PAW
> pseudopotentials: with PAW/ultrasoft pseudopotentials the internal projections re-orthonormalise
> the pseudo-wavefunctions without the augmentation charges, which breaks the point symmetry of the
> PAO Hamiltonian at a few high-symmetry k-points (§3.2); the symmetry audit below is the resulting
> error bar. Literature values are quoted
> from the sources named in §5; when read from published figures they are approximate ranges.

## 1. Bottom line

| system | quantity | PAOFLOW | literature | status |
|---|---|---|---|---|
| gapped graphene (TB) | D3h relations σ^{yyy} = −σ^{yxx} = −σ^{xxy}, σ^{xxx} = 0; same for χ | exact (relative deviations 1e-15, forbidden 4e-16 of the peak) | analytic | pass |
| GaAs | \|χ_xyz\| at the lowest photon energy (scissor 0.86 eV) | 120 pm/V (PAW, augmented projections, 40³); 103 pm/V (norm-conserving, 40³); 110 pm/V (first PAW runs, 40³ and 60³) | 166–180 pm/V measured at 0.12 eV; 207–245 pm/V TDDFT + scissor (0.7–0.9 eV) | 35–50 % low (see §3.2) |
| GaAs | \|χ_xyz\| at 1.5 eV; at 3 eV; peak | PAW (augmented) 416; 397; 725 pm/V at 2.63 eV; norm-conserving 320; 206; 492 pm/V at 2.54 eV; first PAW runs 337–341; 348 (40³) → 390 (60³); 610 (40³) → 480 (60³) | hundreds of pm/V between 1.5 and 3 eV (E1 region) | order of magnitude and shape plausible; the resonance region is not converged at 60³ |
| GaAs | σ_xyz shift current | −32 µA/V² at 3.9 eV, −3.6 at 2.0 eV (PAW, augmented, 40³); −28 and −2.3 (norm-conserving 40³); −27 (60³) to −30 (40³) in the first PAW runs | tens of µA/V² (Nastos–Sipe 2006, Ibañez-Azpiroz 2018) | order of magnitude |
| GaAs | symmetry audit (current code) | PAW 40³: σ_xyz = σ_yzx and σ_xxx = 0 to 3e-8 of the peak, T-odd files 1e-9, χ_yzx/χ_xyz within 1.7 % (1–2 eV) and 1 % (2–4 eV), χ_xxx 6e-5 of the peak; norm-conserving 40³: shift current exact to 1e-10, χ_yzx/χ_xyz within 10 % / 2 %, χ_xxx 1.2 % (degenerate-gauge error of the SHG kernel, §3.2). First PAW runs (projection defect, fixed): 2–5 % | — | shift current exact; SHG error bar 1–10 % until the U(N)-covariant treatment (plan C2) |
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
| GaAs | QE example10 (PAW, 60/600 Ry, 16³ nscf, nbnd 50; symmetry-reduced, and repeated with `nosym`/`noinv` on the full grid) | 40³ and 60³ (40³ on the nosym nscf; 40³ with the augmented projections, pthr 0.90) | 0.05 | 0.86 eV (PBE gap 0.534 → 1.39) | 62 PAO orbitals (internal BASIS); 18 bands > 0.95 with the old projections, 12 (21 > 0.90) with the augmented ones | 15 min at 40³, 48 min at 60³, 8 ranks |
| GaAs, norm-conserving | PseudoDojo nc-sr-05 Ga/As (As 3d in valence, nelec 28), 90/360 Ry, symmetric 16³ nscf, nbnd 50 | 40³ | 0.05 | 0.86 eV (PBE gap 0.519 → 1.38) | same 62 orbitals, 18 bands > 0.95 (14 valence + 4 conduction) | 14 min, 8 ranks |
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

*Fig. 2. (a) |χ_xyz|, Re χ_xyz, Im χ_xyz of GaAs (PAW, augmented projections, 40³, scissor
0.86 eV, η 0.05 eV) with χ_yzx on top of it, |χ_xyz| of the norm-conserving run (40³), the
measured low-frequency values (166–180 pm/V at 0.12 eV) and the static range of a TDDFT +
scissor calculation; (b) the shift current σ_xyz, σ_yzx and the forbidden σ_xxx of the PAW
run, and σ_xyz of the norm-conserving run.*

| photon energy (eV) | run | 0 | 1.0 | 1.5 | 2.0 | 3.0 | peak |
|---|---|---|---|---|---|---|---|
| \|χ_xyz\| (pm/V) | 40³ | 110 | 152 | 341 | 254 | 348 | 610 at 2.63 |
| \|χ_xyz\| (pm/V) | 60³ | 110 | 180 | 337 | 305 | 390 | 480 at 2.61 |
| \|χ_yzx\| (pm/V) | 60³ | 111 | 184 | 340 | 306 | 391 | 482 at 2.61 |
| \|χ_xxx\| (forbidden, pm/V) | 60³ | 0.02 | 0.12 | 1.2 | 1.1 | 3.6 | 26 at 2.67 |
| σ_xyz (µA/V²) | 40³ | 0 | 0 | −0.10 | −7.9 | −16.0 | −29.6 at 3.90 |
| σ_xyz (µA/V²) | 60³ | 0 | 0 | −3.7 | −3.0 | −16.8 | −26.7 at 4.00 |
| σ_yzx (µA/V²) | 60³ | 0 | 0 | −3.1 | −2.3 | −16.8 | −26.7 at 4.00 |
| σ_xxx (forbidden, µA/V²) | 60³ | 0 | 0 | 0 | 0 | −0.02 | 0.35 at 3.21 |
| \|χ_xyz\| / \|χ_yzx\| (pm/V) | nosym 40³ | 112 / 112 | 155 / 146 | 345 / 357 | 256 / 257 | 348 / 348 | 623 / 596 at 2.63 |
| σ_xyz / σ_yzx (µA/V²) | nosym 40³ | 0 | 0 | −0.10 / −0.10 | −6.7 / −1.6 | −16.1 / −16.5 | −29.6 / −29.6 at 3.90 |

Files: `Workspace/PAOFLOW/validation/nonlinear_optics_GaAs/paoflow/output_nk40_tol10meV/`,
`paoflow/output_nk60/` and `paoflow_nosym/output_nk40/` (the same 62-orbital basis on the
no-symmetry nscf). These three runs were made with the projections before the fix described
below; the corrected PAW and norm-conserving runs follow.

**Static susceptibility.** The lowest-energy |χ_xyz| = 110 pm/V is below the measured
166–180 pm/V (0.12 eV) and the 207–245 pm/V of a TDDFT + scissor calculation (static limit,
scissor 0.7–0.9 eV). Independent-particle LDA/PBE + scissor values in the literature fall in the
same 100–250 pm/V bracket; the result is at its low end. The static value is converged in k (110 pm/V at
both 40³ and 60³); what is missing are the local-field and excitonic effects. The scissor itself lowers χ
(the 1/ω factors), as in all scissor calculations.

**Spectra.** |χ_xyz| rises through the two-photon onset at half the gap (0.7 eV), reaches
337–341 pm/V at 1.5 eV (converged) and peaks near 2.6 eV; the peak height is not converged
(610 pm/V at 40³, 480 pm/V at 60³: the resonance region between 2.3 and 3.5 eV is a dense set of
sharp structures that η = 0.05 eV does not smooth at these grids), and the E1 region (3 eV)
gives 348 → 390 pm/V. Published spectra (Rashkeev 1998, Nastos 2005, Wang 2017 Fig. 4, the
measurements of Bergfeld and Daum 2003) show |χ| of several hundred pm/V between 1.5 and 3 eV
with the strongest structure near the E1 transitions; the shape is reproduced at the level of
the broadening. Quoting the resonance region needs 100³ or a larger η.

**Shift current.** σ_xyz starts at the scissored gap (1.4 eV), stays within a few µA/V² up to
2.5 eV and reaches −27 (60³) to −30 (40³) µA/V² at 3.9–4 eV; the order of magnitude agrees with
the LDA + scissor calculations of Nastos and Sipe (2006) and Ibañez-Azpiroz *et al.* (2018),
which also place the largest values at 3–4 eV. Between 40³ and 60³ the curve above 3 eV moves by
≈ 10 % and the onset region (1.4–2.5 eV, small values, few k-points) changes shape; the
−8 µA/V² dip of the 40³ run at 2.0 eV (absent in σ_yzx) is a sampling artifact that the 60³
run removes. The sign convention (J = σ E E*) is documented in the docs page.

**Symmetry audit.** At 40³ χ_yzx tracks χ_xyz within 5 % between 1 and 2 eV and 7 % above;
at 60³ within 2 % and 4 %. The forbidden χ_xxx is 0.02 % of χ_xyz at 0 eV but 11 % (40³) / 5 %
(60³) of the peak at the 2.6 eV resonance; at 60³ σ_yzx − σ_xyz is 2.5 % of the peak and σ_xxx
1.3 %; the T-odd shift and injection files and the circular injection are 1e-5 of their natural
scale. **A no-symmetry nscf does not remove this:** the same basis on the `nosym`/`noinv` 16³
nscf (4096 k) gives 6 % / 26 % for χ_yzx/χ_xyz, 9 % for χ_xxx and a 17 % spread between σ_yzx
and σ_xyz at 40³ (table above).

The cause was traced with the `diag_*.py` scripts of the validation folder. The PAO Hamiltonian
obeys the C2z relation H(gk) = D H(k) D† at the grid points to 1e-14 at general k but fails by up
to 0.65 eV at 16 of the 4096 points (all on the k_z = 0 plane, little group of order 4, in the
p–d and d–d blocks), so its real-space images are symmetric only to 0.8 meV and the eigenvalues
interpolated at an off-grid k spread by 5–25 meV over the 24 T_d images, which the 1/ω factors
amplify. The Wigner–Seitz images and weights are symmetric; the band selection (the energy shift,
tested with all 18 bands kept and with the isolated groups of 13 and 10 bands) and a Löwdin
construction of H(k) change nothing; the QE eigenvalues are symmetric to 4e-10. The projected
valence density matrix (gauge invariant) built from the *internal projections* violates the
antiunitary C2z·T symmetry by 1–2e-2 at the Δ-line points, whereas the one built from the
`projwfc.x` projections obeys it to 4e-11. The raw pseudo-wavefunction projections obey it to
1e-10: the violation appears in the Löwdin re-orthonormalisation of the 50 PAW
pseudo-wavefunctions that the internal projections apply (their plain overlaps deviate from
orthonormality by up to 0.78 for the Ga 3d states, and the truncated set is not symmetric where
the 50-band cut crosses a multiplet). **The fix (same day).** The internal projections now take the augmented metric
S = 1 + Σ q_ij |β_i⟩⟨β_j| of ultrasoft / PAW pseudopotentials (`projection/augmentation.py`:
projections ⟨φ̃|S|ψ⟩, atomic orbitals Löwdin-orthonormalised with ⟨φ|S|φ⟩, the bands used as
stored; `docs/internals/internal-basis.md`). On the same PAW GaAs save ⟨ψ|S|ψ⟩ − 1 = 3e-6, the
antiunitary error of the projections drops from 2e-2 to 2e-10, H(k) is symmetric on the grid,
the eigenvalue spread over the 24 images of an off-grid k is 3e-3 meV (was 5–25) and the
real-space Hamiltonian is 20 times better localised; on the UPF pseudo-atomic basis the
projectabilities agree with `projwfc.x` to 1.5e-3. The projectabilities of the all-electron
62-orbital basis change (they were inflated by the re-orthonormalisation): the Ga 3d bands
project at 0.91 and 12 instead of 18 bands pass 0.95, so the PAW rerun below uses
`pthr = 0.90` (21 bands kept). The norm-conserving rerun (PseudoDojo nc-sr-05 Ga/As with the As 3d in
valence, 90 Ry, symmetric 16³ nscf, same 62-orbital basis) confirms the diagnosis: the
antiunitary error of the internal projections drops from 2e-2 to 6e-11, the C2z error of H(k)
on the grid from 0.65 eV to 3e-9 eV at every point, the eigenvalue spread over the 24 images of
an off-grid k from 5–25 meV to 3e-8 meV, and the real-space Hamiltonian beyond 40 bohr is
15 times smaller (median 1.9e-4 instead of 2.9e-3 eV), i.e. the PAW asymmetry was also the
reason for the slow real-space decay.

**Norm-conserving GaAs, 40³** (`paoflow_nc/output_nk40`; nelec 28 with the As 3d in valence,
14 occupied bands, PBE gap 0.519 eV, scissor 0.86 eV, η 0.05 eV, 14 min on 8 ranks):

| photon energy (eV) | 0 | 1.0 | 1.5 | 2.0 | 3.0 | peak |
|---|---|---|---|---|---|---|
| \|χ_xyz\| (pm/V) | 103 | 143 | 320 | 242 | 206 | 492 at 2.54 |
| \|χ_yzx\| (pm/V) | 103 | 142 | 294 | 241 | 207 | 492 at 2.54 |
| \|χ_xxx\| (forbidden, pm/V) | 0.004 | 0.6 | 0.8 | 0.2 | 0.1 | 6 at 1.25 |
| σ_xyz = σ_yzx (µA/V²) | 0 | 0 | −0.16 | −2.3 | −14.4 | −28.0 at 3.88 |
| σ_xxx (forbidden, µA/V²) | < 1e-9 | | | | | |

The shift current is now symmetric to 1e-10 (σ_yzx − σ_xyz and σ_xxx below 1e-10 of the peak,
the T-odd files 1e-10 as well), the static χ_xyz (103 pm/V) and the shape of the spectra agree
with the PAW run, and the forbidden χ_xxx is 1.2 % of the peak. **A residual χ_yzx/χ_xyz
difference of up to 10 % (1–2 eV) and 2 % (2–4 eV) survives in the SHG only**, and so does a
non-zero circular injection current (which T_d forbids): both come from the k-points on the
Λ lines of the grid, where bands are exactly degenerate and the eigenvector gauge returned by
`eigh` is arbitrary. The k-resolved test (`paoflow_nc/diag_cov2.py`) shows the shift-current
kernel covariant to 1e-11 at every tested k, the SHG histograms covariant to 1e-10 at general,
Δ and Σ points but different by 2.5 % and 18 % at a Λ point, and the injection kernel
gauge-dependent there. This is the U(N)-covariant treatment of degenerate bands of the phase-2
plan (C2), not a Hamiltonian problem; its weight decreases with the grid density (170 % at
8³, 10 % at 40³ in the 1–2 eV window).

**PAW GaAs with the augmented projections, 40³** (`paoflow/output_nk40_aug`; example10
inputs, `pthr = 0.90` → 21 bands, scissor 0.86 eV, η 0.05 eV; this is what the example in
this folder now computes):

| photon energy (eV) | 0 | 1.0 | 1.5 | 2.0 | 3.0 | peak |
|---|---|---|---|---|---|---|
| \|χ_xyz\| (pm/V) | 120 | 176 | 416 | 301 | 397 | 725 at 2.63 |
| \|χ_yzx\| (pm/V) | 120 | 174 | 417 | 302 | 397 | 725 at 2.63 |
| \|χ_xxx\| (forbidden, pm/V) | 1e-4 | 0.01 | 0.001 | 0.002 | 0.001 | 0.04 |
| σ_xyz = σ_yzx (µA/V²) | 0 | 0 | −0.08 | −3.6 | −17.7 | −32.2 at 3.90 |
| σ_xxx (forbidden, µA/V²) | < 1e-7 | | | | | |

The shift current obeys the T_d relations to 3e-8 of the peak, the T-odd files are 1e-9,
χ_yzx tracks χ_xyz within 1.7 % (1–2 eV) and 1 % (2–4 eV) and the forbidden χ_xxx is 6e-5 of
the peak: with only 9 valence bands (no semicore in the PAW valence) the degenerate-gauge
error of the SHG is five times smaller than in the norm-conserving run. Between the three
Hamiltonians (first PAW runs, PAW with the augmented projections and 21 bands, norm-conserving
with the As 3d in valence) the static χ_xyz spans 103–120 pm/V, the 1.5 eV value 320–416 pm/V,
the peak 490–725 pm/V and σ_xyz(3.9 eV) 27–32 µA/V²: the resonance region depends on the
pseudopotential and on the kept-band set as much as on the k-grid, while the static value and
the onset are stable to 15 %. Symmetrising the tensors (phase-2 plan C3)
would hide the error bar, not remove it. With the former 0.1 meV degeneracy tolerance the
leakage was 33 % and the shift current had a spurious −218 µA/V² peak at 1.61 eV: that is why
the default is now 10 meV.

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

- **GaAs k-grid.** The static χ_xyz (110 pm/V), the value at 1.5 eV and the shift current above
  3 eV are converged between 40³ and 60³ (≤ 10 %); the resonance region 2.3–3.5 eV is not (peak
  610 → 480 pm/V) and needs 100³ or a larger η before it is quoted.
- **GaAs symmetry.** The 2–5 % spread between equivalent components of the first runs came
  from the internal projections of the PAW pseudo-wavefunctions (§3.2), not from the symmetry
  expansion; the augmented metric implemented the same day removes it (Hamiltonian symmetric
  to 1e-9 eV, shift current symmetric to 1e-10). What survives is the gauge of exactly
  degenerate bands on the Λ lines in the SHG and injection kernels (10 % / 2 % between
  equivalent χ components at 40³): the U(N)-covariant degenerate treatment of the phase-2 plan
  (C2).
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
