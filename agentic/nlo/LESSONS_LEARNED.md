# Lessons learned (GaAs, GeS and gapped graphene, October 2026)

Read together with `AGENT_WORKFLOW.md`. The ingestion lessons of `agentic/thermo_optics/LESSONS_LEARNED.md` apply unchanged. Each entry: **symptom → cause → what to do**, with the instance.

## The model

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | A forbidden component is 30 % of the allowed ones; a huge shift-current peak just above the gap | Nearly degenerate PAO pairs (semicore and ghost bands) with the 0.1 meV degeneracy tolerance: their 1/ω factors amplify meV-level numerical asymmetries | `degeneracy_tol` = 10 meV (the default since October 2026); check that the allowed components do not change | GaAs: σ_xxx/σ_xyz 0.33 → 0.008; the spurious −218 µA/V² peak at 1.61 eV disappeared. |
| 2 | Equivalent components differ by 5–10 % (χ_xyz vs χ_yzx) | The PAO Hamiltonian expanded from a symmetry-reduced nscf is symmetric only to ~1 meV in real space; the interpolated eigenvalues spread by 10 meV over the point-group images of an off-grid k | Run the nscf with `nosym = .true., noinv = .true.`; otherwise quote the spread as the symmetry error bar | GaAs 16³ symmetric nscf: 5 % at 1–2 eV, 11 % for χ_xxx at the 2.5 eV resonance. |
| 3 | No gap in the PAO Hamiltonian; responses start at 0.3 eV | The basis tables do not represent the valence bands (projectability 0.90 for the Ge 3d bands with the repository BASIS and PseudoDojo pseudopotentials): only 14 of 20 valence bands above `pthr` | Generate `BASIS_PS` from the pseudopotentials (`paoflow-genbasis-ps`, preset `standard`); require all valence bands and a few conduction bands above 0.95 | GeS: 26 bands above 0.95 with BASIS_PS, gap 1.68 eV recovered. |
| 4 | A small basis gives projectability 0.45 | Semicore states need their own shells in the basis | Use the configuration that projects the semicore bands (GaAs: the 62-orbital example10 set) | GaAs 26-orbital basis: unusable. |
| 5 | A tight-binding model gives the wrong selection rules | The model's hopping cells were inconsistent with its atom positions (bonds of 1.5 alat) | The position operator needs cells consistent with `tau`; the hard-coded graphene model was fixed (bands unchanged) | `models.graphene`. |
| 6 | The run died with 5.7 GB per rank | Pair kernels and three-band terms held for a whole k-block | Chunked since October 2026 (16 MB kernels); `kblock` 64 for nawf > 60 | GaAs 40³ before chunking. |
| 7 | QE Davidson aborts on the full-grid nscf ("S matrix not positive definite") | PAW + many bands at some k | `diagonalization = 'cg'` (ppcg was removed in QE 7.5); 2 pools instead of 8 | GaAs nosym 16³. |
| 8 | A 2D vc-relax lands on a cell 8 % too small | The cell relaxation of a free-standing polar monolayer with `cell_dofree='2Dxy'` found another minimum | Fix the in-plane cell at the literature value and relax the positions | GeS: a = 4.40, b = 3.68 Å; vc-relax gave 4.06/3.53. |
| 9 | projwfc was run | Copied from an older example | Not needed: the projections are internal | GaAs, GeS validation. |

## Conventions and comparison

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 10 | Model χ twice the quoted "d" | d_ijk = χ_ijk/2 | `NonlinearModel.d_coefficient`; record which one the source quotes | GaAs d_14 ≈ 85–90 pm/V ↔ χ_xyz ≈ 170 pm/V. |
| 11 | Experimental static values scatter by 10 % | Measured relative to reference crystals (quartz, KDP) whose accepted values were revised | Quote the range with the sources (166, 172, 180 pm/V at 0.12 eV for GaAs) | Levine-Bethea, Roberts, Eyres. |
| 12 | 2D literature values differ by a factor of several | Sheet responses divided by different layer thicknesses, or quoted per unit cell height | State the thickness convention; `effective_3d(kind, abc, thickness_nm)`; compare the ratio between components as well | GeS: 46 nm µA/V² ≈ 75 µA/V² for 0.6 nm. |
| 13 | The onset of |χ| is at half the gap | Two-photon resonance ω = E_g/2 | Expected; use it as an anchor of the gap | Gapped graphene: 0.5 eV for a 1 eV gap. |
| 14 | Spectra noisy with 0.05 eV broadening | The 40³ grid (GaAs) is not converged for sharp features | Compare peak positions and integrated weights, show 40³ and 60³, raise η for the figure | GaAs Re/Im χ. |
| 15 | The model has no sub-gap SHG resonances | Excitons and local fields are outside the independent-particle response | Say so; compare above the gap and at the static limit | 2D crystals. |
