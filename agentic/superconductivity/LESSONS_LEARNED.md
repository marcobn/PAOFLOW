# Lessons learned (Pb and MgB₂ examples, EPW comparisons, October 2026)

Read together with `AGENT_WORKFLOW.md`. Function names refer to `PAOFLOW.agentic.superconductivity` and the shared toolkit `PAOFLOW.agentic.thermo_optics`. The ingestion lessons of `agentic/thermo_optics/LESSONS_LEARNED.md` apply unchanged.

Each entry: **symptom → cause → what to do**, with the instance.

## Definitions and parameters

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | Tc "agrees" after μ* was changed | μ* is the one free parameter; Tc moves by 10–20 % between 0.10 and 0.13 | Quote Tc at the run's μ* and over 0.10–0.13 (`tc(mu_star)`); a μ* tuned to Tc is a diagnostic, stated | Pb, MgB₂ examples at μ* = 0.1. |
| 2 | Two "Tc" of the same run differ | Linearised kernel versus Δ²(T) → 0 extrapolation; McMillan versus Allen–Dynes | Quote the definition with every Tc; the kernel and Δ² values agree to 0.02 K when the temperature grid brackets Tc | Pb: 5.69 vs 5.71 K. |
| 3 | The model gap is 2 % smaller than "the gap" in the paper | Δ(iω₀) (order parameter) versus the real-axis gap edge (Padé / analytic continuation) versus a coherence-peak position | Match the definition (`gaps('imag'|'pade'|'acon')`); Padé and acon agree to 10⁻³ meV | Pb: 1.016 vs 1.037 meV at 0.3 K. |
| 4 | Measured λ 1.55 vs model 1.2 for Pb | Scalar-relativistic PBE; spin-orbit raises λ (Heid *et al.* 2010); the tunnelling inversion has its own μ* | Say so; the FR run is a proposed run with cost; compare α²F shapes, not only λ | `examples/elphon/Pb/README.md`. |
| 5 | A specific-heat λ disagrees with the e-ph λ | 1 + λ from γ needs the band N_F of the same calculation; spin fluctuations and SOC enter γ | Use the calculation's N_F (`fs_coupling.npz`, `dos_ef` of the anisotropic run) and state it | Pb N_F 0.251 (PAO) vs 0.235 (Wannier) /eV/spin. |
| 6 | The gap ratio exceeds BCS | Strong coupling: 2Δ/k_BTc = 3.53 [1 + 12.5 (Tc/ω_log)² ln(ω_log/2Tc)] | Quote the strong-coupling ratio as the baseline, not 3.53 | Pb ≈ 4.3–4.5. |

## Convergence and method

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 7 | λ scatters ±8 % between dense k-grids | 50 meV fixed smearing is far below the level spacing \|v_F\| dk | `--adaptive-smearing` (flat from 24³ to 72³ for Pb); quote both | Pb table in `examples/elphon/Pb/README.md` §5. |
| 8 | λ follows N_F | N_F at σ = 0.05 eV needs ≥ 40³ k (MgB₂: 18 % high at 24³, 8 % low at 32³, settled from 48³) | Converge N_F first; quote N_F with λ | MgB₂ "Reference results". |
| 9 | The anisotropic solution has 75 % of the Fermi surface losing its gap at 30 K while the rest keeps it to 41 K | Unequal dense grids (k 40³ / q 20³): k+q reaches a sublattice, the equations split | Equal grids (`--nk N --nq N`); the drivers warn | MgB₂ tutorial grids. |
| 10 | Anisotropic gaps and Tc change by 30 % between 24³ and 32³ | The σ sheets are small; not converged in k | ≥ 40³ equal grids (2.5 h on 20 cores) or a larger smearing; say "not converged" | MgB₂: 1.9 … 4.9 (24³) vs 1.1 … 4.9 meV (32³). |
| 11 | λ 1.6 from a cheap run | `--coarse-q` is not converged in q | Never compare coarse-q values; propose the dense run | Pb. |
| 12 | λ drops from 1.20 to 1.14 with the q-grid | Dense q 24³ → 48³ | Quote the q-convergence with λ_tr and ρ (6 %) | Pb. |
| 13 | ME run has no Tc from Δ² → 0 | The temperature grid does not bracket Tc | Set the temperatures of `me.py` around the Allen–Dynes Tc; check `converged` | `GapModel.tc`, `describe()`. |
| 14 | The two codes differ by 7 % in λ on the same inputs | The PAO (18 orbitals) and Wannier (4 functions) interpolations give different N_F; (Z₀ − 1)/λ is the same | Report the N_F with the λ; EPW on the same grids is the baseline, agreeing to 1 % when N_F agrees | Pb vs EPW; MgB₂ tutorial grids (1 %). |
| 15 | The isotropic Tc is half the measured one for a two-gap metal | The isotropic equations average the σ and π couplings | Use `me_aniso.py` (equal grids); compare the gap distribution peaks with the measured sheet gaps | MgB₂: anisotropic Tc ≈ 2× isotropic; measured 39 K. |

## Comparison

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 16 | α²F "agrees" in λ but not in shape | λ is one moment; peak positions and the cumulative λ(ω) per branch carry the comparison | `cumulative_lambda`, `lambda_in_window` per phonon window against the tunnelling α²F | Pb transverse / longitudinal peaks. |
| 17 | A tunnelling DOS does not match Δ | The conductance is the quasiparticle DOS at the measurement temperature, thermally broadened | `quasiparticle_dos(T)` at that T, broadened with the Fermi-function derivative; compare curves | — |
| 18 | Point-contact gaps of MgB₂ (2.3 / 7.1 meV) vs the mean gap | A multi-gap spectrum shows the distribution's peaks, not its mean | `gap_peaks(T)`; `sheet_gaps(T, {'sigma': [...], 'pi': [...]})` with the band assignment stated | MgB₂ bands 6, 7 (σ) and 8 (π) in the example. |
| 19 | Δ(T) "fits BCS" on both sides | The BCS curve scaled to Δ(0) and Tc is the baseline, not the model; both sides sit near it | Quote the deviation from the BCS baseline for model and experiment; the strong-coupling ratio is the discriminating number | — |
