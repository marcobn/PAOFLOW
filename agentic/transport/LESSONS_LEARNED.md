# Lessons learned (Si mobility and Pb resistivity, October 2026)

Read together with `AGENT_WORKFLOW.md`. Function names refer to `PAOFLOW.agentic.transport` and the shared toolkit `PAOFLOW.agentic.thermo_optics`. The ingestion lessons of `agentic/thermo_optics/LESSONS_LEARNED.md` (replots, vector vs raster, axes, markers, tracers) apply unchanged.

Each entry: **symptom → cause → what to do**, with the instance.

## Observables and samples

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | Model mobility 10–30 % below a "measured" one | The measurement is a Hall mobility (μ_H = r_H μ_drift, r_H ≈ 1.0–1.3 for Si holes) | Compare drift with drift; otherwise state the Hall factor and its source in the header and the caveat box | Si holes: the drift values 470–500 cm²/Vs, Hall values higher. |
| 2 | Model mobility far above the experiment at low T, or at high doping | The model is phonon-limited; the sample has ionised impurities | Treat the model as the upper bound; add a Matthiessen impurity mobility (`bl.matthiessen_mobility`, `ch.impurity_channel`) as a *diagnostic* with the doping stated; compare in the phonon-limited window | Si: above ~100 K and n ≲ 10¹⁴ cm⁻³. |
| 3 | σ(T) or Seebeck off by orders of magnitude while μ agrees | The model carrier density is intrinsic (or the `--doping` of the run); the sample is doped | Compare μ, or rerun with `--doping` matched to the sample; never compare σ at different n | `mobility.dat` columns n_e, n_h. |
| 4 | A "quadrupole" series equals the dipole-only one | A copied driver kept `QUADRUPOLES = None` (or the system.py default) while the folder name said otherwise | Read the flags in the driver log / `carrier_scattering.npz`, not the folder name; proposed runs are exact commands with `--quadrupoles` / `--no-long-range` and a `--tag` | Si `Si_dfpt_quad`: the series had to be postponed. |
| 5 | ρ(T) of the model below the experiment by a constant | The sample's residual resistivity ρ₀ (impurities, defects) | Subtract the stated ρ₀ or add it to the model (`resistivity(T, rho0_uohm_cm=)`); quote `bl.matthiessen_offset` (rms before/after, slope ratio) as the diagnostic; compare the slope dρ/dT | Pb: 16.2 vs ≈ 21 µΩ cm at 300 K is *not* a residual (the slope differs too). |
| 6 | Pb ρ(300 K) 25 % low, λ 1.2 vs 1.55 measured | Scalar-relativistic PBE coupling; spin-orbit raises λ in Pb; q-grid moves ρ by 6 % | Say so; do not scale; propose the FR run and the q convergence as runs with cost | `examples/elphon/Pb/README.md` §5, the FR campaign on fedora. |
| 7 | Two Boltzmann runs of the same channel differ by 15 % | One used the former non-local velocity term (removed October 2026), which changed the band velocities | Record the code version (commit) of every run in the inventory; rerun old runs before comparing | Pb 19.3 vs 16.2 µΩ cm at 300 K. |
| 8 | Emissivity totals too low at low T | The Planck integral is truncated by the photon-energy grid (`planck_coverage` < 1) | Check `planck_coverage` at each T; propose a lower `--omega-min` | Pb: coverage 0.75 at 10 K, 0.95 at 20 K, > 0.99 above 50 K. |

## Convergence and model choices

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 9 | Hole mobility drifts with the dense grid, electrons do not | The warped heavy-hole bands need finer k/q; the smearing floor interacts with the grid | Quote μ with the convergence series; converge holes separately; adaptive smearing | Si: μ_h 495 → 517 → 541 cm²/Vs at 60³ → 72³ → 84³; μ_e within 2 %. |
| 10 | λ (and λ_tr) scatter ±8 % between k-grids at fixed smearing | 50 meV is far below the level spacing \|v_F\| dk of the grid | Use `--adaptive-smearing` (flat from 24³ to 72³ for Pb); quote both | Pb table in `examples/elphon/Pb/README.md`. |
| 11 | ω_p off by 20 % | The Fermi-surface plasma frequency needs ≥ 48³ k | Check `describe()` for the k-grid; ω_p does not depend on q | Pb: 6.9 eV at 24³, 8.3 eV at 48³. |
| 12 | SERTA and MRTA differ by a few % for electrons, more for holes | Momentum relaxation weights (1 − cos θ) | Report both; MRTA is the one to compare with drift mobilities | Si: 1283 / 1302 (e), 495 / 492 (h) at 60³. |
| 13 | Power-law exponent of μ(T) depends on the window | μ ∝ T⁻ᵖ holds only over a limited range; intervalley scattering changes p | Fit experiment and model over the *same* window and say which | Si electrons: p ≈ 2.4 between 200 and 400 K. |
| 14 | Model values needed at temperatures not in the run | Coarse temperature grid | Interpolate with the physical law (power law for μ, activated for n, linear for ρ above θ_D/5, power below) and label it; propose the missing temperatures as a run | `MobilityModel.mobility(T, interpolate=True)`. |
| 15 | Mobility changes by 10 % when the long-range model is switched | Polar (Fröhlich, quadrupole) corrections to the coupling; dipole-only vs quadrupoles | Record `long_range` / `quadrupoles` from `describe()`; non-polar Si: the dipole term is tiny (Z* ≈ 0), quadrupoles matter | SiC (polar) vs Si. |
| 16 | Screening flag changes μ by 10–14 % | Poncé's dielectric rescaling of g (`--screening-eps`) | State the ε used (measured ε is the target) and quote with/without | Si: −9.9 / −14.2 % (e / h). |

## Comparison

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 17 | A single global rms hides a crossing of the curves | Different exponents: the curves cross inside the window | Split the window; report the exponents and the crossing temperature | μ_h(T) of Si against the measured p ≈ 2.2–2.4. |
| 18 | Baseline rms = 0 or meaningless | A constant baseline on a flat curve; a Bloch-Grüneisen fit with free ρ₀, A and θ to four points | Void it; use the published calculation or the inter-experiment spread | Pb ρ(T) with few points. |
| 19 | Bloch-Grüneisen fit returns θ at the scan edge | Too few low-T points to fix θ; ρ(T) linear in the data | Fix θ to the literature Debye temperature (`theta_k=`) and fit ρ₀ and A only | Pb: θ_BG ≈ 100 K. |
