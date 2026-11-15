# Lessons learned (Pb, MgB₂ and Si ARPES examples, October 2026)

Read together with `AGENT_WORKFLOW.md`. Function names refer to `PAOFLOW.agentic.arpes` and the shared toolkit `PAOFLOW.agentic.thermo_optics`. The ingestion lessons of `agentic/thermo_optics/LESSONS_LEARNED.md` apply unchanged.

Each entry: **symptom → cause → what to do**, with the instance.

## Alignment and extraction

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 1 | λ from the same data ranges from 1.0 to 1.6 | λ depends on the bare band: a linear fit far from E_F, a DFT band, a parabola through k_F give different v_bare | Use the same protocol on both sides (`cv.linear_band` over the authors' window on the model's dressed dispersion too); quote λ by method (slope of Re Σ, velocity ratio, dΓ/dT); the model's exact bare band (`bare_dispersion`) is extra information, not the comparison | Pb: slope λ 1.11 vs the run's λ 1.03. |
| 2 | Dispersions offset in k by 0.01–0.05 1/Å | Different k_z (photon energy), surface relaxation, the model path's origin | Compare versus k − k_F with each side's own k_F (`fermi_momentum`); never shift by hand | Pb Γ–X segment at 0.575 of the line. |
| 3 | The Fan-Migdal Σ of a metal shows no kink | The level spacing \|v_F\| Δq of the dense q-grid (≈ 0.3 eV at 18³) exceeds the phonon energies; fixed widths give noise, adaptive widths smear | Resolve the kink with the isotropic (or anisotropic) Σ from α²F; use the Fan-Migdal form for linewidths and wide windows; propose a denser q-grid only with its cost | Pb: Im Σ ≈ −7.3, Re Σ ≈ −6.5 meV flat at 18³ q (plateau right, structure absent). |
| 4 | Measured widths larger than the model's by a constant | Momentum resolution (MDC) or the Fermi cutoff and energy resolution (EDC); impurity scattering Im Σ(0) | Compare MDC widths at the same resolution (`ARPES['resolution']`), `method='lorentzian'`; subtract and state the impurity offset Im Σ(0) | Pb 2 meV resolution in the example. |
| 5 | Relative intensities of bands disagree by orders of magnitude | Photoemission matrix elements depend on photon energy and polarisation; the model's PAO matrix elements are approximate | Compare positions and widths, never absolute intensities; normalise per MDC; use `intensity_power` only for display | Si at He I: the s-like band 30× the p-like ones. |
| 6 | Kink energy off by a few meV | E_F referencing (Au reference vs the PAO Fermi level); the energy grid of Σ | `mt.best_shift` in E as a diagnostic; quote the kink against the α²F peaks | Pb Re Σ extrema at ±7 meV, phonons ≈ 11 meV. |
| 7 | Model bands do not shift with temperature | Fan term only: no Debye-Waller term, so no zero-point or thermal band-edge shifts | Compare linewidths versus T, not band positions versus T; say so | Si band edges. |
| 8 | Smallest linewidths all equal | The Lorentzian width η (floor) of the Fan-Migdal run | Values below ≈ η are the floor, not physics; propose a smaller η with a denser q-grid | Si: 20 meV η; VBM 1.3 meV at 10 K is below the floor. |
| 9 | Γ(T) not linear | Below Θ_D the phonon occupation is not classical; above it Γ = 2πλ_nk k_B T + Γ₀ | Fit the slope above Θ_D only; λ_nk from the slope is the robust number | Si valence states 300–600 K. |
| 10 | The MDC peak method gives widths 10 % too large | Fewer than five path points per MDC width (grid-limited half-maximum) | `method='lorentzian'`, or rerun with `--npts` larger; the position is fine either way | `curves.half_maximum_width` docstring. |
| 11 | Re Σ has the opposite sign in the paper | Convention: ω = ε_b + Re Σ(ω) here; some authors plot ε_b − ω or the "kink" Re Σ with the opposite sign | Check the kink direction (the dressed band is flatter near E_F); state the convention | — |
| 12 | Pb bands near E_F differ from the paper's DFT | Scalar-relativistic PBE; spin-orbit coupling changes the Pb Fermi surface and raises λ | Note it; the FR run is a proposed run with cost | Pb SR vs FR campaign. |
| 13 | Dispersion extracted from a colour image has spurious peaks | Non-monotonic colour maps (jet) are not monotonic in brightness | `band_map_from_image(channel=...)` with a sequential channel, or calibrate against the scale bar; state the assumption; validate against the authors' fits where available | `images.band_map_from_image`. |
| 14 | The σ-band kink of a two-gap metal is underestimated | The isotropic Σ averages the σ and π couplings | Use `--self-energy anisotropic` (needs `fs_coupling.npz`) and the state nearest the crossing | MgB₂: λ_nk 0.33 (π) to 0.99 (σ). |
| 15 | The segment does not contain the kink | `--segment-ev` too small, or the full path too coarse for MDCs | Cover at least twice the kink energy; `--npts` for ≥ 5 points per width | Pb `segment_ev` in `system.py`. |

## Comparison

| # | Symptom | Cause | What to do | Instance |
|---|---|---|---|---|
| 16 | A single rms over the dispersion hides the kink | The deviation is concentrated within the phonon energies | Split windows: below the kink / above; report the velocity ratio per window | — |
| 17 | The "experimental Σ" agrees with everything | It was extracted with a bare band fitted to the same data | Treat an extracted Σ as derived; compare the dispersion and the widths as well | — |
| 18 | Baseline from a tunnelling α²F needs a temperature | `eliashberg_self_energy(omega, a2F, w, T)` takes T; the measurement temperature | Evaluate at the measured T; state the α²F source (McMillan-Rowell) | Pb tunnelling α²F. |
