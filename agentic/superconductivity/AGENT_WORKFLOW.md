# Superconductivity workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) comparing PAOFLOW's **electron-phonon superconductivity** with **experimental data of any form** for **any conventional superconductor**: the Eliashberg function α²F(ω) and its moments, the critical temperature (McMillan, Allen–Dynes, isotropic and anisotropic Migdal–Eliashberg), the gap Δ(T), its edges and ratio, the gap and λ distributions of multi-gap metals, the quasiparticle density of states, and the mass enhancement. The model is an electron-phonon *system folder* (`examples/elphon/<System>` layout; see "The model pipeline"). The stages are those of the thermo-optics workflow (`agentic/thermo_optics/AGENT_WORKFLOW.md`), whose ingestion recipes apply unchanged; this playbook adds what is specific to superconductivity: the observables, the definitions and parameters that must be stated (μ*, the gap definition, the temperature) and the baselines.

The tooling is `PAOFLOW.agentic.superconductivity` over the shared toolkit `PAOFLOW.agentic.thermo_optics`. A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `digitize_sources.py` (from `agentic/thermo_optics/templates/digitize_template.py`) | the sources (PDF, image, table, deck) | `data/<source>.csv`, `checks/*_overlay.png` |
| `analysis.py` | `data/*.csv` + the model output (read-only) | `data/model_curves.csv`, `tables/metrics.csv`, `tables/tc.csv`, `tables/*.csv`, `tables/summary.md` |
| `plot_figures.py` | CSV files only | `figures/*.pdf` (vector) + `*.png` (600 dpi) |

```python
from PAOFLOW.agentic.superconductivity import AnisotropicGapModel, EliashbergModel, GapModel
from PAOFLOW.agentic.superconductivity import baselines as bl
from PAOFLOW.agentic.thermo_optics import digitize as dg, ingest, metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset, read_dataset
```

Read `LESSONS_LEARNED.md` once before starting.

## The model pipeline (electron-phonon system folder)

`system.py` holds the configuration (prefix, coupling source, coarse grids, PAO basis, long-range model, cache) and the `SUPERCONDUCTIVITY` defaults (dense `nk`, `nq`, `sigma_ev`, `adaptive_smearing`, `mu_star`, `fsthick_ev`, `fs_coupling`, `transport`, the Migdal–Eliashberg temperatures). The coarse vertex is cached in `cache/`, so reruns pay only the dense step. The drivers write fixed folders (`output/`, `output/me/`, `output/me_aniso/`): rename a previous one to `<name>_<label>` before a rerun.

| driver (from the system folder; `--help` lists every flag) | output | what it gives | cost |
|---|---|---|---|
| `superconductivity/eliashberg.py [--nk --nq --sigma-ev --adaptive-smearing --mu-star --fs-coupling --transport]` | `output/eliashberg.npz` (`omega`, `a2F`, `lambda`, `omega_log`, `omega_2`, `mu_star`, `Tc_mcmillan`, `Tc_allen_dynes`, `lambda_qv`, …), `alpha2F.dat`, [`fs_coupling.npz`, `alpha2F_tr.dat`] | α²F, λ, ω_log, ω₂, McMillan / Allen–Dynes Tc; the Fermi-surface pair coupling for the anisotropic equations; α²F_tr for transport | Pb 48³/24³: 2 min on 8 cores; MgB₂ 40³/40³: 2.5 h on 20 cores |
| `superconductivity/me.py [--mu-star --a2f <EPW a2f>] (temperatures from system.py)` | `output/me/migdal_eliashberg.npz`, `gap_vs_T.dat`, `max_eigenvalue.dat`, `<prefix>.imag_iso_<T>`, `pade_iso_<T>`, `acon_iso_<T>`, `qdos_iso_<T>` | isotropic Migdal–Eliashberg: Δ(iω₀)(T), Z, the real-axis gap edges (Padé, analytic continuation), the quasiparticle DOS, Tc from the linearised kernel and from Δ²(T) → 0 | minutes |
| `superconductivity/me_aniso.py [--nk N --nq N --temps --mu-star]` | `output/me_aniso/migdal_eliashberg_aniso.npz`, `gap_vs_T_aniso.dat`, `<prefix>.lambda_FS`, `imag_aniso_<T>`, gap distributions, `qdos_<T>` | anisotropic (Fermi-surface-restricted) equations: Δ_nk(T), λ_nk, gap distributions, Tc | tens of minutes to hours |
| `superconductivity/plot_me.py` | `output/*.png` | quick-look figures | seconds |

`EliashbergModel.from_system(folder)`, `GapModel.from_system(folder, output='output/me')` and `AnisotropicGapModel.from_system(folder, output='output/me_aniso')` open these; `describe()` prints the configuration, the `SUPERCONDUCTIVITY` defaults, the scalars (λ, ω_log, ω₂, μ*, Tc's, λ_tr, ω_p), the temperatures and gaps, the convergence flags. Proposed runs are written as these commands, with their flags and cost.

---

## 0. Operating rules (always)

1. **Read-only inputs.** Never write into calculation folders or source folders.
2. **Ask before compute.** Comparing existing output needs no new run; Tc for another μ* and the BCS curves are formulas (`EliashbergModel.tc(mu_star)`, `baselines`). Dense Eliashberg and anisotropic Migdal–Eliashberg runs are proposals with cost.
3. **Provenance on every number**; a replot is not the original; a tunnelling-inverted α²F carries the inversion's own μ* and assumptions (record them in `conditions`).
4. **Uncertainty on every digitised number.**
5. **Look at every overlay.**
6. **A baseline with every metric**: a published calculation on the same inputs (EPW on the tutorial grids), the BCS curve, the Tc-inverted λ, the spread between measurements.
7. **Parameters and definitions are stated.** μ* (quote Tc for the run's value and for 0.10–0.13); the gap definition (Δ(iω₀) order parameter, Padé / analytic-continuation gap edge, coherence-peak position, BCS-fit Δ(0)); the temperature (model gaps are computed at discrete T: interpolate linearly and label, or compare at the computed ones); the Tc definition of the model (linearised kernel versus Δ²(T) → 0) and of the experiment (onset, midpoint, zero resistance).
8. **No silent fitting.** A λ inverted from the measured Tc (`bl.lambda_from_tc`) and a μ* that reproduces Tc are *diagnostics*, quoted next to the ab initio values.
9. **Compare the observable that was measured.** A tunnelling conductance is the quasiparticle DOS at that temperature (`quasiparticle_dos`), not Δ; a point-contact spectrum of a two-gap metal gives the gap distribution's peaks (`gap_peaks`), not the mean; a specific-heat γ gives 1 + λ with the band N_F of the calculation.
10. **Correct yourself explicitly** (dated notes).
11. **PAOFLOW conventions.** Inputs are Python arguments; no config files; new library code comes with fast tests and NumPy docstrings.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: model runs, sources, measurement, audience | `describe()`, `ingest.pdf_inventory` | inventory, open questions |
| 1 | Physics map: observable ↔ model quantity ↔ definition / μ* / T ↔ baseline | (reasoning, the user) | `PLAN.md` |
| 2 | Ingestion by form → canonical datasets | thermo-optics `ingest.*`, `digitize.*`, `datasets.*` | `data/*.csv`, `checks/` |
| 3 | Validation of the ingestion | overlays, replots, anchors | accepted datasets |
| 4 | Model at the experimental conditions | the three adapters, `baselines` | `data/model_curves.csv` |
| 5 | Metrics with baselines | `metrics.*`, `baselines.*` | `tables/metrics.csv`, `tables/tc.csv`, `tables/summary.md` |
| 6 | Diagnosis, proposed runs | existing output, the decision tree | ranked causes, run proposals |
| 7 | Publication figures | `plotting.*` | `figures/*.pdf`, `*.png` |
| 8 | Report | `templates/report_template.md` | `REPORT.md` |

## Stage 0: intake

1. **Model.** Open the system folder with the adapters; `describe()`. Note: the dense grids and smearing (fixed or adaptive) of α²F, N_F, μ*, the temperatures of the Migdal–Eliashberg runs (do they bracket Tc?), equal grids for the anisotropic run, convergence flags, whether `--fs-coupling` / `--transport` were used, the code version. What the model cannot contain: anharmonicity and phonon renormalisation, spin fluctuations, the Coulomb interaction beyond μ*, spin-orbit coupling unless the DFT had it, vertex corrections, the isotope effect of the lattice dynamics beyond the harmonic mass scaling.
2. **Sources.** Per file: form and the *measurement*: method (tunnelling, ARPES, STM, point contact, specific heat, μSR, resistive / magnetic Tc), sample, temperature, the gap definition, the inversion assumptions of a tunnelling α²F (μ*, the strong-coupling corrections).
3. **Audience and deliverable.**
4. Report the inventory and the questions before ingesting anything.

## Stage 1: physics map (decide what is compared)

| experimental quantity | model quantity | definition / parameter to match | baseline |
|---|---|---|---|
| Tc | `EliashbergModel.tc(mu_star, formula)`, `GapModel.tc['linear'|'gap']`, `AnisotropicGapModel.tc` | μ* (run's and 0.10–0.13); Tc definition | a published calculation (EPW on the same grids); `bl.lambda_from_tc` (the λ the measured Tc implies at the model's ω_log, μ*) |
| λ (tunnelling, specific heat 1 + λ, dHvA, ARPES) | `scalars['lambda']`, `moments()`; `AnisotropicGapModel.lambda_statistics()` | the N_F of the calculation for 1 + λ; the band for dHvA / ARPES | published λ; the Tc-inverted λ |
| α²F(ω) from tunnelling inversion | `function(omega)`, `cumulative_lambda`, `lambda_in_window` | the inversion's μ*; ω in meV | the inversion's own λ, ω_log; EPW's α²F |
| Δ(T) (tunnelling, ARPES, STM, μSR) | `GapModel.gaps(kind)` / `gap(T, kind, interpolate)`: `'imag'` (Δ(iω₀)) or `'pade'`/`'acon'` (gap edge) | the gap definition; temperatures | `bl.bcs_gap_vs_t(T, Tc, Δ(0))` scaled to the model and to the experiment; the strong-coupling ratio |
| 2Δ(0)/k_BTc | `GapModel.gap_ratio(kind, tc)` | Tc definition | BCS 3.53; `bl.strong_coupling_gap_ratio(Tc, ω_log)` |
| tunnelling conductance dI/dV(V, T) | `quasiparticle_dos(T)` (N_S/N_F) broadened with the measurement temperature | T; the normalisation | the BCS DOS |
| multi-gap spectra (point contact, ARPES per sheet) | `AnisotropicGapModel.gap_peaks(T)`, `sheet_gaps(T, {'sigma': [...], 'pi': [...]})`, `gap_distribution(T)` | T; the sheet assignment (band indices) | the isotropic gap; the anisotropic Tc |
| phonon peaks (Raman, IR, neutron) | `mode_lambda()` (ω_qν, λ_qν) and the peaks of α²F | ω in meV | the phonon dispersion of `phonons/` |

Rules: **flag out-of-scope physics** (anharmonicity, spin fluctuations, SOC, multiband Coulomb, the isotope effect beyond harmonic scaling); **confirm with the user** before ingestion.

## Stage 2: ingestion by form

Follow Stage 2 and the raster recipe of `agentic/thermo_optics/AGENT_WORKFLOW.md`. Canonical datasets (`datasets.COLUMNS`) with these conventions:

| quantity | x, x_unit | y, y_unit | conditions keys |
|---|---|---|---|
| `tc` | (one row) | `K` | `method` (resistive, magnetic, specific heat), `definition` (onset, midpoint, zero) |
| `gap` | `temperature`, `K` | `meV` | `method` (tunnelling, arpes, stm, musr, point_contact), `definition` (edge, peak, bcs_fit, order_parameter), `sheet` (sigma, pi, …) |
| `gap_ratio`, `lambda`, `omega_log` | (one row) | `1`, `1`, `meV` | `method`, `mu_star` (for inversions), `definition` |
| `a2f` | `omega`, `meV` | `1` | `method=tunnelling_inversion`, `mu_star`, `lambda_quoted` |
| `tunnelling_dos` | `energy`, `meV` | `1` (normalised conductance) | `temperature_k` (column), `method` |

Keep the printed units and convert in `analysis.py`.

## Stage 3: validation of the ingestion

Overlays; replots; anchors: Δ(0) against the quoted value, the α²F peaks against the phonon energies, Tc against the handbook value; assumptions in the header and the report caveat box.

## Stage 4: model adapter and condition alignment

- **`EliashbergModel`**: `omega_ev`, `a2f`, `scalars`, `function(omega_ev, transport)`, `cumulative_lambda`, `moments`, `lambda_in_window(min, max)`, `tc(mu_star, formula)`, `mode_lambda()`, `describe()`.
- **`GapModel`**: `temperatures`, `tc` (dict), `gaps(kind)`, `gap(T, kind, interpolate)`, `gap_zero`, `gap_ratio`, `bcs_gap(temps)`, `mass_renormalisation(T)`, `quasiparticle_dos(T)`, `real_axis(T, method)`, `max_eigenvalue()`, `describe()`.
- **`AnisotropicGapModel`**: `temperatures`, `tc`, `bands`, `weights`, `lambda_nk`, `gaps(T, kind)`, `gap_range(T)`, `gap_distribution(T, kind)`, `gap_peaks(T)`, `sheet_gaps(T, groups)`, `lambda_statistics()`, `quasiparticle_dos(T)`, `describe()`.
- **Alignment.** The same μ* on both sides where it enters (inversions, Tc); the same gap definition; temperatures matched (`mt.match_conditions`) or interpolated linearly in T and labelled; the tunnelling DOS compared at the measurement temperature; for multi-gap metals the sheets assigned by band index (state the assignment).
- **Write every model curve used** to `data/model_curves.csv` (`provenance='calculated'`, μ*, definition and interpolation in `note`).

## Stage 5: comparison metrics with baselines

| Tool | Use |
|---|---|
| `tables/tc.csv`: model Tc by formula and μ* (0.10, 0.13, the run's), measured Tc, `bl.lambda_from_tc` | the Tc comparison with its sensitivity |
| `mt.curve_deviation(T, Δ_exp, T, Δ_model, 'linear')` and against `bl.bcs_gap_vs_t` | Δ(T) deviation (meV) with the BCS baseline |
| `mt.curve_deviation(ω, α²F_exp, ω, α²F_model, 'linear', window)` per phonon window; `cumulative_lambda` at the window edges | α²F shape and the coupling per branch |
| gap ratio table: model (`gap_ratio`), measured, BCS, `bl.strong_coupling_gap_ratio` | |
| `gap_peaks` / `sheet_gaps` versus measured gaps per sheet | multi-gap metals |
| `mt.curve_deviation` of `quasiparticle_dos` versus normalised conductance | tunnelling DOS |

One row per (comparison, dataset, condition, window) in `tables/metrics.csv`, with `baseline`, `baseline_rms`, `note`.

## Stage 6: diagnosis

Cheap checks first: μ* sensitivity (`tc(mu_star)`); the convergence of λ and N_F with the dense grids and the smearing (fixed-width λ scatters ±8 % between grids; adaptive widths are flat; N_F needs ≥ 40³ k at σ = 0.05 eV for MgB₂, ω_p ≥ 48³ for Pb); the q-grid (Pb: 24³ → 48³ moves λ from 1.20 to 1.14); equal dense grids for the anisotropic equations (sublattice splitting otherwise); the temperature grid bracketing Tc; the `converged` flags; isotropic versus anisotropic gaps for multi-sheet metals; the code version. Then physics: scalar-relativistic versus spin-orbit (Pb λ 1.2 vs 1.55 measured), anharmonicity (MgB₂ E₂g), the PAO N_F versus Wannier N_F (Pb 0.251 vs 0.235 /eV/spin), the long-range model for polar metals. Then propose runs as exact driver commands with cost. See the decision tree.

## Stage 7: publication figures

As in the thermo-optics playbook. Superconductivity specifics: α²F with the tunnelling α²F as markers or a dashed gray line and λ(ω) on its own panel; Δ(T) with the BCS curve dashed; gap distributions as histograms per sheet with the measured gaps as vertical markers; Tc bars by formula and μ* against the measured value; one axis per panel.

## Stage 8: report

Follow `templates/report_template.md`: the caveat box names μ*, the gap definition, the Tc definitions, the temperatures, the grids and smearing, and the code version.

---

## Decision tree for a discrepancy

```
model - experiment discrepancy
 ├─ same definition?                 -> gap (order parameter / edge / peak), Tc (kernel / Delta^2, onset / midpoint), lambda (which N_F)
 ├─ same parameter?                  -> mu* on both sides; the inversion's assumptions for a tunnelling alpha^2F
 ├─ same temperature?                -> computed T, interpolation labelled; temperatures bracket Tc
 ├─ converged?                       -> dense k / q grids, smearing (fixed vs adaptive), N_F, equal grids (aniso), converged flags
 ├─ anisotropy?                      -> isotropic vs anisotropic gaps and Tc; sheet assignment
 ├─ ingestion error?                 -> overlay, replot, axis reading
 ├─ missing physics?                 -> SOC, anharmonicity, spin fluctuations, Coulomb beyond mu*, vertex corrections
 └─ none                             -> report unexplained, with what was excluded and which run or measurement would decide
```

## Ask the user at the start

1. The system folder(s) (read-only) and which `superconductivity/output*` runs to compare, including older ones.
2. The sources, with their figures or tables and their provenance; the gap and Tc definitions; the μ* of any inversion.
3. The sample and method: tunnelling, ARPES, STM, point contact, specific heat, μSR; purity; temperature.
4. The deliverable and its audience.
5. The compute budget: may dense Eliashberg or anisotropic Migdal–Eliashberg runs be launched?

## Worked examples (in the repository)

- **Pb** (`examples/elphon/Pb/README.md`, "Reference results"): λ 1.221, ω_log 4.50 meV, Tc 4.77 / 5.21 K (McMillan / Allen–Dynes, μ* = 0.1) at 48³/24³; EPW on the same Wannierised run 1.151, 4.44 meV, 4.38 / 4.76 K; measured Tc 7.2 K and λ ≈ 1.55 (tunnelling, McMillan and Rowell 1965); isotropic ME Δ(iω₀) 1.016 meV, gap edge 1.037 meV at 0.3 K, Tc 5.69 K (kernel) / 5.71 K (Δ² → 0); (Z₀ − 1)/λ = 0.925 in both codes; the difference from EPW follows N_F (0.251 vs 0.235 /eV/spin).
- **MgB₂** (`examples/elphon/MgB2/README.md`, "Reference results"): λ 0.578, ω_log 62.3 meV, Tc 14.7 / 15.1 K on the tutorial grids (EPW 0.573, 61.7 meV, 14.2 / 14.6 K); anisotropic gaps 1.4 … 7.7 meV at 5 K, two peaks (π ≈ 2, σ ≈ 5–8 meV), anisotropic Tc about twice the isotropic one (measured 39 K; gaps ≈ 2 and 7 meV, Choi *et al.* 2002); equal dense grids required; N_F converged from 48³ k.
