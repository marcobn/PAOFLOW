# Transport workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) comparing PAOFLOW's **electron-phonon transport** with **experimental data of any form** for **any material**: the phonon-limited carrier transport of semiconductors (drift mobilities of electrons and holes in the SERTA and MRTA approximations, carrier densities, conductivity, Seebeck coefficient) and the transport of metals (resistivity from α²F_tr with residual and electron-electron channels, λ_tr, the Fermi-surface plasma frequency, the extended-Drude infrared optics, the thermal emissivity), plus the Boltzmann bridge that adds impurity channels to PAOFLOW's Boltzmann transport. The model is an electron-phonon *system folder* (`examples/elphon/<System>` layout; see "The model pipeline" below). The stages are those of the thermo-optics workflow (`agentic/thermo_optics/AGENT_WORKFLOW.md`), whose ingestion recipes (tables, text, vector and raster figures, decks) apply unchanged; this playbook adds what is specific to transport: the observables, the conditions that must be matched (doping, Hall factor, residual resistivity, temperature window) and the baselines.

The tooling is `PAOFLOW.agentic.transport` (adapters, baselines, channels) over the shared toolkit `PAOFLOW.agentic.thermo_optics` (datasets, units, ingest, digitize, metrics, plotting). A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `digitize_sources.py` (from `agentic/thermo_optics/templates/digitize_template.py`) | the sources (PDF, image, table, deck) | `data/<source>.csv` (canonical, provenance header), `checks/*_overlay.png` |
| `analysis.py` | `data/*.csv` + the model output (read-only) | `data/model_curves.csv`, `tables/metrics.csv`, `tables/*.csv`, `tables/summary.md` |
| `plot_figures.py` | CSV files only | `figures/*.pdf` (vector) + `*.png` (600 dpi) |

```python
from PAOFLOW.agentic.transport import MobilityModel, ResistivityModel
from PAOFLOW.agentic.transport import baselines as bl, channels as ch
from PAOFLOW.agentic.thermo_optics import digitize as dg, ingest, metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset, read_dataset
```

Read `LESSONS_LEARNED.md` once before starting.

## The model pipeline (electron-phonon system folder)

The calculation is a folder of the `examples/elphon/<System>` layout (`paoflow-gen elphon`, or a copy of an example with `system.py` pointing at the QE data): `system.py` holds the configuration (prefix, coupling source, coarse grids, PAO basis, long-range model, cache) and the per-workflow defaults (`TRANSPORT`, `SUPERCONDUCTIVITY`); each workflow folder holds thin drivers over `PAOFLOW.elphon.workflows` and writes its own `output/`. The coarse vertex is cached in `cache/`, so reruns pay only the dense step.

| driver (from the system folder) | output | what it gives | cost |
|---|---|---|---|
| `transport/mobility.py [--nk --nq --temps --doping --window-ev --smearing-ev --adaptive-smearing --target-gap-ev --varshni --electrons-save --screening-eps --quadrupoles --no-long-range --tag NAME]` | `transport/<tag>/mobility.dat` (T, E_F, n_e, n_h, μ_e/μ_h SERTA, μ_e/μ_h MRTA, σ SERTA/MRTA, Seebeck MRTA), `carrier_scattering.npz` (run metadata, state-resolved rates), `scattering_rates.npz` (energy-resolved SERTA/MRTA rates, band edges), `rates_<T>K.dat` | phonon-limited drift mobilities, carrier densities, conductivity, Seebeck of a semiconductor | Si: 3.5 min at 48³ (20 cores); 60³–84³ for converged holes |
| `superconductivity/eliashberg.py --transport [--nk --nq --sigma-ev --adaptive-smearing --mu-star]` | `superconductivity/output/eliashberg.npz` (+ `a2F_tr`, `lambda_tr`, `omega_plasma_ev`), `alpha2F.dat`, `alpha2F_tr.dat` | α²F, α²F_tr, λ, λ_tr, ω_log, Tc, the Fermi-surface plasma frequency of a metal | Pb: 2 min at 48³/24³ (8 cores) |
| `transport/resistivity.py [--temps --rho0 --b-ee --interband-dielectric DIR --thickness-um --angles --omega-min/max/step --simple-drude]` | `transport/output/resistivity.dat` (T, ρ total and per channel, µΩ cm), `eps_<T>K.dat`, `emish_<T>K.dat`, `emis_th<deg>_<T>K.dat`, `emist.dat`, `emissivity.npz` | ρ(T) by Matthiessen's rule (e-ph from α²F_tr + residual + Fermi liquid), extended-Drude ε(ω, T), the thermal emissivity of the metal | seconds (no vertex run) |
| `transport/plot_mobility.py`, `transport/plot_resistivity.py [--experiment rho.dat]` | `output/*.png` | quick-look figures (the study makes its own) | seconds |
| `PAOFLOW.transport(scattering_channels=ch.metal_channels(npz, rho0, b_ee))` or `ch.carrier_channels(rates_npz, kind, band_edges, extra=[ch.impurity_channel(mu, m_eff)])` | the Boltzmann tensors of a PAOFLOW run (`main.py`) | σ(T), Seebeck, κ_e with the e-ph channel plus impurity / residual / e-e channels | a PAOFLOW run (minutes) |

`MobilityModel.from_system(folder, output='<tag>')` and `ResistivityModel.from_system(folder, output='<tag>')` open these outputs; `describe()` prints the `system.py` configuration and the run metadata (grids, smearing, doping, gap and scissor mode, electrons provenance, screening, window states; channels, λ, λ_tr, ω_p, the Drude model and geometry of the emissivity) for the report. Proposed runs are written as these commands, with their flags and cost. Never run them into an existing output: give a `--tag`, or copy the folder for `resistivity.py` (it writes `transport/output/`).

---

## 0. Operating rules (always)

1. **Read-only inputs.** Never write into calculation folders or source folders. Renders and extracted images go to the study's `sources/`.
2. **Ask before compute.** Ask before any new calculation or anything above ~10 min of CPU. Comparing existing output needs no new run. `transport/resistivity.py` with other channels and a PAOFLOW Boltzmann run with impurity channels are cheap; a new dense vertex run is not.
3. **Provenance on every number**: `measured`, `transcribed`, `vector-extracted`, `digitised`, `calculated`; a replot in a later paper is not the original ("as replotted in …").
4. **Uncertainty on every digitised number** (calibration residuals, marker size in data units, assumptions in the dataset header).
5. **Look at every overlay.**
6. **A baseline with every metric**: a published calculation of the same quantity (EPW, Boltzmann, literature), the spread between measurements, or a trivial model (power law, Bloch-Grüneisen, a constant).
7. **Conditions are matched, not assumed.** Temperature, doping, carrier type, residual resistivity, measurement method. Model values between grid temperatures are *interpolated models*: label them and quote the bracketing sensitivity.
8. **No silent fitting.** A residual resistivity added to the model (`rho0_uohm_cm`), a Hall factor, or a Matthiessen impurity mobility are *stated corrections*, quoted next to the uncorrected deviation and justified by the sample.
9. **Compare the observable that was measured.** Drift ≠ Hall mobility (the Hall factor r_H, 1.0–1.3); conductivity needs the measured carrier density, not the intrinsic one; ρ(T) of a sample includes ρ₀; a phonon-limited mobility is an *upper bound* for a doped sample.
10. **Correct yourself explicitly** (dated notes).
11. **PAOFLOW conventions.** Inputs are Python arguments; no config files; new library code comes with fast tests and NumPy docstrings.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: model runs, sources, sample, audience | `describe()`, `ingest.pdf_inventory` | inventory, open questions |
| 1 | Physics map: observable ↔ model quantity ↔ conditions ↔ baseline | (reasoning, the user) | `PLAN.md` (table) |
| 2 | Ingestion by form → canonical datasets | thermo-optics `ingest.*`, `digitize.*`, `datasets.*` | `data/*.csv`, `checks/` |
| 3 | Validation of the ingestion | overlays, replots, sanity anchors | accepted datasets |
| 4 | Model at the experimental conditions | `MobilityModel`, `ResistivityModel`, `baselines`, `channels` | `data/model_curves.csv` |
| 5 | Metrics with baselines | `metrics.*`, `baselines.*` | `tables/metrics.csv`, derived tables, `tables/summary.md` |
| 6 | Diagnosis, proposed runs | existing output, cheap recomputations, the decision tree | ranked causes, run proposals |
| 7 | Publication figures | `plotting.*` | `figures/*.pdf`, `*.png` |
| 8 | Report | `templates/report_template.md` | `REPORT.md` |

## Stage 0: intake

1. **Model.** Open the system folder: `MobilityModel.from_system(folder, output=...)` and/or `ResistivityModel.from_system(folder, output=...)`; `describe()` gives the configuration and the run metadata. Read `system.py` for the defaults the run inherited and the driver log for the flags actually used (lesson 4: copied drivers drift). Note what the model *cannot* contain: no impurity or defect scattering (phonon-limited), no Hall factor (drift mobilities), intrinsic or fixed doping statistics, the temperature range, the SERTA/MRTA approximations, the Fan term only, no spin-orbit coupling unless the DFT had it.
2. **Sources.** For each file record its form (text or scanned PDF, embedded bitmap or vector figure, table, deck, image) and the *sample conditions* stated in the text: doping and carrier type, purity (ρ₀, RRR), method (Hall, time-of-flight, four-probe, conductivity + Hall density), temperature range, geometry (thin film vs bulk: surface and grain-boundary scattering).
3. **Audience and deliverable.**
4. Report the inventory and the questions before ingesting anything.

## Stage 1: physics map (decide what is compared)

One row per comparison: *observable (source, figure) ↔ model quantity ↔ conditions to match ↔ comparison space ↔ window ↔ baseline*. Rules:

- **Semiconductors.**
  - μ(T): `log10` space, the phonon-limited window is the intrinsic / high-purity regime (Si: above ~100 K for samples with n ≲ 10¹⁴ cm⁻³); quote the power-law exponent (`MobilityModel.power_law_exponent`, `bl.fit_power_law`) over the same window for model and experiment.
  - μ(n) at fixed T: the model is the phonon-limited limit; the impurity-limited branch needs a Matthiessen impurity channel (`bl.matthiessen_mobility`, `ch.impurity_channel`) and is a *diagnostic*, not a prediction.
  - Hall mobility: divide by a stated Hall factor or compare with drift measurements only.
  - σ(T), Seebeck(T): only when the measured carrier density matches the model doping (`--doping`); otherwise compare μ.
  - Baselines: the published EPW/Boltzmann value at the same level of theory (e.g. Poncé 2018 for Si: 1343 / 722 cm²/Vs), the spread between experiments, a power law.
- **Metals.**
  - ρ(T): `linear` space above about θ_D/5 (the linear regime; quote the slope dρ/dT with `ResistivityModel.slope` / `bl.fit_linear`), `log10` below it (the T^n regime). The experimental curve contains ρ₀: subtract the stated ρ₀, or add it to the model (`resistivity(T, rho0_uohm_cm=...)`), and say which.
  - RRR, ρ₀: sample properties; `residual_resistivity_ratio`.
  - λ_tr, ω_p: against published values and other calculations; both drive ρ, so quote them with their convergence (grids, smearing).
  - Emissivity totals and spectra: `emissivity_total`, `emissivity_spectrum` at the measured angle / thickness; `planck_coverage` must be ≈ 1 at the temperatures compared.
  - Baselines: the Bloch-Grüneisen fit to the experiment (`bl.fit_bloch_gruneisen`: ρ₀, A, θ), a published calculation, the Matthiessen offset (`bl.matthiessen_offset`).
- **Prefer derived, robust quantities** where the model has known offsets: exponents, slopes, ratios between temperatures, ω_p, λ_tr.
- **Flag out-of-scope physics up front**: impurity and defect scattering, grain boundaries, surface scattering of films, phonon drag in the Seebeck coefficient, Kondo or magnetic scattering, superconducting transitions, saturation of ρ at high T.
- **Confirm with the user** before ingestion.

## Stage 2: ingestion by form

Follow Stage 2 and the raster recipe of `agentic/thermo_optics/AGENT_WORKFLOW.md`. Everything downstream reads the canonical long format (`datasets.COLUMNS`): `dataset, quantity, x, x_unit, y, y_unit, temperature_k, conditions, provenance, source, locator, y_rel_uncertainty, note`. Transport conventions:

| quantity | x, x_unit | y, y_unit | conditions keys |
|---|---|---|---|
| `mobility_drift_e`, `mobility_drift_h`, `mobility_hall_e`, `mobility_hall_h` | `temperature`, `K` (or `carrier_density`, `cm-3` at fixed T) | `cm2/Vs` | `doping_cm3`, `carrier`, `method` (hall, tof, conductivity), `hall_factor` if stated |
| `carrier_density` | `temperature`, `K` | `cm-3` | `doping_cm3`, `method` |
| `conductivity`, `resistivity` | `temperature`, `K` | `S/m`, `uOhm cm` | `rho0_uohm_cm`, `rrr`, `sample`, `geometry` (bulk, film + thickness) |
| `seebeck` | `temperature`, `K` | `uV/K` | `doping_cm3`, `carrier` |
| `emissivity_total`, `emissivity_spectral` | `temperature`, `K` / `wavelength`, `um` | `1` | `angle_deg`, `surface` |
| `lambda_tr`, `plasma_frequency` | (one row) | `1`, `eV` | `method` |

`temperature_k` is NaN for an x = temperature curve (the temperature is x); use it for isotherms. Keep the printed units and convert in `analysis.py` (`thermo_optics.units.to_kelvin`; mobility and resistivity units by hand, stated in the header).

## Stage 3: validation of the ingestion

Calibration residuals; sanity anchors (ρ(300 K) of a clean metal from a handbook, μ_e of Si ≈ 1400 cm²/Vs, the intrinsic carrier density); cross-figure checks; assumptions in the header and the report caveat box. Log axes: read the decades, not the ticks (lesson 14 of the thermo-optics workflow).

## Stage 4: model adapter and condition alignment

- **`MobilityModel`** (`transport/<tag>/mobility.dat`): `mobility(T, carrier='e'|'h', kind='serta'|'mrta', interpolate)` (ln μ linear in ln T off the grid), `carrier_density(T, carrier, interpolate)` (ln n linear in 1/T), `conductivity(T, kind, interpolate)`, `seebeck(T, interpolate)`, `power_law_exponent(carrier, kind, window)`, `bracket(T)`, `metadata` (grids, smearing, doping, gap, scissor mode, electrons provenance, screening, window states), `describe()`.
- **`ResistivityModel`** (`transport/<tag>/resistivity.dat`, `emissivity.npz`, `eliashberg.npz`): `resistivity(T, channel='total'|<name>, rho0_uohm_cm, interpolate, law='linear'|'power')`, `slope(window, channel)`, `residual_resistivity_ratio(rho0, t_low, t_high)`, `emissivity_total(T, opaque, interpolate)`, `emissivity_spectrum(T, kind, angle_deg, interpolate)`, `dielectric(T, interpolate)`, `channels`, `eliashberg` (λ, λ_tr, ω_p, ω_log, μ*, Tc), `describe()`.
- **Corrections, stated:** the Hall factor (divide the measured Hall mobility, or multiply the model, by r_H and say which value and source); the residual resistivity (add the sample's ρ₀ to the model, or subtract it from the experiment); Matthiessen's rule for an impurity-limited sample (`bl.matthiessen_mobility(mu_ph, mu_imp)` with μ_imp from Brooks-Herring or from the low-T data, as a diagnostic).
- **Temperatures.** `mt.match_conditions(exp_T, model_T, tolerance)`. Beyond the tolerance, interpolate (label it and quote the bracketing biases) or drop the point. Mobilities interpolate as power laws, carrier densities as activated quantities, resistivities linearly (power law below θ_D/5).
- **The Boltzmann bridge** (when the user wants σ, Seebeck or κ_e with extra channels): build the channel list with `ch.metal_channels(eliashberg_npz, rho0_uohm_cm, b_ee_per_ev)` or `ch.carrier_channels(scattering_rates_npz, kind, band_edges, extra=[ch.impurity_channel(mu_imp, m_eff)])`, pass it to `PAOFLOW.transport(scattering_channels=...)` in a `main.py` next to the study (a new run: ask first), and treat its output as another model run (adapter: `MobilityModel`-like reader of PAOFLOW's `sigma*.dat`, `Seebeck*.dat`; write it as a small class in the study).
- **Write every model curve used** to `data/model_curves.csv` (`provenance='calculated'`, the interpolation and corrections in `note`).

## Stage 5: comparison metrics with baselines

| Tool | Use |
|---|---|
| `mt.curve_deviation(x_e, y_e, x_m, y_m, space, window)` | n, bias, rms, median, max\|res\| (`log10` for μ, n, σ; `linear` for ρ above θ_D/5, emissivity) |
| `bl.fit_power_law(T, mu, window)`, `MobilityModel.power_law_exponent` | exponents of μ(T) for experiment and model over the same window |
| `bl.fit_linear(T, rho, window)`, `ResistivityModel.slope` | dρ/dT of experiment and model |
| `bl.fit_bloch_gruneisen(T, rho)` | ρ₀, A, θ of the experiment: the trivial model and the sample's ρ₀ |
| `bl.matthiessen_offset(T_e, rho_e, T_m, rho_m, window)` | the residual that reconciles the model with the sample (diagnostic), rms before/after, slope ratio |
| `mt.arrhenius_fit(T, n)` | activation energies of carrier densities |
| `mt.best_shift` | rigid shifts in T or ln μ (diagnostic only) |

One row per (comparison, dataset, condition, window) in `tables/metrics.csv`, with `baseline`, `baseline_rms` and `note`. Derived quantities (exponents, slopes, ρ₀, θ, λ_tr, ω_p, RRR) get their own tables. Void a baseline that has no meaning.

## Stage 6: diagnosis

Look for **one cause explaining several discrepancies** first. Cheap checks on existing output before any new run:

- *Semiconductors*: SERTA vs MRTA (`mobility.dat` has both); the window states and smearing (`metadata`); the grid convergence of the available runs (Si holes: 60³ → 84³ moves μ_h by 10 %, electrons by 2 %); the long-range and quadrupole flags actually used (`carrier_scattering.npz`, the driver log); the electrons provenance (PBE vs Hubbard-corrected masses); the screening (`coupling_screening`); the doping and gap statistics (`doping_cm3`, `gap_ev`, scissor mode: they move n and σ, hardly μ); the Hall factor and the sample's impurity content (Matthiessen as a diagnostic).
- *Metals*: λ_tr and ω_p against their convergence with the k-grid (ω_p needs ≥ 48³ k for Pb), the q-grid and the smearing (fixed-width λ scatters ±8 % between grids; adaptive widths are flat); ρ₀ of the sample (`matthiessen_offset`); the e-e term; the Fan-only coupling and scalar-relativistic DFT (Pb: PBE λ = 1.2 vs 1.55 measured; spin-orbit raises it); the extended vs simple Drude at the temperatures compared; the Planck coverage of the emissivity integrals.

Then recomputations that need no vertex run (`transport/resistivity.py --rho0 --b-ee`, other geometry; a PAOFLOW Boltzmann run with impurity channels). Then propose runs as exact driver commands of the system folder (pipeline table), each with `--tag`, its expected effect and its cost. See the decision tree.

## Stage 7: publication figures

As in the thermo-optics playbook (`plotting.publication_style`, `temperature_colors` for ordered conditions, experiment as markers, this work as solid lines, published theory gray dashed, direct labels + style legend, `provenance_note`, `save_figure` → PDF + 600 dpi PNG; look at every PNG). Transport specifics: μ(T) on log-log axes with the exponent annotated; ρ(T) linear with ρ₀ stated in the caption; isotherms μ(n) with the phonon-limited model as a horizontal bound and the Matthiessen curve dashed as a diagnostic; one axis per panel.

## Stage 8: report

Follow `templates/report_template.md`: provenance caveat box (including the Hall factor, ρ₀ and doping assumptions), bottom-line table, data inventory, model run (`describe()` tables), results per comparison, diagnosis and proposed runs, how to reproduce.

---

## Decision tree for a discrepancy

```
model - experiment discrepancy
 ├─ same observable?                      -> drift vs Hall (r_H), sigma vs mu (carrier density), rho with/without rho0
 ├─ same sample?                          -> doping / purity: Matthiessen impurity channel or rho0 as a diagnostic;
 │                                           film vs bulk (surface, grain boundaries)
 ├─ same conditions?                      -> temperature window (phonon-limited regime), bracketing T, interpolation law
 ├─ ingestion error?                      -> overlay, replot, log-axis decades, label inference
 ├─ convergence?                          -> dense k/q grids, smearing (fixed vs adaptive), window states, Planck coverage
 ├─ model choices?                        -> SERTA vs MRTA, long-range / quadrupoles, screening, electrons (PBE vs Hubbard),
 │                                           scalar-relativistic vs SOC, simple vs extended Drude
 ├─ missing physics?                      -> impurities, defects, grain boundaries, phonon drag, e-e, saturation, Kondo
 └─ none                                  -> report unexplained, with what was excluded and which run or measurement would decide
```

## Ask the user at the start

1. The system folder(s) (read-only) and which `transport/<tag>` / `superconductivity/output` runs to compare, including older ones.
2. The sources, with their figures or tables and their provenance. Is any of them a replot?
3. The sample: doping and carrier type, purity (ρ₀ / RRR), method (Hall or drift), geometry, temperature range.
4. The deliverable and its audience.
5. The compute budget: may new runs (dense vertex, Boltzmann with extra channels) be launched?

## Worked examples (October 2026, on the user's workstation)

- **Si mobility**: `Workspace/PAOFLOW/production/Si_mobility_quadrupoles/` (12³ nscf / 6³ DFPT q, 10 meV smearing, 300 K, dipole-only long-range model). Converged values e / h (SERTA | MRTA): 60³ 1283 / 495 | 1302 / 492; 72³ 1314 / 517 | 1350 / 515; 84³ 1285 / 541 | 1326 / 539 cm²/Vs. Baseline: Poncé *et al.* 2018 (EPW, 1343 / 722); experiment ≈ 1400 / 470–500 (drift). The quadrupole series is pending (the first attempt reran the dipole-only model: lesson 4). Table in `examples/elphon/Si/README.md` §5.
- **Pb resistivity**: `examples/elphon/Pb/README.md` §5 and `Workspace/PAOFLOW/production/Pb_superconductivity_transport/`: λ_tr = 0.94, ħω_p = 8.34 eV (48³/24³, σ = 0.05 eV); ρ = 0.52, 2.4, 5.2, 10.8, 16.2, 21.7, 32.6 µΩ cm at 20, 50, 100, 200, 300, 400, 600 K against ≈ 21 µΩ cm measured at 300 K; refining q to 48³ moves ρ(300 K) by 6 %; total hemispherical emissivity 0.012 / 0.037 / 0.071 at 100 / 300 / 600 K.
