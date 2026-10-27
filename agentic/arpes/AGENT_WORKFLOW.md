# ARPES workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) comparing PAOFLOW's **simulated ARPES** with **measured spectra** for **any material**: band maps along a cut with the photoemission matrix elements of the PAO orbitals (`PAOFLOW.arpes`), the dispersion dressed by the electron-phonon self-energy (the kink, λ, the renormalised Fermi velocity), the self-energy Σ(ω, T) in its isotropic (α²F), anisotropic (Fermi-surface coupling) or band-resolved Fan-Migdal form, the momentum widths, and the phonon linewidths of every state versus temperature. The model is an electron-phonon *system folder* (`examples/elphon/<System>` layout; see "The model pipeline"). The stages are those of the thermo-optics workflow (`agentic/thermo_optics/AGENT_WORKFLOW.md`), whose ingestion recipes apply unchanged; this playbook adds what is specific to ARPES: the observables, the alignments that must be made explicit (k_F, E_F, the bare band, the resolution) and the baselines.

The tooling is `PAOFLOW.agentic.arpes` (`ArpesModel`, `curves`, `images`) over the shared toolkit `PAOFLOW.agentic.thermo_optics`. A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `digitize_sources.py` (from `agentic/thermo_optics/templates/digitize_template.py`) | the sources (PDF, image, table, deck) | `data/<source>.csv`, `checks/*_overlay.png`; for band-map images also `data/<source>_map.npz` (k, E, intensity) |
| `analysis.py` | `data/*` + the model output (read-only) | `data/model_curves.csv`, `tables/metrics.csv`, `tables/*.csv`, `tables/summary.md` |
| `plot_figures.py` | CSV / npz files only | `figures/*.pdf` (vector) + `*.png` (600 dpi) |

```python
from PAOFLOW.agentic.arpes import ArpesModel
from PAOFLOW.agentic.arpes import curves as cv
from PAOFLOW.agentic.arpes.images import band_map_from_image
from PAOFLOW.elphon.self_energy import eliashberg_self_energy, mass_enhancement
from PAOFLOW.agentic.thermo_optics import digitize as dg, ingest, metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset, read_dataset
```

Read `LESSONS_LEARNED.md` once before starting.

## The model pipeline (electron-phonon system folder)

`system.py` holds the configuration and the `ARPES` defaults (`photon_ev`, `work_function_ev`, `polarization`, `resolution` (energy, momentum), `window` (E_min, E_max, n), `direction` and `direction_label`, `segment_ev` or the full path, `npts`, `nq`, `eta_ev`, `adaptive_smearing`, `temps_k`, `self_energy`, `intensity_power`). The driver writes `arpes/output/` (no `--outdir`): rename a previous `output/` to `output_<label>/` before a rerun.

| driver (from the system folder) | output | what it gives | cost |
|---|---|---|---|
| `superconductivity/eliashberg.py [--fs-coupling]` | `superconductivity/output/eliashberg.npz` (`fs_coupling.npz`) | α²F (and the Fermi-surface pair coupling) the self-energies are built from | minutes |
| `arpes/arpes.py --self-energy isotropic [--temps T ...] [--direction a b c] [--segment-ev E \| --full-path] [--npts N]` | `arpes_bare.npz`, `arpes_isotropic_<T>K.npz`, `sigma_isotropic_<T>K.dat`, `arpes.png` | the band map across the Fermi crossing and the Engelsberg-Schrieffer Σ(ω, T) from α²F: the kink, λ and Z from the slope | seconds |
| `arpes/arpes.py --self-energy anisotropic ...` | `arpes_anisotropic_<T>K.npz`, `sigma_anisotropic_<T>K.dat` (crossing / weakest / strongest state + isotropic) | Σ_nk(ω, T) of the nearest Fermi-surface states (two-gap metals) | seconds |
| `arpes/arpes.py --self-energy fan-migdal --nq N [--eta-ev] [--adaptive-smearing a] [--full-path]` | `self_energy_fan_migdal.npz` (Σ_nk(ω, T) on the path), `arpes_fan_migdal_<T>K.npz` | band-resolved Fan-Migdal Σ: linewidths of every state versus T (semiconductors, wide windows) | Pb 18³ q: 13 min on 8 cores; Si 12³ q full path: 15 min on 4 cores |

`ArpesModel.from_system(folder, output='output')` opens these; `describe()` prints the configuration, the experiment settings of the maps (photon energy, polarisation, resolution, window), the path, the window bands, the dressed maps and the Fan-Migdal parameters (q-grid, η, adaptive widths). The map layout is `intensity (nk, nw)` on the path distance (1/Å) and the energies (eV, relative to E_F).

---

## 0. Operating rules (always)

1. **Read-only inputs.** Never write into calculation folders or source folders.
2. **Ask before compute.** A Fan-Migdal run or a dense `--nq` is above the 10-minute line; isotropic / anisotropic reruns (other temperatures, segment, resolution) take seconds and still need a renamed `output/`.
3. **Provenance on every number**; a replot is not the original; an *extracted* Σ or λ from a paper carries the authors' bare-band assumption (record it in `conditions`).
4. **Uncertainty on every digitised number**; for band-map images also the calibration residuals of both axes and the colour-scale assumption.
5. **Look at every overlay**, including the model dispersion drawn over a digitised map.
6. **A baseline with every metric**: the authors' own analysis (their λ, Σ), a tunnelling α²F through `eliashberg_self_energy`, a Debye-model or linear-band fit, a second measurement.
7. **Alignments are explicit.** Dispersions are compared versus k − k_F (each side's own k_F); an E_F offset is a diagnostic from `mt.best_shift`, stated; the bare band is defined the same way on both sides (`cv.linear_band` over the same energy window, or the authors' choice); the resolution of the model equals the measured one.
8. **No silent fitting.** Intensity scales, offsets and bare-band choices are quoted next to the raw comparison.
9. **Compare the observable that was measured**: MDC peak positions and widths (not EDC widths, which carry the Fermi cutoff and the energy resolution); linewidths at the band energy; λ by a named method (slope of Re Σ at E_F, velocity ratio over a window, dΓ/dT).
10. **Correct yourself explicitly** (dated notes).
11. **PAOFLOW conventions.** Inputs are Python arguments; no config files; new library code comes with fast tests and NumPy docstrings.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: model runs, sources, measurement conditions, audience | `ArpesModel.describe()`, `ingest.pdf_inventory` | inventory, open questions |
| 1 | Physics map: observable ↔ model quantity ↔ alignment ↔ baseline | (reasoning, the user) | `PLAN.md` |
| 2 | Ingestion by form → canonical datasets and calibrated maps | thermo-optics `ingest.*`, `digitize.*`, `datasets.*`; `images.band_map_from_image` | `data/*.csv`, `data/*_map.npz`, `checks/` |
| 3 | Validation of the ingestion | overlays, replots, anchors (k_F, E_F, known band positions) | accepted datasets |
| 4 | Model at the experimental conditions | `ArpesModel`, `curves` | `data/model_curves.csv` |
| 5 | Metrics with baselines | `metrics.*`, `curves.*`, `eliashberg_self_energy` | `tables/metrics.csv`, derived tables, `tables/summary.md` |
| 6 | Diagnosis, proposed runs | existing output, the decision tree | ranked causes, run proposals |
| 7 | Publication figures | `plotting.*` | `figures/*.pdf`, `*.png` |
| 8 | Report | `templates/report_template.md` | `REPORT.md` |

## Stage 0: intake

1. **Model.** `ArpesModel.from_system(folder, output=...)`; `describe()`. Note: the cut (direction, segment or full path, number of points: at least five points per MDC width for the `'peak'` method, else use `'lorentzian'`), the photon energy and polarisation, the resolution, the energy window, which self-energies and temperatures exist, the Fan-Migdal q-grid and widths. What the model cannot contain: the Fan term only (no Debye-Waller: no band-edge shifts versus T); no impurity or electron-electron self-energy (Im Σ(0) → 0 in the model, finite in experiment); no k_z broadening or surface states unless the slab was built; the isotropic Σ ignores anisotropy; the Fan-Migdal Σ at affordable q-grids is structureless for metals (lesson 3).
2. **Sources.** Per file: form (text or scanned PDF, vector or raster figure, table, image file, deck) and the *measurement conditions* stated: photon energy, polarisation, temperature, energy and momentum resolution, the cut and k_z (photon-energy scan?), surface preparation, doping or substrate, the bare band used by the authors for Σ and λ, the k_F they quote.
3. **Audience and deliverable.**
4. Report the inventory and the questions before ingesting anything.

## Stage 1: physics map (decide what is compared)

One row per comparison: *observable (source, figure) ↔ model quantity ↔ alignment ↔ comparison space ↔ window ↔ baseline*. Observables, from the most to the least robust:

| experimental quantity | model quantity (`ArpesModel`) | alignment / space | baseline |
|---|---|---|---|
| λ (kink) by a stated method | `velocities(kind, T, window_ev)['lambda']` with the same window; `mass_enhancement(kind, T)` (slope of Re Σ) | method matched | the authors' λ; λ from a tunnelling α²F (`mass_enhancement(w, eliashberg_self_energy(...))`) |
| kink energy | `kink_energy(kind, T)` (extremum of Re Σ) | E_F offset as a diagnostic | the phonon energies (α²F peaks) |
| dispersion E(k) from MDC fits | `dispersion(kind, T)` → (`energy_ev`, `k`) versus k − k_F | k − k_F; `linear` in meV; windows: below the kink / above | the bare band (`bare_dispersion`) and a linear fit (`cv.linear_band`) |
| Re Σ(ω), Im Σ(ω) extracted by the authors | `self_energy(kind, T)`; or `self_energy_from_map(kind, T)` extracted the same way against the same bare-band protocol | ω in meV; `linear`; impurity offset Im Σ(0) subtracted and stated | the authors' Σ; Engelsberg-Schrieffer Σ from a tunnelling α²F |
| MDC FWHM(ω) | `dispersion(..., method='lorentzian')['fwhm_k']` at the measured resolution | ω; `linear` or `relative` | the resolution floor; the bare-velocity scaling Δk = 2\|Im Σ\|/v_b |
| linewidth Γ(T) of a state | `linewidth_vs_temperature(band, ik)` (Fan-Migdal); slope dΓ/dT = 2πλ_nk k_B above Θ_D | the same state (band, k); `linear` | the slope formula with the authors' λ_nk; a second state |
| band map (image) | `map(kind, T)['intensity']` with the dispersion overlaid; MDC analysis of both with the same windows | k − k_F, E; intensities normalised per MDC (matrix elements are not compared) | the bare map |

Rules: **flag out-of-scope physics** (impurity Im Σ(0), electron-electron Σ ∝ ω², k_z broadening, surface states, Debye-Waller shifts, superconducting gaps below Tc, final-state effects); **confirm with the user** before ingestion.

## Stage 2: ingestion by form

Follow Stage 2 and the raster recipe of `agentic/thermo_optics/AGENT_WORKFLOW.md`. Canonical datasets (`datasets.COLUMNS`) with these conventions:

| quantity | x, x_unit | y, y_unit | conditions keys |
|---|---|---|---|
| `dispersion` | `k`, `1/A` (relative to the authors' k_F if so published: `k_reference=kF`) | `E`, `meV` or `eV` (relative to E_F) | `photon_ev`, `polarization`, `cut`, `kz`, `resolution_mev`, `method` (mdc, edc), `bare_band` |
| `mdc_fwhm` | `E`, `meV` | `dk`, `1/A` | as above |
| `re_self_energy`, `im_self_energy` | `omega`, `meV` | `meV` | `bare_band` (linear fit window, LDA, parabola), `impurity_offset_mev` |
| `lambda`, `kink_energy` | (one row; `x` = temperature if several) | `1`, `meV` | `method` (slope, velocity ratio, width slope), `bare_band` |
| `linewidth` | `temperature`, `K` | `meV` (FWHM or HWHM: say which) | `band`, `k`, `state`, `method` |

`temperature_k` holds the measurement temperature. **Band-map images**: calibrate both axes (`dg.Axis` from the frame and ticks, residuals recorded), then `band_map_from_image(image, k_axis, e_axis, rows, cols, channel, dark_is_high)` → `data/<source>_map.npz` (`k`, `energy_ev`, `intensity`); record the colour-scale assumption (lesson 13), and run `cv.mdc_dispersion` on it with the same windows as on the model; write the dispersion it gives as a `dispersion` dataset with `method=mdc-from-image`.

## Stage 3: validation of the ingestion

Overlay of the digitised points or of the extracted dispersion on the source image; replot in data coordinates; anchors: k_F quoted by the authors, E_F = 0 at the Fermi cutoff, a known band position; the MDC-from-image dispersion against the authors' own fits where both exist. Assumptions in the header and the report caveat box.

## Stage 4: model adapter and condition alignment

- **`ArpesModel`**: `kinds`, `temperatures(kind)`, `map(kind, T)`, `band_energies()`, `bare_dispersion(band)`, `bare_fermi_momentum()`, `dispersion(kind, T, energy_window, k_window, method='peak'|'lorentzian')`, `fermi_momentum`, `velocities(kind, T, window_ev)` (bare and dressed v_F, λ = |v_b/v| − 1, both k_F), `self_energy_from_map(kind, T)` (Re Σ = ω − ε_b(k(ω)), Im Σ = −Δk v_b/2 against the model's own bare band), `self_energy(kind, T, column)` and `mass_enhancement`, `kink_energy`; Fan-Migdal: `linewidths(T)`, `linewidth_vs_temperature(band, ik)`, `state_self_energy(T, band, ik)`; `mdc`, `edc`; `describe()`.
- **Alignment.** Shift both dispersions to their own k_F (`fermi_momentum`), compare versus k − k_F. Use the same energy windows and the same `method` on model and experiment (the model map analysed with `cv.mdc_dispersion` exactly as the measured one). Define the bare band identically: the authors' linear fit window → `cv.linear_band(energies, k, window)` on the model's dressed dispersion as well; quote also the model's exact bare band. Set the model resolution to the measured one (`ARPES['resolution']`; a rerun of seconds for the isotropic case).
- **Temperatures.** `mt.match_conditions`; Σ(ω, T) of the isotropic form can be recomputed at any T from α²F with `eliashberg_self_energy(omega, a2F, w, T)` on `superconductivity/output/eliashberg.npz` (no driver run), so no interpolation in T is needed for Σ.
- **Impurity offset.** Experimental Im Σ(0) ≠ 0: compare Im Σ(ω) − Im Σ(0) and state the offset, or add it to the model as a stated constant.
- **Write every model curve used** to `data/model_curves.csv` (`provenance='calculated'`, the alignment and the bare-band protocol in `note`).

## Stage 5: comparison metrics with baselines

| Tool | Use |
|---|---|
| `mt.curve_deviation(k_e − kF_e, E_e, k_m − kF_m, E_m, 'linear', window)` | dispersion deviation (meV) below / above the kink |
| `mt.curve_deviation(w, ReΣ_e, w, ReΣ_m, 'linear')`, same for Im Σ − Im Σ(0) | self-energy deviation (meV) |
| λ table: method × side | λ_slope (`mass_enhancement`), λ_velocity (`velocities`), λ_width (dΓ/dT), the authors' λ, tunnelling-α²F λ |
| kink energy versus the α²F peaks | `kink_energy`; `EliashbergModel` of `PAOFLOW.agentic.superconductivity` for the phonon energies |
| `mt.best_shift` in E | the E_F offset that best aligns Re Σ or the dispersion (diagnostic) |
| `np.polyfit(T, Γ)` | dΓ/dT of experiment and model for the same state |

One row per (comparison, dataset, condition, window) in `tables/metrics.csv`, with `baseline`, `baseline_rms`, `note`. Derived quantities (λ by method, kink energies, v_F, k_F, slopes) in their own tables.

## Stage 6: diagnosis

Cheap checks first: the bare-band protocol (λ changes by tens of percent with it: lesson 1); the energy window of the velocity fit; k_F alignment and the segment choice; the resolution used in the model; isotropic versus anisotropic Σ at the measured k (two-gap metals, lesson 14); the Fan-Migdal q-grid and widths (lesson 3); the E_F offset; the impurity offset; the photon energy and polarisation of the maps (matrix elements change relative intensities, not positions: lesson 5); scalar-relativistic versus SOC bands (Pb). Then propose runs as exact driver commands (pipeline table) with cost. See the decision tree.

## Stage 7: publication figures

As in the thermo-optics playbook. ARPES specifics: band maps in a sequential grey or single-hue map with the measured dispersion as markers and the model dispersion as a line, both versus k − k_F; the bare band dashed; Σ panels with Re and Im on separate axes (one axis each); linewidths versus T with the slope annotated; intensities normalised per MDC when overlaying model and experiment. One axis per panel, no dual axes.

## Stage 8: report

Follow `templates/report_template.md`: the caveat box names the bare-band protocol, the k_F and E_F alignments, the resolution, the impurity offset and the colour-scale assumption of any digitised map.

---

## Decision tree for a discrepancy

```
model - experiment discrepancy
 ├─ same bare band?                  -> same window / protocol on both sides; quote lambda per method
 ├─ aligned?                         -> k - k_F on each side; E_F offset (best_shift) as a diagnostic; segment covers 2x the kink
 ├─ same resolution and method?      -> MDC vs EDC; 'lorentzian' widths; model resolution = measured; npts per MDC width
 ├─ same state?                      -> cut, k_z, band index, anisotropy (isotropic vs anisotropic Sigma)
 ├─ ingestion error?                 -> overlay, replot, colour scale, axis calibration residuals
 ├─ convergence of Sigma?            -> Fan-Migdal q-grid and widths (structureless at coarse q); alpha^2F grids and smearing
 ├─ missing physics?                 -> impurity Im Sigma(0), e-e, Debye-Waller shifts, k_z broadening, surface states, SOC
 └─ none                             -> report unexplained, with what was excluded and which run or measurement would decide
```

## Ask the user at the start

1. The system folder(s) (read-only) and which `arpes/output*` runs to compare.
2. The sources, with their figures or tables and their provenance; the bare band the authors used.
3. The measurement conditions: photon energy, polarisation, temperature, resolution, cut and k_z, surface, doping.
4. The deliverable and its audience.
5. The compute budget: may Fan-Migdal or dense-q runs be launched?

## Worked examples (in the repository)

- **Pb** (`examples/elphon/Pb/README.md` §6): isotropic Σ at 10 K from α²F (λ = 1.03 of the run): slope λ = 1.11, Z = 0.48, Im Σ plateau 8.7 meV beyond the phonon energies (≈ 11 meV), Re Σ extrema at ±7 meV; the Fan-Migdal form at 18³ q with adaptive widths is structureless (Im Σ ≈ −7.3 meV, Re Σ ≈ −6.5 meV over the window).
- **MgB₂** (`examples/elphon/MgB2/README.md` §6): the σ-band crossing on Γ–M with the anisotropic Σ of the crossing, weakest and strongest states (λ_nk 0.33–0.99) and the isotropic one.
- **Si** (`examples/elphon/Si/README.md` §6): Fan-Migdal linewidths at 10 / 300 / 600 K (12³ q, 20 meV width): VBM at Γ 1.3 / 2.7 / 5.4 meV; second valence band at X 23.5 / 39.2 / 71.6 meV; conduction band at Γ 53.8 / 70.6 / 105.3 meV; the 20 meV width is the floor of the smallest values; matrix elements: the s-like band is 30× the p-like ones at He I.
