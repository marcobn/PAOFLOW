# Non-linear optics workflow: playbook for the agent

This is the operating manual for an AI agent (or a person) comparing PAOFLOW's **second-order optical responses** (SHG susceptibility χ^{(2)} and the d coefficients, shift current, injection currents, magnetic responses of T-broken crystals; 3D or 2D sheet values) with **experimental data and published calculations of any form** for **any non-centrosymmetric crystal**. The model is an output folder of `PAOFLOW.nonlinear_optics` (see "The model pipeline"). The stages are those of the thermo-optics workflow (`agentic/thermo_optics/AGENT_WORKFLOW.md`), whose ingestion recipes apply unchanged; this playbook adds the conventions that must be matched (d vs χ, components and orientation, photon energy, 3D vs sheet), the **symmetry audit** that precedes every comparison, and the diagnostics specific to second-order responses.

The tooling is `PAOFLOW.agentic.nlo` (`NonlinearModel`, `symmetry`) over the shared toolkit `PAOFLOW.agentic.thermo_optics`. A study has three scripts, each started from `templates/`:

| Script | Reads | Writes |
|---|---|---|
| `digitize_sources.py` (from `agentic/thermo_optics/templates/digitize_template.py`) | the sources | `data/<source>.csv`, `checks/*_overlay.png` |
| `analysis.py` | `data/*.csv` + the model output (read-only) | `data/model_curves.csv`, `tables/symmetry.csv`, `tables/metrics.csv`, `tables/summary.md` |
| `plot_figures.py` | CSV files only | `figures/*.pdf` + `*.png` |

```python
from PAOFLOW.agentic.nlo import NonlinearModel
from PAOFLOW.agentic.nlo import symmetry as sym
from PAOFLOW.agentic.thermo_optics import digitize as dg, ingest, metrics as mt
from PAOFLOW.agentic.thermo_optics.datasets import make_dataset, write_dataset, read_dataset
```

Read `LESSONS_LEARNED.md` once before starting.

## The model pipeline

A system folder of the `examples/nlo_examples/<System>` layout: QE `scf.in` / `nscf.in` (uniform grid, symmetry-reduced or `nosym`; ultrasoft / PAW pseudopotentials are projected in the augmented metric since October 2026, check the projectability vector: `LESSONS_LEARNED.md` #2), optionally `relax.in`, a `BASIS_PS/` generated from the pseudopotentials when the repository tables project the valence bands below 0.95, and `main.py`:

| step | what | cost |
|---|---|---|
| `pw.x` scf, nscf | the nscf grid is the coarse grid of the Wigner–Seitz interpolation (16³ for GaAs, 18×18×1 for GeS); `nbnd` large enough for the PAO basis | minutes |
| `paoflow-genbasis-ps --pseudo <upf> --out BASIS_PS --preset standard` | PAO radial tables matched to the pseudopotentials (check the projectability vector: all valence bands and a few conduction bands above 0.95) | seconds |
| `main.py`: `projections(internal=True, ...)`, `projectability()`, `pao_hamiltonian()`, `nonlinear_optics(nk, emin, emax, ne, eta, scissor, components, twoD, degeneracy_tol, position_dipoles, symmetry)` | the responses on the dense `nk` grid, interpolated from `HRs` on the fly; `output/`. `symmetry=True` (default) reduces the grid to the irreducible wedge (GaAs: 24×); `position_dipoles=True` adds the Löwdin dipoles to the position operator (about 2× the cost) | GaAs 62 orbitals: 15 min at 40³ on the full grid, ~1 h at 60³ on 8 ranks; GeS: 13 min at 200² |
| `paoflow-gen` property `nonlinear` | the same `main.py` from prompts | — |

`NonlinearModel(output_dir, two_d, slab_height_nm, system)` opens an output folder; `describe()` prints what it found (kinds, components, energies, static values, peaks). Parameters of the run are read from `main.py` and its log (grid, η, scissor, components, tolerance, basis, the gap and the number of occupied bands printed by `nonlinear_optics`). New runs go to new `output_<label>` folders.

---

## 0. Operating rules (always)

1. **Read-only inputs.**
2. **Ask before compute** above ~10 min (a dense-grid rerun is one).
3. **Provenance and convention on every number.** Experimental SHG coefficients come as d_ijk (= χ_ijk/2), as |χ| or as d_eff with a geometry; relative to a reference crystal (quartz d_11, KDP d_36) whose accepted value changed over the decades; at a stated photon energy (1.06 µm = 1.17 eV, 1.55 µm = 0.80 eV, CO₂ 10.6 µm = 0.117 eV); photocurrents as a 3D conductivity, a sheet conductivity or a responsivity with a thickness. Record all of this in `conditions`.
4. **Uncertainty on every digitised number.**
5. **Look at every overlay.**
6. **A baseline with every metric**: a published calculation at the same level of theory (LDA/PBE + scissor), a second measurement, Miller's rule or the Kleinman-symmetric static estimate.
7. **The symmetry audit comes first.** Before comparing anything: the point group of the crystal (`sym.point_group_operations(name)`), the forbidden components present in the output and their leakage, the relations between equivalent components and how well the run obeys them, the Kleinman error at ω → 0. These numbers are the symmetry error bar of every model value.
8. **State scissor, η, grid, tolerance and basis with every model number**; interpolations between photon energies are linear and labelled.
9. **Compare the observable that was measured**: the component and orientation, the photon energy, 3D vs sheet, d vs χ; for 2D crystals the literature's thickness convention.
10. **Correct yourself explicitly.**
11. **PAOFLOW conventions**: Python arguments, no config files, tests and NumPy docstrings for library changes.

---

## Stage overview

| Stage | Goal | Tools | Output |
|---|---|---|---|
| 0 | Intake: model run, sources, conventions, audience | `describe()`, `ingest.pdf_inventory` | inventory |
| 1 | Symmetry audit and physics map | `symmetry_check`, `kleinman_check`, (reasoning) | `tables/symmetry.csv`, `PLAN.md` |
| 2 | Ingestion → canonical datasets | thermo-optics `ingest.*`, `digitize.*`, `datasets.*` | `data/*.csv`, `checks/` |
| 3 | Validation of the ingestion | overlays, anchors (accepted static values) | accepted datasets |
| 4 | Model at the measured conditions | `NonlinearModel`, unit conversions | `data/model_curves.csv` |
| 5 | Metrics with baselines | `metrics.*` | `tables/metrics.csv`, `tables/summary.md` |
| 6 | Diagnosis, proposed runs | existing output, the decision tree | ranked causes, run proposals |
| 7 | Figures | `plotting.*` | `figures/` |
| 8 | Report | `templates/report_template.md` | `REPORT.md` |

## Stage 0: intake

1. **Model.** Open the output with `NonlinearModel`; read `main.py` and the log for NK, η, scissor, components, `twoD` and the slab height, the degeneracy tolerance, the basis and the projectability vector, the gap and number of occupied bands, nosym or not, the code version. Note what the model cannot contain: excitons and local-field effects (independent-particle response), the scissor fixes the gap but not the band dispersion, phonon-assisted and finite-lifetime effects beyond the constant η.
2. **Sources.** Per file: form, the quoted quantity and its convention (rule 3), the sample (bulk, film, monolayer on a substrate), the photon energy, the method (Maker fringes, phase matching, relative to a reference crystal, photocurrent under which illumination).
3. **Audience and deliverable.**
4. Report the inventory and the questions before ingesting.

## Stage 1: symmetry audit and physics map

- **Audit** (`tables/symmetry.csv`): point group → `sym.forbidden_components`, `sym.independent_relations`; for each kind (`shga`, `shift`, ...) at a few photon energies: the leakage of the forbidden components present, the deviation between equivalent components (e.g. χ_xyz vs χ_yzx of GaAs), the Kleinman error at ω → 0 (`kleinman_check`), the T-odd kinds for a T-symmetric crystal (must vanish). Quote the largest of these as the symmetry error bar.
- **Physics map**, one row per comparison: *source quantity (convention) ↔ model quantity ↔ photon energy / component / orientation / 3D-or-sheet ↔ space ↔ baseline*. Prefer: static χ (lowest photon energy of the run; compare with the measurement at its photon energy, both below the gap), peak positions and heights of |χ(ω)|, onsets (`onset`: the gap or half the gap), shift-current peaks and sign, the ratio between components.
- **Space**: `linear` for |χ| and σ spectra, `relative` for static values and peaks (quote percentages).
- **Out-of-scope physics up front**: excitons (sub-gap SHG resonances, enhanced shift currents in 2D), local fields, surface SHG in experiments on films, the sample's absorption and phase matching in d_eff measurements.
- **Confirm with the user** before ingestion.

## Stage 2: ingestion by form

Stage 2 and the raster recipe of the thermo-optics playbook. Conventions of the canonical table:

| quantity | x, x_unit | y, y_unit | conditions keys |
|---|---|---|---|
| `shg_d` (d coefficient) or `shg_chi` | `photon_energy`, `eV` (or `wavelength`, `um`) | `pm/V` (3D) or `pm2/V` (sheet) | `component` (xyz, d14, d_eff), `reference` (quartz d11 = 0.30 pm/V, ...), `method`, `sample`, `thickness_nm` (2D), `scissor_ev` (theory) |
| `shift_current` | `photon_energy`, `eV` | `uA/V2` or `nm uA/V2` | `component`, `polarization`, `thickness_nm`, `gap_ev`, `method` (calc / photocurrent) |
| `injection_current` | `photon_energy`, `eV` | `A V-2 s-1` | `component`, `helicity` |

Published calculations are `provenance='calculated'` with their level of theory (`functional`, `scissor_ev`, `grid`, `eta_ev`) in `conditions`.

## Stage 3: validation of the ingestion

Overlays and replots; anchors: the accepted static d of the material (GaAs d_14 ≈ 85–90 pm/V, i.e. χ_xyz ≈ 170 pm/V at 0.12 eV), the gap at the onset, a quoted peak value.

## Stage 4: model at the measured conditions

- `NonlinearModel.value(kind, abc, energy)` at the photon energy of the measurement (below the gap: compare with the static value and state the model's lowest energy); `d_coefficient(chi)` for d_ijk; `effective_3d(kind, abc, thickness_nm)` for 2D literature values quoted per thickness; `tensor(kind, energy)` for d_eff of a measurement geometry (combine components with the stated polarisations).
- **Scissor.** The model's scissor sets the gap; compare at the same gap (rerun with the measured gap if the run used PBE), and say that the scissor keeps the DFT matrix elements.
- **Write every model curve/value used** to `data/model_curves.csv` (`provenance='calculated'`, run parameters in `note`).

## Stage 5: comparison metrics with baselines

| Tool | Use |
|---|---|
| `mt.curve_deviation(x_e, y_e, x_m, y_m, 'linear', window)` | \|χ(ω)\| and σ(ω) spectra per window (below gap / E0–E1 / above) |
| `mt.best_shift` | a rigid photon-energy shift between model and experiment (diagnostic: gap error beyond the scissor) |
| `mt.threshold_crossing`, `onset` | onsets |
| percentages | static values, peak heights, component ratios |
| `symmetry_check`, `kleinman_check` | the symmetry error bar quoted next to every metric |

One row per (comparison, dataset, condition, window) in `tables/metrics.csv`, with `baseline`, `baseline_rms`, `symmetry_error`, `note`.

## Stage 6: diagnosis

Cheap checks first: the symmetry audit (leakage and equivalent-component spread: with the current code the shift current is exact and the SHG carries the degenerate-gauge error bar of `LESSONS_LEARNED.md` #9; runs older than October 2026 on PAW/ultrasoft inputs carry the projection error of #2; a nosym nscf does not help), the degeneracy tolerance (0.1 meV gives spurious peaks; 10 meV default), the projectability vector (a gapless or distorted PAO Hamiltonian → `BASIS_PS`), η vs the feature widths, k-convergence between the available grids (GaAs 40³ vs 60³), the scissor (gap and the renormalised matrix elements), the onset vs the gap (half the gap for SHG), the sign conventions of the field and the current. Then physics outside the model: excitons, local fields, surface SHG, the reference-crystal value behind the experimental d. Then propose runs as `main.py` settings with cost. See the decision tree.

## Stage 7: figures

As in the thermo-optics playbook (`plotting`): model as lines, experiments as markers, published calculations gray dashed; |χ|, Re χ, Im χ on one panel per component; σ(ω) per component with the forbidden ones drawn thin as the leakage; 2D sheet units stated; one axis per panel.

## Stage 8: report

`templates/report_template.md`: caveat box (conventions, scissor, grid, η, tolerance, nosym, basis), the symmetry table, bottom line, inventory, model run, results per comparison, diagnosis, how to reproduce.

---

## Decision tree for a discrepancy

```
model - experiment discrepancy
 ├─ same quantity?             -> d vs chi (factor 2), component / orientation, d_eff geometry, 3D vs sheet (thickness)
 ├─ same photon energy / gap?  -> below-gap static vs the run's lowest energy; scissor to the measured gap; onset vs gap
 ├─ symmetry error bar?        -> forbidden-component leakage, equivalent-component spread, Kleinman: degenerate-gauge error bar of the SHG (lesson #9), old PAW projections (#2), tolerance
 ├─ ingestion error?           -> overlay, units (pm/V vs esu: use the source's own conversion), reference crystal
 ├─ convergence?               -> nk (40^3 -> 60^3 -> 100^3), eta, shg_bin, nbnd / basis (projectability)
 ├─ method?                    -> independent particles (no excitons, no local fields), scissor with DFT matrix elements
 └─ none                       -> report unexplained, with what was excluded and the run or measurement that would decide
```

## Ask the user at the start

1. The output folder(s) and their `main.py` (grid, η, scissor, components, twoD, basis, nosym).
2. The sources, with the quoted quantity, its convention, photon energy and sample.
3. The crystal's point group and orientation (which axis is x), and for 2D the thickness convention of the literature.
4. The deliverable and its audience.
5. The compute budget.

## Worked example

`examples/nlo_examples/REPORT.md`: GaAs (χ_xyz(0) = 110 pm/V at 40³ against 166–180 pm/V measured at 0.12 eV and 207–245 pm/V from TDDFT + scissor; peak 610 pm/V at 2.63 eV; σ_xyz −30 µA/V² at 3.9 eV; symmetry error bar 5 % from the symmetry-expanded Hamiltonian), monolayer GeS (sheet σ^{xxx} 46 nm µA/V², literature 100–150 µA/V² with a thickness convention), gapped graphene (D3h relations to 1e-15).
