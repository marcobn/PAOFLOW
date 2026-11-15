# Agentic ARPES: band maps, kinks and linewidths vs experiment

`PAOFLOW.agentic.arpes` compares PAOFLOW's simulated photoemission (the `arpes/`
workflow of an electron-phonon system folder: `arpes/arpes.py` over
[PAOFLOW.arpes](arpes.md) and `elphon.self_energy`) with measured ARPES data:
dispersions from MDC fits, momentum widths, self-energies and λ extracted by the
authors, linewidths versus temperature, and band-map images. The data toolkit is
the one of [thermo-optics](thermo-optics.md).

The agent playbook, lessons learned, prompts and templates live in
`agentic/arpes/`.

## Modules

| module | role |
|---|---|
| `curves` | array-level ARPES analysis, shared by model and measured maps: `refine_peak`, `peak_position`, `half_maximum_width` (parabolic peak height, inverse quadratic crossings; grid-limited), `lorentzian`, `fit_lorentzian`, `mdc_dispersion(k, w, intensity, energy_window, k_window, method)` (the dressed dispersion k(ω) and the momentum FWHM), `edc_dispersion`, `fermi_momentum`, `linear_band` (the bare-band protocol E = v (k − k_F) over a window), `self_energy_from_dispersion` (Re Σ = ω − ε_b(k(ω)), Im Σ = −Δk\|v_b\|/2 against a callable or tabulated bare band), `kink_energy`, `mass_enhancement_from_velocities` |
| `model` | `ArpesModel`: `arpes/<output>/` with `arpes_bare.npz`, `arpes_<kind>_<T>K.npz` (the `IntensityMap.save` layout; `metadata` parsed back), `sigma_<kind>_<T>K.dat` (isotropic, or anisotropic with one column pair per state) and `self_energy_fan_migdal.npz`: `map`, `band_energies`, `bare_dispersion`, `bare_fermi_momentum`, `dispersion`, `fermi_momentum`, `velocities` (bare and dressed v_F, λ), `self_energy_from_map`, `self_energy`, `mass_enhancement`, `kink_energy`, `linewidths(T)`, `linewidth_vs_temperature(band, ik)`, `state_self_energy`, `mdc`, `edc`, `describe()`; `from_system` |
| `images` | `band_map_from_image(image, k_axis, energy_axis, rows, cols, channel, dark_is_high)`: a calibrated bitmap (`thermo_optics.digitize.Axis` on both axes) as `(k, energy_ev, intensity)` in the model layout, so the same MDC analysis applies |

## Design choices

- **One analysis for both sides.** The model maps are analysed as an experimentalist
  analyses the measured ones (MDC peaks and widths, a linear bare band over a stated window),
  because λ and Σ depend on that protocol more than on the data; the model's exact bare band
  is extra information.
- **Alignment at k_F** on each side, never by a hand shift; E_F offsets are diagnostics.
- **Momenta on the path distance** (1/Å), energies relative to E_F in eV; the `'peak'`
  method is cheap and grid-limited (about five points per MDC width), the `'lorentzian'`
  method is the accurate one for widths.
- **Fan-Migdal linewidths** are read on shell (−Im Σ_nk(ε_nk)) for every path state and
  temperature, the quantity semiconductor photoemission reports.

## Tests

`tests/unit/agentic/arpes/`: Lorentzian peaks and widths; MDC/EDC dispersions and
widths on a synthetic spectral function with a known Σ (Re Σ = −λω, constant Im Σ), the
self-energy and λ recovered; `ArpesModel` on a synthetic `arpes/output` (maps, sigma
tables, Fan-Migdal widths versus T, `from_system`, metadata parsing); a drawn band in a
synthetic image recovered through the axis calibration.
