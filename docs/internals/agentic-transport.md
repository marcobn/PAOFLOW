# Agentic transport: mobilities, resistivity and emissivity vs experiment

`PAOFLOW.agentic.transport` compares PAOFLOW's electron-phonon transport with
experimental data: the phonon-limited carrier transport of semiconductors
(`transport/mobility.py` of an electron-phonon system folder) and the transport
of metals from the transport Eliashberg function (`transport/resistivity.py`),
plus the bridge that feeds the same scattering channels, with impurity and
residual terms, to `PAOFLOW.transport(scattering_channels=...)`. The data
toolkit (canonical datasets with provenance, units, PDF/raster ingestion,
digitising, metrics, publication plots) is the one of
[thermo-optics](thermo-optics.md), `PAOFLOW.agentic.thermo_optics`.

The agent playbook, lessons learned, prompts and templates live outside the
package, in `agentic/transport/`.

## Modules

| module | role |
|---|---|
| `mobility` | `MobilityModel`: the adapter for `transport/<tag>/mobility.dat` and `carrier_scattering.npz` (`elphon.carrier_scattering`): `mobility(T, carrier, kind, interpolate)` (SERTA / MRTA drift mobilities, ln μ linear in ln T off the grid), `carrier_density` (ln n linear in 1/T), `conductivity`, `seebeck`, `power_law_exponent`, `bracket`, `metadata` (grids, smearing, doping, gap, scissor mode, electrons provenance, screening, window states), `describe()`; `from_system(folder, output=)` |
| `resistivity` | `ResistivityModel`: the adapter for `transport/<tag>/resistivity.dat` (total and per-channel ρ in µΩ cm, channels named in the header), `emissivity.npz` (extended-Drude ε(ω, T), hemispherical / directional / total emissivity, Planck coverage) and the `eliashberg.npz` scalars (λ, λ_tr, ω_p, ω_log, Tc): `resistivity(T, channel, rho0_uohm_cm, interpolate, law)` (Matthiessen: a residual resistivity is added after the fact), `slope`, `residual_resistivity_ratio`, `emissivity_total`, `emissivity_spectrum`, `dielectric`, `describe()`; `from_system` |
| `baselines` | `bloch_gruneisen`, `fit_bloch_gruneisen` (ρ₀, A, θ; a coarse θ scan + bounded refinement, θ or ρ₀ can be fixed), `fit_power_law`, `fit_linear`, `matthiessen_offset` (the residual that reconciles a phonon-limited model with a sample: rms before/after, slope ratio), `matthiessen_mobility` |
| `channels` | `metal_channels(eliashberg_npz, rho0_uohm_cm, b_ee_per_ev)` (the list `transport/resistivity.py` builds: `TransportSpectralChannel` + `ResidualResistivity` + `FermiLiquidRate`), `carrier_channels(scattering_rates_npz, kind, band_edges, extra)` (`EnergyResolvedRate` aligned to band edges + extra channels), `impurity_rate_from_mobility`, `impurity_channel` (a `ConstantRate` reproducing a mobility) |

`PAOFLOW.agentic.thermo_optics.paoflow_transport` remains as an alias of
`PAOFLOW.agentic.transport.mobility`.

## Design choices

- **Adapters are thin and read-only** and return the *observable* at a condition;
  interpolation laws are physical (power law for μ, activated for n, linear for ρ
  above θ_D/5, power law below) and every off-grid value is meant to be labelled.
- **Corrections are explicit arguments.** The Hall factor, the residual resistivity and
  Matthiessen impurity mobilities are quoted next to the uncorrected comparison; the
  library offers them as diagnostics (`matthiessen_offset`, `matthiessen_mobility`), not as
  fitted models.
- **Baselines are trivial models with named parameters** (Bloch-Grüneisen, power laws),
  so a deviation is always quoted against something.
- **The Boltzmann bridge reuses the run outputs**: no re-derivation of the vertex is needed
  to add impurity or residual scattering to σ, Seebeck and κ_e.

## Tests

`tests/unit/agentic/transport/`:
- `MobilityModel` on a synthetic `transport/output` (power-law and activated interpolation,
  exponents, `from_system`, the alias);
- `ResistivityModel` on a synthetic metal output (channel parsing, interpolation laws, residual
  addition, slope, RRR, emissivity spectra and totals, dielectric function, `from_system`);
- Bloch-Grüneisen limits and the recovery of (ρ₀, A, θ) from a synthetic curve; power-law,
  linear and Matthiessen-offset fits; channel lists from synthetic `eliashberg.npz` and
  `scattering_rates.npz` files, impurity rates.
