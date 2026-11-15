# Agentic superconductivity: α²F, Tc and gaps vs experiment

`PAOFLOW.agentic.superconductivity` compares PAOFLOW's electron-phonon
superconductivity (the `superconductivity/` workflow of an electron-phonon system
folder: `eliashberg.py`, `me.py`, `me_aniso.py`) with experimental data: the
Eliashberg function and its moments, McMillan / Allen-Dynes / Migdal-Eliashberg
critical temperatures, the gap versus temperature and its ratio, the gap and
λ_nk distributions of multi-gap metals, the quasiparticle density of states. The
data toolkit is the one of [thermo-optics](thermo-optics.md).

The agent playbook, lessons learned, prompts and templates live in
`agentic/superconductivity/`.

## Modules

| module | role |
|---|---|
| `eliashberg` | `EliashbergModel`: `superconductivity/output/eliashberg.npz` (`omega` in eV, `a2F`, `lambda`, `omega_log`, `omega_2`, `mu_star`, `Tc_mcmillan`, `Tc_allen_dynes`, `lambda_qv`, with `--transport` also `a2F_tr`, `lambda_tr`, `omega_plasma_ev`): `function(omega, transport)`, `cumulative_lambda`, `moments`, `lambda_in_window`, `tc(mu_star, formula)`, `mode_lambda`, `describe()`; `from_system` |
| `gap` | `GapModel` (`output/me/migdal_eliashberg.npz`): `gaps(kind)` / `gap(T, kind, interpolate)` for `'imag'` (Δ(iω₀)), `'pade'` and `'acon'` (gap edges), in meV; `gap_zero`, `gap_ratio`, `bcs_gap` (the BCS curve scaled to the model), `mass_renormalisation`, `quasiparticle_dos`, `real_axis`, `max_eigenvalue`, `tc` (`'gap'`, `'linear'`); `AnisotropicGapModel` (`output/me_aniso/migdal_eliashberg_aniso.npz`): `gaps(T, kind)`, `gap_range`, `gap_distribution`, `gap_peaks` (the gaps of a multi-gap metal), `sheet_gaps(T, {name: bands})`, `lambda_statistics`, `quasiparticle_dos`, `tc`; both `from_system` |
| `baselines` | `a2f_moments`, `cumulative_lambda`, `mcmillan_tc`, `allen_dynes_tc` (the formulas of `PAOFLOW.elphon.eph_kq.mcmillan_allen_dynes_tc`), `lambda_from_tc` (inversion by bracketing), `BCS_GAP_RATIO`, `bcs_gap_zero_mev`, `bcs_gap_vs_t` (the universal BCS Δ(T) from the gap equation in reduced units, scaled to a given Δ(0) and Tc), `strong_coupling_gap_ratio` (Carbotte's 2Δ/k_BTc correction) |

## Design choices

- **Definitions and parameters are arguments**: the gap kind (order parameter or
  spectral edge), the Tc definition, μ*. `EliashbergModel.tc(mu_star)` gives the μ*
  sensitivity without a run; `lambda_from_tc` gives the coupling a measured Tc implies as a
  diagnostic.
- **Gaps in meV, frequencies in eV** at the interface, as the drivers write them
  (`gap_vs_T.dat` in meV, `eliashberg.npz` in eV).
- **The BCS curve is a baseline, not a model**: it is scaled to each side's Δ(0) and Tc,
  and the strong-coupling ratio replaces 3.53 for strong-coupling metals.
- **Multi-gap metals** are compared through the distribution peaks and sheet averages with
  an explicit band assignment, as point-contact and ARPES data are reported.

## Tests

`tests/unit/agentic/superconductivity/`: moments of an Einstein-like α²F, Tc formulas
against `mcmillan_allen_dynes_tc`, the inversion, the BCS curve against Mühlschlegel's
values (t = 0.5, 0.9), the adapters on synthetic `eliashberg.npz`,
`migdal_eliashberg.npz` (a BCS-shaped Δ(T)) and `migdal_eliashberg_aniso.npz` (two gap
peaks, sheet averages), `from_system`.
