# Scattering channels: relaxation rates for transport and optics

`PAOFLOW.boltzmann.scattering_channels` provides one abstraction for relaxation
rates. It serves two consumers:
- **Boltzmann transport.** `PAOFLOW.transport(scattering_channels=[...])`
  combines the channels through `boltzmann.do_Boltz_tensors.get_tau`.
- **Intraband optics of metals.** These are the Drude and extended-Drude
  dielectric functions of `PAOFLOW.response.drude`, and the thermal emissivity
  of `PAOFLOW.response.emissivity`.

Every channel returns a **rate** $1/\tau$ (1/s) from a single method:

```python
channel.rate(temperature_k, energy_ev=None, omega_ev=None)
```

- **DC (transport) rate:** `energy_ev` is the carrier energy relative to the
  chemical potential.
- **Optical rate:** `omega_ev` is the photon energy.
- **Matthiessen's rule:** `matthiessen(channels, T, ...)` sums the rates.

## Channels

| class | rate | use |
|---|---|---|
| `ConstantRate(rate_per_s)` | constant | impurities, defects |
| `ResidualResistivity(rho0_ohm_m, plasma_ev)` | $\varepsilon_0\omega_p^2\rho_0$ | metals, residual resistivity |
| `FermiLiquidRate(b_per_ev)` | $\hbar/\tau = b[(E-\mu)^2+(\pi k_BT)^2]$ (DC); $b[(\hbar\omega)^2+(2\pi k_BT)^2]$ (optical) | electron–electron scattering (Gurzhi) |
| `PhenomenologicalChannel(model, params, weight)` | $w/\tau_{model}(k_BT, \lvert E\rvert)$ | the semiconductor models of `do_tau_models` (`'acoustic'`, `'optical'`, `'polar_optical'`, `'polar_acoustic'`, `'impurity'`), a `TauModel`, or any `f(kT_eV, \|E\|_eV, params) -> tau` |
| `TransportSpectralChannel(omega_ev, a2f_tr)` / `.from_npz('eliashberg.npz')` | Allen (below) | electron–phonon scattering from the dense-q e-ph run with `transport=True` |

## Electron–phonon channel: transport Eliashberg function

`eliashberg_dense_q(..., transport=True, alat=, cell_volume=)` accumulates two
extra quantities next to $\lambda_{\mathbf q\nu}$:
- the transport coupling, with the trace form of Allen's weight (invariant under
  the irreducible-q folding);
- the Fermi-surface plasma frequency.

The transport coupling is

$$
\alpha^2F_{tr}(\omega) = \frac{1}{N_F\langle v^2\rangle}\sum_{\mathbf q\nu}\frac{w_{\mathbf q}}{N_k}
\sum_{\mathbf k mn}|g_{mn\nu}|^2\,\mathbf v_{n\mathbf k}\cdot(\mathbf v_{n\mathbf k}-\mathbf v_{m\mathbf k+\mathbf q})
\,\delta(\varepsilon_{n\mathbf k})\delta(\varepsilon_{m\mathbf k+\mathbf q})\,\delta(\omega-\omega_{\mathbf q\nu}).
$$

The plasma frequency is

$$
\omega_{p,\alpha\beta}^2 = \frac{4\pi e^2 g_s}{\Omega N_k}\sum_{n\mathbf k}v^\alpha_{n\mathbf k}v^\beta_{n\mathbf k}\,\delta(\varepsilon_{n\mathbf k}-\mu).
$$

The band velocities are the PAO velocity operator
$\partial_{\mathbf k}H + iH(\boldsymbol\tau_j-\boldsymbol\tau_i)$, optionally
with the non-local pseudopotential term. They are taken on the diagonal and
averaged over degenerate subspaces, which makes them gauge invariant.

The channel derives three quantities from $\alpha^2F_{tr}$:
- **DC rate** (high T: $2\pi\lambda_{tr}k_BT/\hbar$), with $x=\hbar\omega/2k_BT$:
  $1/\tau_{tr}(T) = \frac{4\pi k_BT}{\hbar}\int\frac{d\omega}{\omega}\alpha^2F_{tr}\,[x/\sinh x]^2$.
- **Optical rate** (Allen 1971; Shulga, Dolgov and Maksimov 1991):
  $1/\tau(\omega,T) = \frac{\pi}{\hbar\omega}\int d\Omega\,\alpha^2F_{tr}(\Omega)
  \big[2\omega\coth\frac{\Omega}{2T}-(\omega+\Omega)\coth\frac{\omega+\Omega}{2T}+(\omega-\Omega)\coth\frac{\omega-\Omega}{2T}\big]$.
  Its ω→0 limit is the DC rate.
- **Mass enhancement** $\lambda(\omega,T)$, from causality of the memory function
  $M = 1/\tau(\omega) - i\omega\lambda(\omega)$:
  $\lambda(\omega) = \frac{2}{\pi}\int_0^\infty\frac{r(\omega')-r(\omega)}{\omega'^2-\omega^2}d\omega'$,
  with $r=\hbar/\tau$ and $\lambda(0,T{=}0)=\lambda_{tr}$.

## Consumers

**Boltzmann transport.** `get_tau` accepts channels mixed with the old string
labels and `TauModel`s. The rates enter the same harmonic sum, with the band
energies relative to $E_F$. For metals:

```python
from PAOFLOW.boltzmann.scattering_channels import TransportSpectralChannel, ResidualResistivity
eph = TransportSpectralChannel.from_npz('output/eliashberg.npz')
pf.transport(tmin=50, tmax=600, nt=12, emin=-0.2, emax=0.2, ne=41,
             scattering_channels=[eph, ResidualResistivity(1e-9, plasma_ev=8.0)])
```

**Drude optics, resistivity and emissivity** (`response/drude.py`,
`response/emissivity.py`):
- `drude_dielectric` and `extended_drude_dielectric` give the intraband
  dielectric function:
  $\varepsilon_D = -\omega_p^2/\{\omega[\omega(1+\lambda(\omega,T)) + i/\tau(\omega,T)]\}$.
- `dc_resistivity` gives $\rho = 1/(\varepsilon_0\omega_p^2\tau)$, broken down
  by channel.
- `plasma_frequency_tensor` gives $\omega_p^2$ from band velocities on any grid.
- `metal_thermal_emissivity` combines the interband ε with the extended Drude
  term. The interband ε comes from `dielectric_tensor(..., intraband=False)` on
  the extended basis.
- `emissivity_from_dielectric` is the shared slab/hemispherical/Planck tail,
  also used by the semiconductor `thermal_emissivity`.

The generated `transport.elphon.py` (paoflow-gen, EPW source, Eliashberg
property, "transport Eliashberg function") chains these steps for a metal.

## Fixes made with this module

- **`scattering_weights`:** non-empty weights used to raise, because the
  length check compared them with an empty list.
- **`tau_dict` and `doping_conc`:** values passed to `transport()` / `doping()`
  now replace stored ones. Previously only the first call counted.
- **`dielectric_tensor(d_tensor=None)`:** now computes the diagonal components;
  it used to crash.
- **Documentation:** the built-in τ models receive $k_BT$ in eV, not kelvin.

## Validation (Pb, 48³ k / 24³ q, σ = 0.05 eV, non-local velocity)

| quantity | PAOFLOW | reference |
|---|---|---|
| $\lambda$ / $\lambda_{tr}$ | 1.20 / 0.94 | |
| $\hbar\omega_p$ | 8.0 ± 0.2 eV (48³–80³ k) | 7.4 eV (optical Drude fit, Ordal *et al.* 1985) |
| $\rho$(300 K) | 17.1 µΩ cm (Drude); 19.3 µΩ cm (Boltzmann at the same $\mu$) | about 21 µΩ cm (experiment) |
| total emissivity, 300 / 600 K | 0.037 / 0.072 | polished Pb: a few % |

- **k-grid.** The Fermi-surface quantities of Pb need at least 48³ k at
  σ = 0.05 eV. Smaller smearings need denser grids.
- **q-grid.** Refining q from 24³ to 48³ changes λ_tr by −7% (0.94 → 0.87) and
  ρ(300 K) from 17.1 to 16.0 µΩ cm.
