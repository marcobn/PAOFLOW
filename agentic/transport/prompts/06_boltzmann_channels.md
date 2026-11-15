# 06: Boltzmann transport with extra scattering channels

**When:** the comparison needs sigma(T), the Seebeck coefficient or kappa_e with impurity / residual / e-e channels that the phonon-limited run lacks, and the user allows a PAOFLOW run.

```
Build the channel list from the run outputs with PAOFLOW.agentic.transport.channels:
  metals:          metal_channels('<superconductivity/output/eliashberg.npz>', rho0_uohm_cm=<rho0>, b_ee_per_ev=<b>)
  semiconductors:  carrier_channels('<transport/<tag>/scattering_rates.npz>', kind='mrta', band_edges=<(vbm, cbm) or None>,
                                    extra=[impurity_channel(<mu_imp cm2/Vs>, <m_eff>)])
Write <study>/boltzmann/main.py (the standard PAOFLOW workflow: read_atomic_proj_QE, projectability, pao_hamiltonian,
interpolated_hamiltonian(nfft=<grid>), pao_eigh, gradient_and_momenta, adaptive_smearing, dos, transport(tmin, tmax, tstep,
scattering_channels=<list>), finish_execution), state its cost, and ask before running it. Treat the output as a new
model run: a small read-only adapter for sigma*.dat / Seebeck*.dat / kappa*.dat in the study, curves to
data/model_curves.csv with the channels named in note, metrics with the same baselines.
```

**Check:** the channel parameters (rho0, b, mu_imp, m_eff) are quoted with their sources; the run was approved; the channel list is printed in the log.
