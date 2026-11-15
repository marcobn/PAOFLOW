# 07: Diagnosis

**When:** the metrics show discrepancies.

```
Diagnose the discrepancies with the decision tree of AGENT_WORKFLOW.md: observable (drift/Hall, sigma/mu, rho0),
sample (doping, purity, film), conditions (T window, interpolation), ingestion, convergence (grids, smearing,
window states, Planck coverage), model choices (SERTA/MRTA, long-range and quadrupoles, screening, electrons
provenance, SOC, Drude model), missing physics. First look for one cause that explains several discrepancies.
Use only existing output for checks (both approximations in mobility.dat, the metadata of carrier_scattering.npz,
the convergence series available, the per-channel columns of resistivity.dat, the eliashberg.npz scalars), then the
recomputations that need no vertex run (transport/resistivity.py --rho0/--b-ee, a Boltzmann run with extra channels).
Rank the causes with evidence (files, numbers), then propose new runs as exact driver commands of the system folder
(transport/mobility.py ... --tag, superconductivity/eliashberg.py --transport ..., transport/resistivity.py ...), with
their cost estimate and the discrepancy each would settle. Do not launch anything.
```

**Check:** each proposed cause is backed by a number from a file; proposed runs have a cost and a decision value.
