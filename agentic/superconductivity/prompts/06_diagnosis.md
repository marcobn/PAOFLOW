# 06: Diagnosis

**When:** the metrics show discrepancies.

```
Diagnose the discrepancies with the decision tree of AGENT_WORKFLOW.md: definitions (gap, Tc, lambda / N_F),
parameters (mu*, the inversion's assumptions), temperatures (bracketing Tc, interpolation), convergence (dense k / q
grids, fixed vs adaptive smearing, N_F, equal grids for the anisotropic run, converged flags), anisotropy (isotropic vs
anisotropic gaps and Tc, sheet assignment), ingestion, missing physics (SOC, anharmonicity, spin fluctuations, Coulomb
beyond mu*). First look for one cause that explains several discrepancies (N_F convergence usually moves lambda, Tc
and the gaps together). Use only existing output for checks (tc(mu_star), the convergence series available, the mode-
resolved lambda_qv, the lambda_nk distribution), then propose runs as exact driver commands of the system folder
(superconductivity/eliashberg.py ... --adaptive-smearing / --nk / --nq, me.py with temperatures around Tc,
me_aniso.py --nk N --nq N) with their cost and the discrepancy each would settle. Do not launch anything.
```

**Check:** each proposed cause is backed by a number from a file; proposed runs have a cost and a decision value.
