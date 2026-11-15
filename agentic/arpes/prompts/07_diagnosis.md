# 07: Diagnosis

**When:** the metrics show discrepancies.

```
Diagnose the discrepancies with the decision tree of AGENT_WORKFLOW.md: bare band, alignment (k_F, E_F, segment),
resolution and method, state (cut, k_z, band, isotropic vs anisotropic Sigma), ingestion (colour scale, calibration),
convergence of Sigma (Fan-Migdal q-grid and widths, alpha^2F grids and smearing), missing physics (impurity Im Sigma(0),
e-e, Debye-Waller, k_z broadening, surface states, SOC). First look for one cause that explains several discrepancies.
Use only existing output for checks (other windows and methods on the same maps, the sigma_*.dat columns, the
anisotropic states, the Fan-Migdal widths versus the isotropic plateau), then propose runs as exact driver commands
(arpes/arpes.py ... after renaming output/; superconductivity/eliashberg.py --fs-coupling) with their cost and the
discrepancy each would settle. Do not launch anything.
```

**Check:** each proposed cause is backed by a number from a file; proposed runs have a cost and a decision value.
