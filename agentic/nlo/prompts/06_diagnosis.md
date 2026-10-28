# 06: Diagnosis

```
Diagnose the discrepancies with the decision tree of AGENT_WORKFLOW.md: quantity and convention, photon energy and gap,
symmetry error bar (nosym, tolerance), ingestion, convergence (nk, eta, shg_bin, basis projectability), method
(independent particles, scissor). Use existing output first (the symmetry table, other grids available, the
projectability vector, the forbidden-component leakage), then propose reruns as main.py settings (NK, ETA, SCISSOR,
components, twoD, nosym nscf, BASIS_PS) into new output_<label> folders, each with cost and the discrepancy it would
settle. Do not launch anything.
```

**Check:** each cause is backed by a number from a file; proposed runs have a cost and a decision value.
