# 09: Diagnosis

**When:** the metrics show discrepancies.

```
Diagnose the discrepancies with the decision tree of AGENT_WORKFLOW.md: geometry, conditions, ingestion, grid limits,
global offsets, method choices, missing physics. First look for one cause that explains several discrepancies.
Use only existing output for checks (stored parameter sensitivities, component splits, bracketing temperatures,
alternative definitions of the observable, SERTA vs MRTA), then the recomputations that need no kernel run
(optics/emissivity.py for geometry, external dielectric function or direct scissor). Rank the causes with evidence
(files, numbers), then propose new runs as exact driver commands of the system folder (optics/absorption.py ...,
optics/dielectric.py, each with --outdir), with their cost estimate and the
discrepancy each would settle. Do not launch anything.
```

**Check:** each proposed cause is backed by a number from a file; proposed runs have a cost and a decision value.
