# 00: Kickoff (new study)

**When:** first message of a new comparison.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/nlo/ folder, then start the non-linear optics
workflow for <SYSTEM / PROPERTY> (e.g. "SHG and shift current of GaAs", "sheet shift current of monolayer GeS").

Model (read-only): an output folder of PAOFLOW.nonlinear_optics (examples/nlo_examples/<System> layout)
- folder: <path>; outputs to compare: output/<label> ...; main.py settings: <nk, eta, scissor, components, twoD,
  degeneracy_tol, basis>; nscf with nosym: <yes/no>; older runs: <names or "none">
- what it computes and its known limits: <independent particles with a scissor, no excitons / local fields, ...>
Crystal: point group <name and axis convention, e.g. -43m; mm2 with the polar axis x>, orientation of x, y, z;
for 2D the slab height of the run and the thickness convention of the literature.
Experiment / literature:
- sources: <paths>; the figures / tables that matter; the quoted quantities and their conventions (d vs chi,
  component, photon energy, reference crystal, 3D vs sheet, scissor and functional of calculations)
Study folder (write only here): <projects/<study>/ or a path next to the calculation>
Deliverable and audience: <...>, written for <audience>
Compute budget: <cores, may dense-grid reruns be launched?>

Do Stage 0 (intake), the symmetry audit (forbidden components and their leakage, equivalent components, Kleinman at
w -> 0, T-odd kinds) and Stage 1 (physics map): source quantity (convention) <-> model quantity <-> photon energy /
component / 3D-or-sheet <-> space <-> baseline. List your questions. Do not ingest yet.
```

**Check:** every source has a form, a provenance and a stated convention; the symmetry audit numbers are quoted; out-of-scope physics is flagged.
