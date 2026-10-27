# 00: Kickoff (new study)

**When:** first message of a new comparison.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/thermo_optics/ folder, then start the
thermo-optics workflow for <SYSTEM / PROPERTY> (e.g. "optical absorption and emissivity of Si vs T").

Model (read-only): an electron-phonon system folder (examples/elphon/<System> layout)
- system folder: <path>; runs to compare: optics/<outdir>; older runs: <names or "none">
- what it computes and its known limits: <e.g. phonon-assisted + direct absorption, no excitons, 0.05 eV grid>
Experiment:
- sources: <paths: PDFs, images, tables, decks>; the figures / tables that matter: <e.g. "Sato Fig. 3, 5, 8, 9">
- provenance: <raw | transcribed | replotted in a later paper>; sample and conditions if known
Study folder (write only here): <projects/<study>/ or a path next to the calculation>
Deliverable and audience: <paper figures | internal check | collaborator package>, written for <audience>
Compute budget: <cores, may new runs be launched?>

Do Stage 0 (intake) and Stage 1 (physics map) only: inventory the model output (parameters, grids, temperatures,
units, missing physics) and every source (form per page/figure: vector, embedded bitmap, scan, table, text; stated
conditions). Propose the comparison table: observable <-> model quantity <-> conditions to match <-> space <-> windows
<-> baseline. List your questions. Do not digitise yet.
```

**Check:** every source has a form and a provenance; conditions (T, thickness, doping, geometry) are listed; out-of-scope physics is flagged before any comparison.
