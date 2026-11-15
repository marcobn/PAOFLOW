# 00: Kickoff (new system)

**When:** first message of a new analysis.

```
Read the agent playbook AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/skeaf/ folder, then start the SKEAF agentic workflow for <SYSTEM> (<space group / structure, magnetic or not, SOC or not>).

Calculation:
- pyskeaf/SKEAF output: <path to folder with qo_EF_*_freqvsangle_*.out>
- BXSF files: <path> ; driver script: <path to main_skeaf.py> ; cell: <scf.in path or vectors + centring>
- older / reference runs (do not overwrite): <paths or "none">
Experiment:
- data: <paths>; form: <table | text | image | pptx/pdf>; provenance: <raw | transcribed | digitised>
- directions measured / rotation: <e.g. rotation [001] -> [110] in 5 deg steps; LK fits at 0, 45, 90 deg>
- angle convention: <field normal to (hkl) or along [hkl]; how the rotation angle is defined>
Goal: <assign the measured branches / test a hypothesis / prepare a package for collaborators>
Compute budget: <cores, max minutes per run without asking>.

Write the analysis script projects/<system>/analysis.py from templates/analysis_template.py (inputs as Python arguments, no config
files), run Stage 0 (fermi_surface_inventory) and Stage 1 (field_directions), and report: run parameters, bands crossing E_F, degenerate groups,
Fermi-surface pieces, NumOrbCopy histogram, direction <-> angle table, and questions for me.
```

**Check:** no band crossing E_F is missing; the convention (real vs reciprocal) is stated; nothing was written into the calculation folders.
