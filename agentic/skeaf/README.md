# SKEAF agentic workflow

This workflow compares PAOFLOW pyskeaf / SKEAF extremal-orbit calculations with measured quantum oscillations (dHvA, SdH, torque). The experimental data can be tables, free text or images, such as FFT-vs-angle maps or slide decks. An AI agent drives the analysis through the `PAOFLOW.qo_analysis` library and follows the playbook in this folder. You steer it with the prompts.

## Before you start

- **Install PAOFLOW with the extra:** `pip install -e ".[qo_analysis]"` from the repository root.
- **Have a finished pyskeaf rotation:** a folder with `qo_EF_*_freqvsangle_*.out` files, its BXSF files, and the pyskeaf driver script.
- **Have the experimental data** in any form, and know its provenance (raw, transcribed or digitised).

## Start a study

```bash
cd PAOFLOW/agentic/skeaf
mkdir -p projects/<system>/data      # put (copies of) the experimental files here
claude                               # or another agent; AGENTS.md / CLAUDE.md load automatically
```

Then paste `prompts/00_kickoff.md` with its `<...>` fields filled in. The agent will:
1. write `projects/<system>/analysis.py` from `templates/analysis_template.py`;
2. run the first stages;
3. report back with questions.

Continue with the prompts listed in `prompts/README.md`. To resume in a later session, use `prompts/15_resume_session.md`.

The calculation folders are only read: outputs go to `projects/<system>/`, and new pyskeaf runs go to new folders.

## Contents

| Path | What |
|---|---|
| `AGENTS.md`, `CLAUDE.md` | instructions the agent loads when opened here |
| `AGENT_WORKFLOW.md` | the playbook: operating rules, Stages 0–10, decision tree |
| `LESSONS_LEARNED.md` | 32 pitfalls (symptom → cause → fix) from the NbSn2 analysis |
| `prompts/` | 17 prompts: kickoff; experimental data as table, text or image; sweep and assignments; convergence; E_F scan; SOC; breakdown; weak peaks; collaborator package; review; resume; collaborator questions |
| `templates/` | analysis script and canonical CSVs (LK table, sweep peaks, branch assignments) |
| `projects/` | your studies, one folder per system (ignored by git) |

## Worked example

`../../examples/qo_analysis_examples/NbSn2/` is the complete NbSn2 analysis on bundled data:
- `main.py` (SOC) and `main_sr.py` (scalar-relativistic);
- `expected_results.json`.

Run `python main.py` there to see every stage end to end without an agent.
