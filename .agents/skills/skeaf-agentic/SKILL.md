---
name: skeaf-agentic
description: "Interpret PAOFLOW pyskeaf (SKEAF) quantum-oscillation calculations against experimental dHvA/SdH/torque data given as tables, text or images. Use when: assigning measured frequency branches to Fermi-surface sheets, digitising FFT-vs-angle figures, testing convergence / Fermi-level / SOC effects on extremal orbits, analysing magnetic breakdown, or preparing a results package for experimental collaborators."
user-invocable: true
---

# SKEAF agentic workflow

Use the `PAOFLOW.agentic.qo_analysis` subpackage and follow its playbook. Do not re-derive it.

1. **Read first:**
   - `agentic/skeaf/AGENT_WORKFLOW.md`: operating rules, Stages 0–10, decision tree.
   - `agentic/skeaf/LESSONS_LEARNED.md`: pitfalls (symptom → cause → fix).
   - The prompt library for users is in `agentic/skeaf/prompts/`.
2. **Write an analysis script** from `agentic/skeaf/templates/analysis_template.py` into `agentic/skeaf/projects/<system>/`. Inputs are Python arguments (no config files). The worked example is `examples/qo_analysis_examples/NbSn2/main.py`.
3. **Non-negotiable rules:**
   - never overwrite pyskeaf output folders;
   - ask before heavy computations (new pyskeaf runs, `breakdown_network`) and smoke-test one angle first;
   - quote every number with its file;
   - give the chance baselines for every match rate;
   - assign branches from their shape over a rotation;
   - label digitised or transcribed experimental numbers as such;
   - correct earlier statements explicitly.
4. **Units:** BXSF energies in Ry, pyskeaf `fermi_energy` in eV, BXSF reciprocal vectors in a.u.⁻¹ without 2π, SKEAF frequencies in kT.
5. **Tests:** `pytest tests/unit/agentic/qo_analysis` (fast, synthetic data with known answers) and `pytest -m integration tests/integration/agentic/qo_analysis` (NbSn2 example, about 2 minutes).
