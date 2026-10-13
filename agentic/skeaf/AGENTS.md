# Agent instructions: SKEAF agentic workflow

You are running the SKEAF agentic workflow: interpreting PAOFLOW pyskeaf / SKEAF extremal-orbit calculations against experimental quantum-oscillation data.

1. **Read before acting:**
   - `AGENT_WORKFLOW.md` (operating rules, Stages 0–10, decision tree);
   - `LESSONS_LEARNED.md` (pitfalls).
   Follow them; do not re-derive the method.
2. **Use the library** `PAOFLOW.qo_analysis` (class `QuantumOscillationAnalysis`) from an analysis script. Write the script as `projects/<system>/analysis.py`, starting from `templates/analysis_template.py`. Inputs are Python arguments; do not use config files. The worked example is `../../examples/qo_analysis_examples/NbSn2/main.py`.
3. **Where to write:**
   - Write only inside `projects/<system>/`. Never write into calculation folders.
   - Do not modify `src/PAOFLOW/` or the files in this folder unless the user asks.
   - If the library needs a fix, propose it with a test.
4. **Non-negotiable:**
   - never overwrite pyskeaf output;
   - ask before heavy computations (new pyskeaf runs, `breakdown_network`), and smoke-test one angle first;
   - quote every number with its file;
   - give chance baselines with every match rate;
   - assign branches from their shape over the rotation;
   - label digitised or transcribed experimental numbers as such;
   - correct earlier statements explicitly.
5. **Units:**
   - BXSF energies in Ry;
   - pyskeaf `fermi_energy` in eV;
   - BXSF reciprocal vectors in a.u.⁻¹ without 2π;
   - SKEAF frequencies in kT.
6. **Prompts.** The user steers you with `prompts/NN_*.md`. If they paste one, do what it asks and report what its **Check** line names.
