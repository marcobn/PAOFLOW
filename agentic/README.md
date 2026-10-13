# PAOFLOW agentic workflows

Each folder here is a self-contained workflow that an AI coding agent (Claude Code, Codex, or similar) runs on top of PAOFLOW's library code. The library code lives in `src/PAOFLOW/`. The workflows only import it, and their files are never installed with the package.

| Folder | Workflow | Library code |
|---|---|---|
| `skeaf/` | Interpreting pyskeaf / SKEAF extremal-orbit calculations against experimental quantum-oscillation data (dHvA, SdH, torque) given as tables, text or images | `PAOFLOW.qo_analysis` |

How to start any workflow:
1. Go to its folder.
2. Read its `README.md`.
3. Open the agent there.
4. Paste the kickoff prompt.

Each workflow folder contains:
- `AGENTS.md` / `CLAUDE.md`: agent instructions, loaded automatically when the agent is opened in the folder;
- the playbook and lessons learned;
- a `prompts/` library;
- `templates/`;
- a `projects/` area, ignored by git, for your own studies.
