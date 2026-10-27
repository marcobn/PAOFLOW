# 00: Kickoff (new study)

**When:** first message of a new comparison.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/superconductivity/ folder, then start the
superconductivity workflow for <SYSTEM / PROPERTY> (e.g. "Pb: alpha^2F, Tc and the gap vs tunnelling",
"MgB2: the two gaps vs point-contact and ARPES").

Model (read-only): an electron-phonon system folder (examples/elphon/<System> layout)
- system folder: <path>; runs to compare: superconductivity/<output folders> (eliashberg.npz, me/, me_aniso/);
  older runs: <names or "none">
- what it computes and its known limits: <e.g. alpha^2F at 48^3/24^3 with 50 meV smearing, mu* = 0.1, isotropic ME
  0.3-6 K; scalar-relativistic, harmonic phonons>
Experiment:
- sources: <paths: PDFs, images, tables, decks>; the figures / tables that matter: <e.g. "McMillan-Rowell Fig. 3 alpha^2F">
- provenance: <raw | transcribed | digitised | inverted (tunnelling alpha^2F with mu* = ...)>
- measurement: <method (tunnelling, ARPES, STM, point contact, specific heat, muSR, resistive Tc), sample, temperature,
  gap and Tc definitions>
Study folder (write only here): <projects/<study>/ or a path next to the calculation>
Deliverable and audience: <paper figures | internal check | collaborator package>, written for <audience>
Compute budget: <cores, may dense Eliashberg / anisotropic ME runs be launched?>

Do Stage 0 (intake) and Stage 1 (physics map) only: inventory the model output (describe(): grids, smearing, N_F,
mu*, lambda, omega_log, Tc by formula, ME temperatures and convergence flags, anisotropic sheets) and every source
(form; method; definitions; the inversion's mu*). Propose the comparison table: observable <-> model quantity <->
definition / mu* / temperature <-> space <-> baseline (EPW on the same grids, BCS curve, Tc-inverted lambda, a second
measurement). List your questions. Do not ingest yet.
```

**Check:** every source has a form, a provenance and its definitions; μ* and the gap / Tc definitions are settled before any comparison; out-of-scope physics (SOC, anharmonicity, spin fluctuations) is flagged.
