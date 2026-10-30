# 00: Kickoff (new study)

**When:** first message of a new comparison.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/transport/ folder, then start the transport
workflow for <SYSTEM / PROPERTY> (e.g. "phonon-limited mobility of Si vs T", "resistivity and emissivity of Pb").

Model (read-only): an electron-phonon system folder (examples/elphon/<System> layout)
- system folder: <path>; runs to compare: transport/<tag> (mobility.dat or resistivity.dat) and
  superconductivity/output (eliashberg.npz); older runs: <names or "none">
- what it computes and its known limits: <e.g. phonon-limited SERTA/MRTA drift mobilities, intrinsic statistics,
  dipole-only long-range model; or rho(T) from alpha^2F_tr, scalar-relativistic, no residual>
Experiment:
- sources: <paths: PDFs, images, tables, decks>; the figures / tables that matter: <e.g. "Jacoboni 1977 Fig. 2">
- provenance: <raw | transcribed | replotted in a later paper>
- sample and method: <doping and carrier type, purity (rho0 / RRR), Hall or drift, geometry, temperature range>
Study folder (write only here): <projects/<study>/ or a path next to the calculation>
Deliverable and audience: <paper figures | internal check | collaborator package>, written for <audience>
Compute budget: <cores, may new runs be launched (dense vertex, Boltzmann with extra channels)?>

Do Stage 0 (intake) and Stage 1 (physics map) only: inventory the model output (describe(): grids, smearing,
doping, gap/scissor, long-range and screening flags, channels, lambda_tr, omega_p, temperatures) and every source
(form per page/figure; stated sample conditions). Propose the comparison table: observable <-> model quantity <->
conditions to match (T window, doping, Hall factor, rho0) <-> space <-> windows <-> baseline. List your questions.
Do not ingest yet.
```

**Check:** every source has a form, a provenance and the sample conditions; drift vs Hall and ρ₀ are settled before any comparison; out-of-scope physics (impurities, films, phonon drag, SOC) is flagged.
