# 00: Kickoff (new study)

**When:** first message of a new comparison.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/arpes/ folder, then start the ARPES workflow
for <SYSTEM / PROPERTY> (e.g. "the electron-phonon kink of Pb along Gamma-X", "phonon linewidths of Si vs T").

Model (read-only): an electron-phonon system folder (examples/elphon/<System> layout)
- system folder: <path>; runs to compare: arpes/<output folder(s)>; self-energies present: <isotropic | anisotropic |
  fan-migdal>; older runs: <names or "none">
- what it computes and its known limits: <e.g. isotropic Sigma from alpha^2F at 10 K on a segment across k_F, 2 meV
  resolution, Fan term only, scalar-relativistic bands>
Experiment:
- sources: <paths: PDFs, images, tables, decks>; the figures / tables that matter: <e.g. "Fig. 2(b) band map, Fig. 3 Re Sigma">
- provenance: <raw | transcribed | digitised | extracted by the authors (with their bare band: <...>)>
- measurement conditions: <photon energy, polarisation, temperature, energy / momentum resolution, cut and k_z, surface,
  doping>
Study folder (write only here): <projects/<study>/ or a path next to the calculation>
Deliverable and audience: <paper figures | internal check | collaborator package>, written for <audience>
Compute budget: <cores, may Fan-Migdal / dense-q runs be launched?>

Do Stage 0 (intake) and Stage 1 (physics map) only: inventory the model output (describe(): cut, points, photon energy,
polarisation, resolution, window, self-energies and temperatures, Fan-Migdal q-grid and widths) and every source (form
per figure; conditions; the authors' bare band and k_F). Propose the comparison table: observable <-> model quantity <->
alignment (k - k_F, E_F offset, bare-band protocol, resolution) <-> space <-> windows (below / above the kink) <-> baseline.
List your questions. Do not ingest yet.
```

**Check:** every source has a form, a provenance and the measurement conditions; the bare-band protocol and the k_F alignment are settled before any comparison; out-of-scope physics (impurity Im Σ(0), Debye-Waller shifts, k_z, SOC) is flagged.
