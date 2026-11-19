# 00: Kickoff (new study)

**When:** first message of a new SSCHA study.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/sscha/ folder, then start the SSCHA workflow for
<SYSTEM / QUESTION> (e.g. "is rocksalt TaN stable at 2800 K, and above which temperature?", "force constants of
<X> at 300, 600, 900 K for the thermal cells").

Calculation inputs:
- scf of the primitive cell: <path to <prefix>.save>, pseudopotentials <dir>, cutoffs, k-grid, occupations
  (smearing type and width)
- harmonic phonons: <ph.x run (fildyn prefix, q-grid) or "none yet">
- metal or insulator: <...>; electrons of the force runs: <parent / Fermi-Dirac at T / undecided>
Machine: <workstation / Slurm cluster (partition, cores per node, account) / other>; environment line for jobs:
<...>; scratch <path>, permanent storage <path>
Budget: <core-hours>; may runs be launched without asking: <no / up to N core-hours>
Experiment (optional): <temperatures with what they mean, measured phonons; sources>
Study folder (write only here): <projects/<study>/>
Deliverable and audience: <...>, written for <audience>

Do Stage 0 (intake): the harmonic picture (fine-mesh softest mode and dispersion of the ph.x run, the electrons it
used), the machine and budget, the experimental references. List your questions. Do not launch anything.
```

**Check:** the harmonic instabilities are quoted with q, frequency and the electrons of the run; the budget and the machine are stated.
