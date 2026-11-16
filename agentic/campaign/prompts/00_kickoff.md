# 00: Kickoff (new campaign)

**When:** first message of a new campaign.

```
Read AGENT_WORKFLOW.md and LESSONS_LEARNED.md in PAOFLOW's agentic/campaign/ folder, then start the campaign workflow
for <GOAL> (e.g. "thermal emissivity of <systems> at <temperatures> on <cluster>", "electron-phonon production of <N>
systems").

Systems and runs: <list, or where the inputs are>
Scripts that build and submit the runs: <drivers / production workflow / restage scripts, with paths>
Machines: <name: ssh alias, scheduler, partitions, billing (unit, per core hour, granularity), scratch and permanent
paths, purge age, environment line; workstation limits (ranks, nice)>
Budget: <amount and unit>; shared with: <who>; may jobs be launched without asking: <no / within the approved plan>
Other campaigns on these machines (do not touch): <list>; locks file: <path or none>
Analysis afterwards: <domain workflow (thermo_optics, transport, ...) or none>
Study folder (write only here and in the run directories): projects/<study>/
Deliverable and audience: <...>, written for <audience>

Do stage 0 (intake): check what you can on the machines (Machine.shell), list the inventory and your questions.
Do not launch anything.
```

**Check:** every machine has its billing, scratch purge and code version stated; the other campaigns are listed.
