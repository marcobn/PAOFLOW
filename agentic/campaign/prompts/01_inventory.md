# 01: Machine inventory

```
For every machine of the campaign: check access (Machine.shell), the scheduler and the queue (Machine.queue), the
accounting of the last day (Machine.accounting), the billing granularity (a test submission is not needed: ask or read
the site documentation), scratch and permanent paths and free space, the purge age, the environment line of a job,
the installed codes and their versions (QE build, PAOFLOW commit: git log -1), and the load of shared workstations.
Write the table into PLAN.md (machines section). Change nothing.
```

**Check:** the code version of every machine is recorded; an unreachable machine is reported as unknown.
