# <Campaign>: plan

*<date>. Goal: <what is computed, for which systems, to what end>. Analysis afterwards: <domain workflow>.*

## Machines

| machine | access | scheduler, partition | billing (unit per core hour, step) | scratch (purge) | permanent | code versions | limits |
|---|---|---|---|---|---|---|---|
| <name> | <ssh alias> | slurm, <partition> | <1.5 SBU, 24 cores> | <path> (<14 d>) | <path> | QE <x>, PAOFLOW <commit> | <shared budget; other campaigns> |

Other campaigns on these machines (not touched): <list>. Locks file: <path or none>.

## Runs and stages

| run | machine | stages (dependencies) | inputs | cores (billed) | expected time per stage | source of the timing |
|---|---|---|---|---|---|---|
| <run> | <machine> | scf → setup → cells → post | <path> | <114 (120)> | <0.8 h, 1.2 h, 0.7 h> | <pilot / similar run> |

## Cost

| stage | runs | billed cores × hours | charge |
|---|---|---|---|
| <stage> | <n> | <...> | <...> |
| **total** | | | **<...> (uncertainty <...>)** |

**Approved budget:** <filled in by the user: amount, unit, date>

## Preflight

| run | findings (`preflight_pw_input`) | fix or reason accepted |
|---|---|---|
| <run> | <codes> | <...> |

## Pilot

<run, job id, measured time per iteration, iterations, memory; re-estimate; go / no-go and date>

## Monitoring and storage

- Watcher: `python -m PAOFLOW.agentic.campaign.watch campaign.py --interval <s>` (background); stop at `error,warning`.
- Results to permanent storage after: <stages>; scratch purge watched.

## Authority overrides

<anything the user allows or forbids beyond ACTIONS, e.g. "do not cancel without asking", "resubmissions within the
approved budget allowed">
