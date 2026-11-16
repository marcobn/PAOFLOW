# 02: Plan

```
Write PLAN.md from templates/plan_template.md: one row per run (machine, stage chain, dependencies), resources per
stage (cores as a multiple of the billing granularity, pools balanced over k-points with balanced_pools, memory, time
limit with a 1.5x margin, max_seconds for long pw.x runs), the cost per stage and in total with the source of every
timing, the monitoring cadence and the storage plan. If the most expensive run has no measured timing, the plan
starts with its pilot. Ask me to approve the budget.
```

**Check:** every cost is billed cores × time × charge with its source; the approval line is empty until I fill it.
