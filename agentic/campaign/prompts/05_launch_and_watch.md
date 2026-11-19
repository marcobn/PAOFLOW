# 05: Launch and watch

```
Launch the approved runs of PLAN.md with the campaign's scripts (stage chains with afterok dependencies); journal the
job ids. Write projects/<study>/campaign.py from templates/campaign_template.py (one CampaignRun per run: job_prefix,
pw_outputs, logs, results; approved_budget), test it with `python -m PAOFLOW.agentic.campaign.watch campaign.py
--once`, then start the watcher in the background with --interval <seconds>. Report what is queued and running.
```

**Check:** the --once poll lists every run; the watcher runs in the background from the study folder.
