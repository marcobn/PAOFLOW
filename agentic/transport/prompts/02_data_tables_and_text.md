# 02: Experimental data as tables or text

**When:** CSV/XLSX files, tables printed in a paper, or numbers quoted in the text.

```
Ingest <file / paper table N / sentence> into the canonical format (quantities and condition keys of
AGENT_WORKFLOW.md Stage 2: mobility_drift_e / mobility_hall_h / resistivity / carrier_density / seebeck /
emissivity_total, conditions doping_cm3, carrier, method, hall_factor, rho0_uohm_cm, rrr, sample): read it
(ingest.read_table) or transcribe it with the sentence in note; keep the printed units; provenance 'measured'
or 'transcribed'; y_rel_uncertainty from the source. Write <study>/data/<source>.csv with write_dataset and a
header naming the file/table/page, the sample and any correction. Print datasets.summarise and 3 rows next to
the source lines.
```

**Check:** compare the printed rows with the source yourself; the sample conditions are in `conditions`, not lost in `note`.
