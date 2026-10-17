# 02: Experimental data as tables

**When:** CSV/XLSX files or tables printed in a paper.

```
Ingest <file / paper table N> into the canonical format: read it (ingest.read_table, or transcribe), map the columns to
dataset/quantity/x/x_unit/y/y_unit/temperature_k/conditions, keep the printed units (convert later with
thermo_optics.units), provenance 'measured' (raw files) or 'transcribed', y_rel_uncertainty from the source.
Write <study>/data/<source>.csv with write_dataset and a header naming the file/table, page and any correction.
Print the dataset summary (datasets.summarise) and 3 rows next to the source lines.
```

**Check:** compare the printed rows with the source yourself; units and conditions are in the right columns.
