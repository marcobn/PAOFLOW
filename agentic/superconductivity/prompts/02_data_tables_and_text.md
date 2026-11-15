# 02: Tc, gaps, λ, ratios and α²F as tables or text

**When:** numbers printed in tables or quoted in the text.

```
Ingest <file / paper table N / sentence> into the canonical format (quantities tc / gap / gap_ratio / lambda /
omega_log / a2f / tunnelling_dos; conditions method, definition, sheet, mu_star, lambda_quoted, sample): read it
(ingest.read_table) or transcribe it with the sentence in note; keep the printed units; provenance 'measured' or
'transcribed'; y_rel_uncertainty from the source. Write <study>/data/<source>.csv with write_dataset and a header
naming the file/table/page, the method, the definitions and any inversion parameter. Print datasets.summarise and
3 rows next to the source lines.
```

**Check:** compare the printed rows with the source yourself; definitions and μ* are in `conditions`, not lost in `note`.
