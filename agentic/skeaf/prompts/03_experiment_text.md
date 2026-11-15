# 03: Experimental data from free text

```
The experimental information is free text: <paste text or give file / slide numbers>.
Use experiment_data.parse_text_peaks as a first pass, then check EVERY row against the text by hand and write
<project folder>/data/exp_LK_table.csv with a 'source' entry quoting the original line. Normalise Greek letters to
names. List what could not be parsed unambiguously (missing units, ranges, '?', uncertain labels) and ask me.
Do not invent masses or errors that are not in the text.
```

**Check:** uncertain entries (e.g. "11?") are flagged in `source`; nothing was silently rounded.
