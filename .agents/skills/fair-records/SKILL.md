---
name: fair-records
description: "What a code change owes the run record (paoflow_outputs.json) and the paoflow-pack package. Use when: adding or renaming an output file, adding a workflow function outside PAOFLOW.py / Transport.py that writes files, reading a new kind of input file from disk, adding an argument that changes a unit, or editing the recorder, the record schema, the results catalogue vocabulary or gen/pack.py."
user-invocable: true
---

# FAIR records

Every run writes a run record and `paoflow-pack` turns a case into an RO-Crate zip, with no action
from the user. That only stays true if each change pays what it owes below. The design, the record
keys and the status table live in `docs/internals/fair-records.md`; read it before changing the
machinery itself.

## Your change → what it owes

| Your change | What it owes, in the same change | The test that proves it |
|---|---|---|
| A new or renamed output file | A `Kind` in `KINDS` (`src/PAOFLOW/results/catalogue.py`) with real units on `x` and `y`, a one-sentence definition in `TERMS` (`src/PAOFLOW/results/vocabulary.py`, with QUDT / PhySH links where a shared term exists), and the regenerated table in `docs/studio/run-records.md` (`python -m PAOFLOW.results.vocabulary`, pasted between the `vocabulary` markers). A file that is not a result (a log, a scratch table) goes in `SKIPPED_FILES` instead | `test_every_term_is_defined_and_documented`; `uncatalogued_files` in the QE and transport integration tests |
| A workflow function outside `PAOFLOW.py` / `Transport.py` that writes files | One `record_function_call(output_path, name, arguments, inputs=[...])` on the way out, after the files are written (the electron-phonon writers in `src/PAOFLOW/elphon/workflows/` are the pattern). Methods of the two orchestrators are recorded by `@recorded` and owe nothing | `test_plain_function_steps_share_one_entry` is the model for yours |
| A new kind of input file read from disk | Its pattern in `_INPUT_PATTERNS` (`src/PAOFLOW/writers/output_manifest.py`) so `stamp_inputs` hashes it, a regex in `_VERSIONS` when the file states the version of the code that wrote it, and a pattern in `EXTRA_FILES` (`src/PAOFLOW/gen/pack.py`) when the package should carry the file | `tests/unit/gen/test_pack.py` |
| An argument that changes the unit of an output | The unit reaches `axes` of the file in `run_record`, the way the `units` argument does for the frequency axis of the phonon files | `test_frequency_unit_comes_from_the_call` is the model |
| An argument of a new type (not a scalar, list, dict or array) | A branch in `encode_argument` when the value can be stored exactly; otherwise accept that the run becomes `replayable: false` | `test_run_record_arguments_and_failures` |
| A key of the record, the schema, the crate or the vocabulary | The section below | `_validate` (schema), `test_crate_passes_the_validator` |

## What stays fixed

Published packages carry these, so they hold from the merge into `develop` onwards:

- A published term keeps its meaning. A changed layout or definition gets a new term name.
- Record keys are only added. Removing or renaming one bumps `VERSION` and updates
  `run_record.schema.json` and the key table of `fair-records.md` in the same change.
- The recorder observes: arguments, results, exceptions and output files pass through unchanged,
  and a failure to write the record prints one line and the run continues.
- Privacy: the record and the package hold relative paths only, and no hostname, user name,
  cluster or partition name, or scheduler job ID. Check every new record key and every new
  `EXTRA_FILES` pattern against this; a leak ships in a published zip.

## Changing the machinery

Read `docs/internals/fair-records.md` first and update its key table and status table with the
change. `roc-validator` (in the `dev` extra) must be installed, or the crate test skips silently.

## How to check

```
pytest -q tests/unit/results/test_results.py tests/unit/gen/test_pack.py -rs
pytest tests/integration/qe tests/integration/transport -m "integration and not slow"
```

Done when both pass and `-rs` lists no skip for `jsonschema` or `roc_validator`.
