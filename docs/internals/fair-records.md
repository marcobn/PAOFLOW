# Run records and packages for sharing

Every PAOFLOW run writes a run record, `paoflow_outputs.json`, into its output folder. The aim is
that a user gets FAIR data (findable, accessible, interoperable, reusable) for any calculation
without doing anything for it, and can share a calculation with one command. The drivers are the
data-availability statements of journals and funders, and the user's own reproducibility.

There is no flag: the record is always written. Nothing is uploaded anywhere; the mechanism ends at
a zip file a user uploads by hand.

| Word | Meaning |
|---|---|
| Run record | `paoflow_outputs.json`, one per output folder, one entry per run |
| Case | One calculation as a folder: DFT inputs, driver script, output folders, `case.json` |
| Package | The zip that `paoflow-pack` makes from a case |
| Term | One entry of the results catalogue (`bands`, `dos`, ...): the name of an output type |

## What touches the orchestrators

This is the part a maintainer of `PAOFLOW.py` and `Transport.py` needs to know.

- One class decorator, `@recorded` (`src/PAOFLOW/writers/output_manifest.py`), sits above
  `@sparse_aware` on `PAOFLOW` and on `Transport`. It wraps the constructor and every public
  method. No method body changes, and no method or argument is added, so no sparse role changes.
- The wrapper only observes. It passes arguments and results through unchanged, keeps name,
  docstring, signature and the sparse role of each method (`functools.wraps`), and re-raises any
  exception of the method. Output files are not touched: units travel in the record, not in file
  headers.
- Only rank 0 records, and only the calls the script makes. A public method called from another
  one (the constructor's `memory_check`, `report_module_time`, the sparse engine calling the
  host) is not recorded.
- Because the decorator is the outermost one, a call is recorded once, before it is routed to
  the dense or the sparse engine. `call_dense` (`src/PAOFLOW/sparse/dispatch.py`) now unwraps
  with `inspect.unwrap` to reach the dense body through both wrappers.
- After each recorded call rank 0 rewrites the record (read, replace the entry of this run,
  write to a temporary file, rename). The cost is one walk of the output folder; hashing happens
  once for the inputs and once at the end for the outputs.
- A failure while writing the record prints one line and never stops the run.
- `finish_execution()` no longer writes the record itself; its wrapper marks the run finished.
- The state of a run (identifier, calls) lives on the `DataController`, so a `Transport` built
  on the data controller of a `PAOFLOW` object continues the same entry.

## The record, version 2

Version 2 only adds keys to version 1; Studio reads the keys generically.

| Key | Content |
|---|---|
| `run_id` | UUID of the run |
| `status` | `unfinished` until `finish_execution()` returns, then `finished`. A killed or crashed script leaves `unfinished` |
| `started_utc`, `finished_utc` | UTC times (the local-time `started`, `finished` remain) |
| `inputs` | DFT engine, its full version string, the functional, cell and positions (bohr), species with pseudopotential names, and size and sha256 of the XML files and pseudopotentials of the save folder |
| `script` | Path and sha256 of the driver script |
| `calls` | The method calls in order: name, resolved arguments (defaults included), seconds, peak memory of rank 0, `ok` or `failed` |
| `replayable` | False when an argument could only be described (below) |
| `environment` | Versions of Python, numpy, scipy, mpi4py; whether the Rust kernels load |
| `infrastructure` | MPI ranks, threads per rank, nodes and cores (Slurm or PBS variables when set), CPU model, total memory, BLAS library of numpy |
| `files[*]` | As before (path, size, term, labels), listed recursively, plus `axes` (quantity and unit from the catalogue) and, once the run finished, `sha256` |

Rules:

- Arguments: scalars, lists and dicts are stored as they are; an array as shape, dtype and
  sha256; an object (a `TauModel`) as class name and `repr`; a callable as its qualified name.
  The last three make the run "described, not replayable from the record alone".
- Hashing: files up to 50 MB (`HASH_LIMIT`, a first guess) are hashed by the run. Larger files
  are recorded by size and modification time and hashed by `paoflow-pack`.
- Privacy: paths are relative (inputs to the working folder, outputs to the output folder). The
  record holds no hostname, user name, cluster or partition name, or scheduler job ID. An
  absolute path given as an argument is made relative to the current folder, or cut to its last
  name when it lies outside.
- The record is an output. Nothing reads it back as the configuration of a run.
- There is no cap on the number of runs in one record.

## Vocabulary

The results catalogue (`src/PAOFLOW/results/catalogue.py`) is PAOFLOW's controlled vocabulary.
Each `Kind` gains a one-sentence definition, real units where they are empty today, and optional
links to shared terms. A published term keeps its meaning; a changed definition gets a new term.

A local term identifies a file layout (columns, units, energy reference, how spin and tensor
components split across files). A shared term identifies the physical concept. Both are kept.

| Role | Vocabulary | Address pattern |
|---|---|---|
| Quantity and unit on each axis | QUDT | `https://qudt.org/vocab/quantitykind/<Name>` |
| What the dataset is about | PhySH | `https://doi.org/10.29172/<uuid>` |
| Crystal structure | OPTIMADE property definitions | `https://schemas.optimade.org/defs/v1.2/properties/optimade/structures/<name>` |

Terms with no shared counterpart stay local with their definition. NOMAD's dataset-level terms
are added when their addresses resolve (they did not on 2026-11-19).

## `paoflow-pack` and `paoflow-unpack`

- `paoflow-pack` works in any folder that holds a run record and needs no arguments. It asks,
  in plain words, what only a person knows: title, authors (with ORCID), licence (CC-BY-4.0 on
  Enter), and optionally the name of the machine. The answers go to `case.json` (new keys
  `authors`, `license`) and can be given as command-line arguments.
- It computes the hashes the runs deferred and writes, at the top of the package,
  `ro-crate-metadata.json` (Workflow Run Crate when a driver script exists, Process Run Crate
  otherwise; each term a `DefinedTerm` with `sameAs` links) and a `README.md` for people.
- The package holds outputs, driver script, DFT inputs, pseudopotentials, `figures.json`, the
  coupling-cache records and the campaign journal. Wavefunctions are referenced by hash and not
  included. Studio's `jobs.json` is excluded.
- A checklist in ordinary words reports what is missing. It never blocks; the zip is always
  written.
- `paoflow-unpack <zip>` checks every hash, lays out the case folder and prints the steps to
  reproduce. It does not re-run anything.
- Neither command contains network code.

## Studio

A richer "Run information" entry, a "from this run" line under each plot, and a "Package for
sharing" button that calls the code of `paoflow-pack`. The record never depends on Studio.

## Status

| Part | State |
|---|---|
| Recorder, incremental writes, record version 2 for `PAOFLOW` and `Transport` | Done |
| Records for the electron-phonon writers and the `pyskeaf` command line (`record_function_call`, one explicit call each) | Done |
| Library tight-binding runs (`models/_paoflow_runner.py`) | Covered by the recorder; they stay `unfinished` because they return the object without `finish_execution()` |
| JSON Schema (`writers/run_record.schema.json`), definitions and shared-term links (`results/vocabulary.py`), user page (`docs/studio/run-records.md`) | Done |
| Units that are empty in the catalogue | Open: each needs a check against its writer (the k-path axis is a point index and has no unit) |
| Versions and hashes of the phonon and electron-phonon inputs (ph.x, phonopy, EPW) | Planned |
| `paoflow-pack`, `paoflow-unpack` | Planned |
| Studio | Planned |

Out of scope: upload to any repository, `CITATION.cff` and a DOI for the code, a flag to switch
the record off, headers in `.dat` files or new output formats, a command that re-runs a case and
compares numbers.
