# Writing Tutorials and How-Tos

This page describes the files, release assets and tests behind the tutorials and how-tos of the website. How to write a page (page types, voice, physics depth, figures, notation) is in the skill [`.agents/skills/learn/SKILL.md`](https://github.com/marcobn/PAOFLOW/blob/develop/.agents/skills/learn/SKILL.md) and its templates in `references/`, which developers and coding agents share; the checks at its end report the rules a script can test. The terms (case, base case, run card) are defined in [`CONTEXT.md`](https://github.com/marcobn/PAOFLOW/blob/develop/CONTEXT.md).

## Pages and their base cases

Every tutorial and how-to is built on one case of `examples/cases`, its base case, which supplies the inputs and the data the page starts from; the page holds the explanation and its own workflow, and never edits the case. The page opens with a tip that names the case (Studio: *Examples…* → `<case>`; `paoflow-gen --example <case>`). The tip is the only link between the two: the examples catalogue of the website (`docs/learn/examples.md`, written by `docs/conf.py`) lists each page under the case its tip names, and `tests/unit/gen/test_cases.py::test_pages_name_existing_cases` checks that every tip names an existing case.

- A **tutorial** is a notebook, `docs/learn/tutorials/tutorialNN.ipynb`, with its outputs stored; the notebook is its workflow, and inputs it adds to the case (a `projwfc.x` input) sit in `docs/learn/tutorials/tutorialNN/`. When its tip says that it walks through the `main.py` that `paoflow-gen` and Studio write from the answers of a generator case, its visible code cells make the PAOFLOW calls of that driver. Hidden `hide-input` cells draw the figures and a `remove-cell` cell asserts the numbers the text quotes.
- A **how-to** is a Markdown page, `docs/learn/howtos/<task>.md`, whose code is two scripts beside it. The workflow, `docs/learn/howtos/<task>/<name>.py`, is shown in full on the page with `literalinclude` blocks that start and end at its step comments; the inputs only that page needs (a DFT band run, a `projwfc.x` input) sit beside it. The reader downloads the workflow into a copy of the case, so it is named after the task and never after a file of the case, and the page test does the same. `docs/learn/howtos/<task>/figures.py` reads the outputs of the workflow and its printed output, draws the figures of the page, committed under `docs/_static/images/howtos/<task>/`, and asserts the numbers the text quotes.
- A **theory page**, `docs/learn/theory/<topic>.md`, explains the physics behind a family of results: the equations PAOFLOW evaluates, their derivation and the limits of their approximations. It has no base case and runs nothing.


The docs build does not execute notebooks (`nb_execution_mode = 'off'`), so the website shows the committed outputs and figures.

## Case data

Pages read their DFT data from a GitHub release, never from the repository, because a reader downloads a notebook alone. The data of a case are one archive in the release `case-data`, `<case>-v1.tar.gz`, pinned in `src/PAOFLOW/gen/case_data.json` as `"<case>": {"file": ..., "sha256": ...}`, so that the case folder never changes for the website. One archive serves the case and every page built on it. The archive holds paths relative to the case folder: a trimmed Quantum ESPRESSO `.save` (`data-file-schema.xml`, the pseudopotentials, and `atomic_proj.xml` or the `wfc*.dat` wavefunctions, whichever the driver reads), or, for a page whose run exceeds the budget below, the outputs of the slow step and its log.

- `python .github/assets_generation/cases/build_case_data.py <case> <run folder>` packs the files of a run of the case (the patterns per case are in the script, whose docstring records how each published archive was made) and prints the entry of `case_data.json`.
- `.github/assets_generation/cases/upload_case_data.sh <archive>...` publishes archives and replaces one of the same name. Until the first PAOFLOW release that ships `case_data.json`, a rebuilt archive keeps its name, and its checksum is updated in `case_data.json` in the same commit; after that release, installed copies pin the old checksum, so a changed archive gets the next version.
- `PAOFLOW.gen.cases.case_data_archive` downloads an archive once into `~/.cache/paoflow/case-data` and checks its checksum; `fetch_case_data` extracts it into a copy of the case. `paoflow-gen --example <case> --with-data` and Studio's *Use the precomputed data of the case* call them, and the plan of the copy then leaves out the QE steps. An archive placed in the cache by hand is used without a download, which is how a page is tested before its data are published.
- A tutorial's setup cell downloads the archive itself, because a downloaded notebook runs without a checkout; `test_tutorial_follows_its_case` checks that the cell names the pinned file.

Case data never go into the `integration-assets-vN` releases, which hold the inputs and references of the integration tests. The release `tutorial-assets-v1` stays published for older notebooks, unused by the current pages.

## The page test

`tests/integration/docs/test_pages.py` has two parts:

- `test_tutorial_follows_its_case` compares the PAOFLOW calls of the visible cells of each tutorial whose tip says it walks through the generated driver, methods and literal arguments in order, with those of that driver. Arguments the driver builds at run time (`savedir`, `workpath`, `basispath`) are not compared. It runs nothing and needs no data.
- `test_page_runs` runs every tutorial notebook (with `nbclient`) that is not in `UNWRITTEN`, and every how-to in a copy of its case made by `paoflow-gen --example <case> --with-data`: the workflow script the page shows, copied in from beside the page (after the generated `main.py` on a generator case), with its printed output in `<script>.log`, then the figures script of the page in that folder. A case fails when a call raises or an assert of the page fails, so it catches renamed methods and arguments as well as numbers on a page that the code no longer gives. It does not compare figures or stored outputs.

```bash
pytest tests/integration/docs -m "integration and not slow" --durations=0
pytest tests/integration/docs -k plotting
```

The CI job `docs-page-tests` in `.github/workflows/ci.yaml` runs one page per job on every pull request into `develop`, plus one job (`follows`) for the comparison of every tutorial with its driver. Its matrix lists the pages by name, so a new page needs a new matrix entry; the test itself collects new pages by their file names.

## Budgets

- A page runs in at most 5 minutes on a 4-core runner. For a heavier how-to, its data hold the output files and the log of its workflow, and its name goes into the `PRECOMPUTED` set of the page test: the pull-request case runs only the figures script on those files, and the case `<task>-workflow`, which runs the workflow, is marked `slow`.
- The data of a case are at most 10 MB compressed, one archive for the case and all its pages.
- Figures are PNGs at 150 dpi, at most 200 KB each.
- A notebook is re-executed and committed only when its text or one of its results changes, because every re-execution adds its full outputs to the history of the repository.
