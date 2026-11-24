# Writing Tutorials and How-Tos

This page describes the files, release assets and tests behind the tutorials and how-tos of the website. How to write a page (page types, voice, physics depth, figures, notation) is in the skill [`.agents/skills/learn/SKILL.md`](https://github.com/marcobn/PAOFLOW/blob/develop/.agents/skills/learn/SKILL.md) and its templates in `references/`, which developers and coding agents share; the checks at its end report the rules a script can test. The terms (case, base case, run card) are defined in [`CONTEXT.md`](https://github.com/marcobn/PAOFLOW/blob/develop/CONTEXT.md).

## Pages and their base cases

Every tutorial and how-to is built on one case of `examples/cases`, its base case, which holds the inputs and the code the page explains; the page holds the explanation. The page opens with a tip that names the case (Studio: *Examples…* → `<case>`; `paoflow-gen --example <case>`), and the `docs` list of the case's `case.json` names the pages built on it. `tests/unit/gen/test_cases.py::test_case_docs_match_the_pages` checks the two against each other in both directions. Studio, `paoflow-gen --example` and the website therefore point at one copy of each workflow.

- A **tutorial** is a notebook, `docs/learn/tutorials/tutorialNN.ipynb`, with its outputs stored. Its visible code cells make the PAOFLOW calls of its case's driver: the `main.py` that `paoflow-gen` and Studio write from the answers of a generator case, or the `main.py` of a scripts case when the tutorial needs code the generator cannot write. Hidden `hide-input` cells draw the figures and a `remove-cell` cell asserts the numbers the text quotes.
- A **how-to** is a Markdown page, `docs/learn/howtos/<task>.md`, whose code is two scripts: `main.py` of its base case, the workflow, shown in full on the page with `literalinclude` blocks that start and end at its step comments, and `docs/learn/howtos/<task>/figures.py`, kept beside the page, which reads the outputs of `main.py` and its printed output (`main.log`), draws the figures of the page, committed under `docs/_static/images/howtos/<task>/`, and asserts the numbers the text quotes. When several pages share a case, a page's workflow is `<name>.py`. The workflow is a step of the case, so Studio runs it; the figures script is not, so a case holds no code that only serves the website.
- A **theory page**, `docs/learn/theory/<topic>.md`, explains the physics behind a family of results: the equations PAOFLOW evaluates, their derivation and the limits of their approximations. It has no base case and runs nothing.

The README of each base case is its run card (template `references/case.md` of the skill): the steps with their run time, the outputs, and the pages that explain it.

The docs build does not execute notebooks (`nb_execution_mode = 'off'`), so the website shows the committed outputs and figures.

## Release assets

Pages read their DFT data from a GitHub release, never from the repository, because a reader downloads a notebook alone. Until the `case-data` release (one archive per case, pinned in `case.json`) replaces them, the pages read `tutorial-assets-v2`:

- `tutorial_assets.tar.gz`: one folder per tutorial, `tutorialNN/`, with trimmed Quantum ESPRESSO `.save` directories (`atomic_proj.xml`, `data-file-schema.xml`, pseudopotentials) and the QE inputs that made them. `python .github/assets_generation/tutorials/build_assets.py` packs it and `.github/assets_generation/tutorials/upload_release_assets.sh <tag>` publishes it.
- `howto_assets.tar.gz`: one folder per how-to that needs data, `<task>/`. It holds DFT data, or, for a page whose run exceeds the budget below, the precomputed outputs the page reads. `.github/assets_generation/howtos/build_assets.py` packs it; its docstring records how every folder was made, so that a later developer can remake outputs that go stale when the code changes.

The `DATA` table of the page test extracts a page's folder of these archives into the copy of its case, at the paths the case's own runs write. A published archive is never replaced, because readers and CI pin its checksum: a new payload goes under the next tag, and the tag in the tip of every page and the `ASSETS` entries of `tests/integration/docs/test_pages.py` follow it. Tutorial data never go into the `integration-assets-vN` releases, which hold the inputs and references of the integration tests.

## The page test

`tests/integration/docs/test_pages.py` has two parts:

- `test_tutorial_follows_its_case` compares the PAOFLOW calls of each tutorial's visible cells, methods and literal arguments in order, with those of its case's driver. Arguments the driver builds at run time (`savedir`, `workpath`, `basispath`) are not compared. It runs nothing and needs no data.
- `test_page_runs` runs every tutorial notebook (with `nbclient`) that is not in `UNWRITTEN`, and every how-to in a copy of its case: `python main.py > main.log`, then the figures script of the page in that folder. A case fails when a call raises or an assert of the page fails, so it catches renamed methods and arguments as well as numbers on a page that the code no longer gives. It does not compare figures or stored outputs.

```bash
pytest tests/integration/docs -m "integration and not slow" --durations=0
pytest tests/integration/docs -k plotting
```

The CI job `docs-page-tests` in `.github/workflows/ci.yaml` runs one page per job on every pull request into `develop`, plus one job (`follows`) for the comparison of every tutorial with its driver. Its matrix lists the pages by name, so a new page needs a new matrix entry; the test itself collects new pages by their file names.

## Budgets

- A page runs in at most 5 minutes on a 4-core runner. For a heavier how-to, its data hold the output files and the log of its workflow, and its name goes into the `PRECOMPUTED` set of the page test: the pull-request case runs only the figures script on those files, and the case `<task>-workflow`, which runs the workflow, is marked `slow`.
- The input of a page is at most 5 MB compressed.
- Figures are PNGs at 150 dpi, at most 200 KB each.
- A notebook is re-executed and committed only when its text or one of its results changes, because every re-execution adds its full outputs to the history of the repository.
