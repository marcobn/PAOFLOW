# Writing Tutorials and How-Tos

This page describes the files, release assets and tests behind the tutorials and how-tos of the website. How to write a page (page types, voice, physics depth, figures, notation) is in the skill [`.agents/skills/learn/SKILL.md`](https://github.com/marcobn/PAOFLOW/blob/master/.agents/skills/learn/SKILL.md) and its templates in `references/`, which developers and coding agents share; the checks at its end report the rules a script can test.

## Page types and files

A tutorial teaches one workflow on one fixed system and is a notebook, `docs/learn/tutorials/tutorialNN.ipynb`, with its outputs stored. A how-to shows a reader with their own calculation how to make one change, and is a Markdown page, `docs/learn/howtos/<task>.md`, with two scripts in `docs/learn/howtos/<task>/`. `main.py` is the workflow the page teaches: the page shows all of it, pulled with `literalinclude` in blocks that start and end at its step comments, so a reader who downloads it gets exactly the code the page explains. `figures.py` reads the output files of `main.py` and its printed output, saved as `main.log`, draws the figures of the page, which are committed under `docs/_static/images/howtos/<task>/`, and asserts the numbers the text quotes. A tutorial keeps the same parts in one notebook: hidden `hide-input` plotting cells and a `remove-cell` cell of asserts.

A theory page, `docs/learn/theory/<topic>.md`, explains the physics behind a family of results: the equations PAOFLOW evaluates, their derivation and the limits of their approximations. It has no script and no run. Tutorials and how-tos keep only the equations the reader needs to run the workflow and read its output (how-tos none) and link to the theory pages for the rest; the Learn section (`docs/learn/index.md`) holds the three page types.

The docs build does not execute notebooks (`nb_execution_mode = 'off'`), so the website shows the committed outputs and figures.

## Release assets

Pages read their data from the GitHub release `tutorial-assets-vN`, never from the repository, because a reader downloads a notebook or a script alone. The release holds two archives:

- `tutorial_assets.tar.gz`: one folder per tutorial, `tutorialNN/`, with trimmed Quantum ESPRESSO `.save` directories (`atomic_proj.xml`, `data-file-schema.xml`, pseudopotentials). The data of `.github/assets_generation/tutorials/tutorialNN/` are packed by `python .github/assets_generation/tutorials/build_assets.py` and published by `.github/assets_generation/tutorials/upload_release_assets.sh <tag>`.
- `howto_assets.tar.gz`: one folder per how-to that needs data, `<task>/`. It holds DFT inputs, or, for a page whose run exceeds the budget below, the precomputed PAOFLOW outputs the page reads. `.github/assets_generation/howtos/build_assets.py` packs it; its docstring records how every folder was made, so that a later developer can remake outputs that go stale when the code changes.

A published archive is never replaced, because readers and CI pin its checksum. When the payload changes, publish it under the next tag, update the tag in the tip of every page that downloads from it, and update the tag and the `SHA256` entries in `tests/integration/docs/test_pages.py`. PAOFLOW releases record the compatible asset tag in their release notes (see [Release](release.md)). Tutorial data never go into the `integration-assets-vN` releases, which hold the inputs and references of the integration tests.

## The page test

`tests/integration/docs/test_pages.py` runs every tutorial notebook (with `nbclient`) and every how-to (`python main.py > main.log`, then `python figures.py`) in a scratch directory that holds the page's release asset, downloaded once into the test-asset cache and checked against its SHA256. A case fails when a call raises or an assert of the page fails, so it catches renamed methods and arguments as well as numbers on a page that the code no longer gives. It does not compare figures or stored outputs.

```bash
pytest tests/integration/docs -m "integration and not slow" --durations=0
pytest tests/integration/docs -k plotting
```

The CI job `docs-page-tests` in `.github/workflows/ci.yaml` runs one page per job on every pull request into `develop`. Its matrix lists the pages by name, so a new page needs a new matrix entry; the test itself collects new pages by their file names.

## Budgets

- A page runs in at most 5 minutes on a 4-core runner. For a heavier how-to, the asset folder `<task>/` holds the output files and `main.log` of `main.py`, and its name goes into the `PRECOMPUTED` set of the page test: the pull-request case runs only `figures.py` on those files, and the case `<task>-workflow`, which runs `main.py`, is marked `slow`.
- The input of a page is at most 5 MB compressed.
- Figures are PNGs at 150 dpi, at most 200 KB each.
- A notebook is re-executed and committed only when its text or one of its results changes, because every re-execution adds its full outputs to the history of the repository.
