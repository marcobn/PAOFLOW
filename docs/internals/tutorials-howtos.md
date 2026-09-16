# Writing Tutorials and How-Tos

PAOFLOW tutorials and how-tos are published as executable Jupyter notebooks.
Write them for readers arriving from the documentation website, without a
repository checkout or a particular working directory.

## Choose the Document Type

Use a **tutorial** to teach a complete, cumulative workflow from input data to
scientific interpretation. Use a **how-to** to solve one focused task for a
reader who already knows the basic PAOFLOW workflow. Put API inventories in the
reference documentation and extended derivations in explanatory pages.

Start from `docs/tutorials/tutorial-template.ipynb` or
`docs/tutorials/howtos/how-to-template.ipynb`. Copy and rename the appropriate
template; the templates themselves are excluded from the website build.

## Write for Website Readers

Every published notebook must be self-contained from the reader's point of
view:

- Add a Sphinx download link near the top using
  ``{download}`Download this notebook <tutorialNN.ipynb>` ``. For a how-to,
  use its own notebook filename. Sphinx copies the source notebook into the
  built site and generates a working download URL.
- Link to other documentation pages with document-relative links, not paths
  that only make sense in a repository checkout.
- Never tell readers to run from `docs/tutorials/` or refer to inputs through
  `.github/`, `examples/`, or another repository-relative path.
- Download required inputs into the notebook's current directory, or explain
  clearly how to select an existing local input supplied by the reader.
- Use public HTTPS URLs and identify the input's provenance before the first
  code cell.
- State near the top whether the example runs easily on a laptop or requires
  HPC resources, and explain what the asset archive provides as a result.

## Declare Input Data

The opening section must state:

1. Which calculation produced the input, including the material, code,
   exchange-correlation approximation, pseudopotential family, and relevant
   spin treatment.
2. A link to the public `tutorial_assets.tar.gz` release asset.
3. Every required file and its purpose.
4. The approximate download size and runtime when material.
5. Whether the asset is owned by this tutorial or reused from an earlier one.

Numbered tutorials own reusable assets. Later tutorials and how-tos should link
to the owning tutorial and reuse its public URL instead of duplicating data.

## Choose the Asset Payload

Tutorial payloads are generated and published as a release archive; they are
never committed to Git.

- For a tutorial that runs quickly on a laptop, provide only the trimmed
  Quantum ESPRESSO `.save` directory. The reader must generate all PAOFLOW
  output by running the notebook.
- For a tutorial that requires HPC resources, provide both the trimmed `.save`
  directory and every PAOFLOW output file used by plotting cells. Plot the
  packaged files by default, but keep the full generation commands in the
  notebook and explain how to run them on HPC.

Put generated data under
`.github/assets_generation/tutorials/tutorialNN/`. Each tutorial directory must
contain at least one QE `.save` directory. Add an `output/` directory only when
the tutorial requires HPC resources and its plotting cells read those
precomputed PAOFLOW files.

## Download Inputs Portably

Use only the Python standard library for setup so that the input cell works in
a fresh notebook environment. Download the release archive and extract only
the owning tutorial's prefix:

```python
from pathlib import Path
from tarfile import open as open_tar
from urllib.request import urlretrieve

asset_url = (
  'https://github.com/marcobn/PAOFLOW/releases/download/'
  'tutorial-assets-v1/tutorial_assets.tar.gz'
)
archive_path = Path('tutorial_assets.tar.gz')
asset_prefix = 'tutorialNN/'
savedir = Path('tutorialNN/system.save')

if not savedir.is_dir():
  if not archive_path.is_file():
    urlretrieve(asset_url, archive_path)
  with open_tar(archive_path, 'r:gz') as archive:
    members = [
      member for member in archive.getmembers()
      if member.name.startswith(asset_prefix)
    ]
    if not members:
      raise FileNotFoundError(f'{asset_prefix} is missing from {archive_path}')
    archive.extractall(path='.', members=members)
```

Never use a contributor's absolute path, an integration-test cache, or a silent
fallback to unrelated data. Fail with a clear message if download or validation
does not produce every required file.

## Build Tutorial Assets

Generate the QE and PAOFLOW data on the appropriate machine and place it in the
ignored tutorial asset directory:

```text
.github/assets_generation/tutorials/
  tutorial01/
    silicon.save/
  tutorialNN/
    system.save/
    output/       # Only when precomputed HPC plotting data is required
```

Build the archive with the single tutorial asset helper:

```bash
python .github/assets_generation/tutorials/build_assets.py
```

The helper includes XML and UPF files from every `.save` directory and includes
`output/` when present. It writes the ignored `tutorial_assets.tar.gz` beside
the helper. Inspect the archive, then run the upload helper. It regenerates the
checksum and publishes both files under the dedicated `tutorial-assets-v1`
release:

```bash
tar -tzf .github/assets_generation/tutorials/tutorial_assets.tar.gz
.github/assets_generation/tutorials/upload_release_assets.sh
```

Tutorial asset releases are immutable. When the payload changes, increment the
tag and update every tutorial URL in the same change:

```bash
.github/assets_generation/tutorials/upload_release_assets.sh tutorial-assets-v2
```

Do not publish tutorial data under `integration-assets-vN`; those releases are
reserved for CI regression inputs and references.

## Structure and Style

Open with the result, intended audience, prerequisites, learning outcomes or
success criterion, and estimated cost. Then:

- Explain why each PAOFLOW operation is needed before calling it.
- Keep the main workflow linear and put optional branches in tips.
- State what non-obvious arguments control and why their values suit the
  example.
- Show output filenames immediately after the calls that create them.
- Prefer a predefined `GPAO` plotting function whenever one supports the
  generated output, and tell the reader which function is being used. Use
  custom plotting code only when GPAO has no suitable function.
- Label plot axes and units, describe the expected qualitative result, and
  include at least one concrete validity check.
- Separate computed results, physical interpretation, and methodological
  limitations.
- End with cleanup through `finish_execution()` when the workflow requires it.

Tutorials should use numbered stages and build toward a final interpreted
result. How-tos should name their prerequisite tutorial, provide the shortest
complete procedure, and finish with a measurable success check.

## Publish and Validate

Before opening a pull request:

1. Run all cells from top to bottom in a clean directory using only the public
   asset URLs documented in the notebook.
2. Verify the runtime statement matches the packaged payload.
3. For HPC tutorials, verify plotting reads packaged output while the documented
   generation path writes the declared files.
4. Remove stale errors, large logs, and generated data from notebook output.
5. Build the documentation and test the notebook download and asset links from
   the rendered page.
6. Add the notebook to `docs/tutorials/index.md` or
   `docs/tutorials/howtos/index.md`.
7. Confirm the page contains no repository-relative reader instructions.

Use this final check from the repository root:

```bash
rg -n '\.github/|docs/tutorials|examples/' docs/tutorials \
  --glob '*.ipynb' --glob '*.md'
```

Repository paths may appear in contributor-only template comments, but not in
published reader instructions or executable setup.
