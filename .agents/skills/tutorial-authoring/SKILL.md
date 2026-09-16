---
name: tutorial-authoring
description: "Create, revise, or review PAOFLOW tutorials and how-to guides. Use when: documenting a physical property workflow, writing tutorial notebooks, writing task-focused how-tos, standardizing tutorial style, reducing cognitive load, or preparing tutorial assets."
user-invocable: true
---

# Tutorial Authoring

## Scope

Use this skill to create, revise, or review content in `docs/tutorials/` and
`docs/tutorials/howtos/`. It gives agents and human contributors one workflow
for producing documentation that is scientifically useful, executable, and easy
to follow.

Start from `docs/tutorials/tutorial-template.ipynb` for an end-to-end learning
experience or `docs/tutorials/howtos/how-to-template.ipynb` for a focused task.
The templates are excluded from the documentation build. Copy and rename the
appropriate notebook before replacing its placeholders.

## Choose the Document Type

Classify the requested document before drafting it.

| Reader need                                                                 | Document type | Defining characteristic                            |
| --------------------------------------------------------------------------- | ------------- | -------------------------------------------------- |
| Learn a PAOFLOW property workflow from inputs through interpretation        | Tutorial      | Learning-oriented and cumulative                   |
| Accomplish one specific setup, parameter, troubleshooting, or workflow task | How-to        | Goal-oriented and assumes basic familiarity        |
| Look up methods, arguments, defaults, files, or formats                     | Reference     | Accurate technical specification                   |
| Understand theory, design choices, or why a method works                    | Explanation   | Conceptual understanding without a procedural goal |

Choose a **tutorial** when the reader needs a complete path, concepts must be
introduced in sequence, and interpreting the physical result is part of the
learning outcome. A tutorial may compute several intermediate quantities only
when each one is necessary for the final property.

Choose a **how-to** when the reader already understands the basic PAOFLOW call
sequence and has one concrete question, such as selecting sparse controls,
defining a custom path, or checking convergence. Link to the prerequisite
tutorial instead of teaching the base workflow again.

Do not turn API listings into how-tos or theory essays into tutorials. Update or
create reference or explanation documentation when those classifications fit.

## Authoring Workflow

1. **Define the reader and outcome.** Write one sentence stating who the reader
   is and what scientifically meaningful result they will produce.
2. **Set the boundary.** Identify prerequisites, the final observable or task,
   and topics that belong in linked reference, explanation, or follow-up docs.
3. **Verify the workflow.** Inspect the current public PAOFLOW API and a nearby
   working example. Do not derive instructions from memory or stale notebook
   output.
4. **Inventory assets.** List every required input before drafting code. Follow
   the tutorial asset contract and avoid integration-test caches or developer
   machine paths.
5. **Start from the matching template.** Preserve its heading order unless a
   section is demonstrably inapplicable. Adapt placeholders to the property or
   task; do not publish the template itself as a documentation page.
6. **Write in executable increments.** Each numbered stage should explain why
   the step is needed, run the smallest useful code, state what to expect, and
   interpret the result before moving on.
7. **Run from a clean state.** Execute all cells from top to bottom using the
   documented assets and working directory. Remove stale output and errors.
8. **Make the document discoverable.** Add or update the appropriate card and
   toctree entry in `docs/tutorials/index.md` or
   `docs/tutorials/howtos/index.md`.
9. **Review against the checklist.** Treat scientific correctness,
   reproducibility, and readability as release requirements.

## Uniform Style

### Lead With Orientation

The opening must provide, in this order:

1. A literal title naming the property or task.
2. A short statement of the result the reader will obtain.
3. The intended audience and assumed knowledge.
4. Learning outcomes for tutorials or a success criterion for how-tos.
5. Prerequisites, runtime or computational cost when material, and exact assets.

Avoid promotional introductions and long historical background. Link to deeper
theory after giving the reader enough context to begin.

### Minimize Cognitive Load

- Teach one coherent workflow per document.
- Put prerequisite concepts before the step that needs them.
- Introduce a concept immediately before its first use.
- Keep paragraphs short and headings descriptive.
- Prefer one purposeful code cell over fragmented one-line cells.
- Keep each code cell runnable using only earlier cells.
- Use descriptive variable names such as `bands_file` and `energy_grid`.
- Define acronyms and specialized terms on first use.
- Use tables for comparable options, parameters, or trade-offs.
- Put optional branches in tips or separate how-tos, not in the main path.
- Repeat a small amount of essential context when it saves navigation, but keep
  one authoritative source for detailed reference information.

### Explain Physics and Numerics

For every public PAOFLOW operation, explain its purpose before showing the call.
For every non-obvious physical or numerical argument, state:

- what it controls;
- why the shown value is appropriate for the example; and
- when a user should reconsider it.

Use equations only when they clarify the operation or interpretation. Define
all symbols near the equation. Keep derivations in explanation pages unless the
derivation is required to complete the learning objective.

Separate these statements clearly:

- what PAOFLOW computed;
- what the result means physically;
- what numerical or methodological limitation applies; and
- what evidence shows that the result is credible.

### Keep Code and Output Focused

- Use public, physics-readable PAOFLOW methods rather than internal arrays or
  backend functions.
- Keep setup explicit and portable. Use `pathlib.Path` for filesystem paths.
- Show the output filename immediately after the call that creates it.
- Prefer a predefined `GPAO` plotting function whenever one supports the
  output, and tell the reader which function is being used. Write custom
  plotting code only when GPAO has no suitable function.
- Plot only quantities needed to meet the outcome. Label axes and include units.
- State expected qualitative features and at least one concrete validity check.
- Do not embed large generated data, logs, or decorative output in notebooks.
- Call `finish_execution()` when the demonstrated workflow requires PAOFLOW
  cleanup or MPI finalization.

## Tutorial Requirements

A tutorial must:

- guide a newcomer through a complete workflow for one property or coherent set
  of properties;
- build cumulatively from inputs to a final plotted or inspected result;
- explain why each stage is necessary;
- include expected results and scientific interpretation;
- identify method limitations and sensible next steps; and
- be complete for its declared outcome without cataloguing every option.

Use numbered sections for the executable workflow. Keep the main path linear.
If an alternative would interrupt that path, summarize it in a tip and link to
a dedicated how-to or reference page.

## How-To Requirements

A how-to must:

- answer one task phrased as "How do I ...?";
- state the prerequisite tutorial and starting state;
- provide the shortest complete sequence that solves the task;
- explain choices only where they affect a decision;
- include a concrete success check; and
- cover only the most likely failure modes.

Do not duplicate the full initialization and theory narrative from a tutorial.
Reuse a numbered tutorial's asset and declare that dependency explicitly.

## Tutorial Assets

### Ownership and Location

Each asset is owned by the first numbered tutorial that introduces it. Put
generated payload under:

```text
.github/assets_generation/tutorials/tutorialNN/
```

The single `build_assets.py` helper packages these directories as
`tutorial_assets.tar.gz`. Generated archives, checksums, and `.save`
directories must remain ignored by Git.

Later tutorials and how-tos reuse an existing asset when the input calculation
is unchanged. They link to the owning tutorial as a prerequisite and name it in
their opening section instead of duplicating the asset.

### Minimum Contents

Package only files required by the documented workflow:

| File                   | Include when                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `data-file-schema.xml` | Always; initialization reads the crystal, k-grid, and electronic-structure metadata.                               |
| `atomic_proj.xml`      | The notebook calls `read_atomic_proj_QE()`.                                                                        |
| Relevant `.UPF` files  | The notebook constructs projections internally or otherwise reads pseudopotential data from the `.save` directory. |

If another file is indispensable, explain why in the notebook prerequisites.
Do not add a file merely because Quantum ESPRESSO generated it.

Classify the workflow before choosing its payload:

- When an example runs easily on a laptop, package only trimmed QE `.save`
  data. Do not package PAOFLOW output; the notebook generates it.
- When an example requires HPC resources, package trimmed QE `.save` data and
  the PAOFLOW output files consumed by plots. Plot packaged output by default
  while retaining the complete HPC generation commands and parameters.

Exclude wavefunctions unless required, charge-density files, transient logs,
rendered plots, caches, editor metadata, integration-test references, and
duplicate assets owned by another tutorial.

### Portable Paths

Write published notebooks for readers who do not have a repository checkout.
Link to the public release archive and extract only the owning tutorial prefix:

```python
from pathlib import Path
from tarfile import open as open_tar
from urllib.request import urlretrieve

asset_url = (
   'https://github.com/marcobn/PAOFLOW/releases/download/'
   'tutorial-assets-v1/tutorial_assets.tar.gz'
)
archive_path = Path('tutorial_assets.tar.gz')
asset_prefix = 'tutorial01/'
savedir = Path('tutorial01/silicon.save')

if not savedir.is_dir():
   if not archive_path.is_file():
      urlretrieve(asset_url, archive_path)
   with open_tar(archive_path, 'r:gz') as archive:
      members = [m for m in archive.getmembers() if m.name.startswith(asset_prefix)]
      archive.extractall(path='.', members=members)
```

Never use a repository-relative path, absolute contributor path, integration-
test cache, implicit editor working directory, or silent fallback to unrelated
data in reader-facing instructions.

Build payloads with
`python .github/assets_generation/tutorials/build_assets.py`, inspect the
archive, then publish it with `upload_release_assets.sh`. Tutorial asset tags
are immutable: when the payload changes, publish the next
`tutorial-assets-vN` tag and update every notebook URL. Never publish tutorial
data under an `integration-assets-vN` tag.

### Notebook Downloads

Every published tutorial and how-to must provide a source-notebook download
near the top. Use the Sphinx download role so the built website copies and
serves the notebook instead of navigating to the rendered page:

```markdown
{download}`Download this notebook <tutorialNN.ipynb>`
```

### Opening Declaration

Before the first code cell, state:

1. Whether the example runs easily on a laptop or requires HPC resources.
2. The asset directory and owning tutorial.
3. Every required file and its purpose, including HPC plotting outputs.
4. The public release URL and how the setup cell obtains the asset.
5. The approximate download size and generation cost when material.
6. Relevant provenance such as the material, pseudopotential family,
   exchange-correlation approximation, or spin-orbit treatment.

Fail early with a clear missing-asset message rather than relying on a later XML
parser or workflow call to reveal incomplete setup.

### Separation From Test Assets

`.github/assets_generation/tutorials/` contains the single archive builder;
generated tutorial payload remains ignored. `.github/assets_generation/qe/` and
`.github/assets_generation/transport/` generate, package, and publish test
archives and numerical references.

Do not reuse the integration-test archives as a tutorial download. When data
must also support regression tests, keep the tutorial and test archives
separate and make both generation paths explicit.

## Review Checklist

### Purpose and Structure

- [ ] The document is correctly classified as tutorial, how-to, reference, or
      explanation.
- [ ] The audience, outcome, prerequisites, and scope are explicit at the top.
- [ ] Headings are descriptive, concise, and ordered by prerequisite knowledge.
- [ ] Optional material does not interrupt the main path.

### Assets and Reproducibility

- [ ] Required assets and their exact locations are listed before first use.
- [ ] The notebook explains near the top whether it runs easily on a laptop or
      requires HPC resources.
- [ ] Examples that run easily on a laptop supply only QE `.save` data.
- [ ] HPC payloads contain the PAOFLOW files used by plotting cells.
- [ ] HPC plotting reads packaged output while generation steps remain documented.
- [ ] Public asset URLs work without a repository checkout.
- [ ] Tutorial URLs pin an immutable `tutorial-assets-vN` release, not an
      integration-test or PAOFLOW package release.
- [ ] A Sphinx download role provides the source notebook from the built page.
- [ ] All cells run top to bottom in a clean environment.
- [ ] Public calls and argument names match the current PAOFLOW API.
- [ ] Output filenames are stated and generated where promised.
- [ ] Notebook output contains no stale tracebacks or contradictory results.
- [ ] Asset contents follow the minimum-content rules and exclude generated
      results, caches, and test references.
- [ ] Reused assets are credited to the numbered tutorial that owns them.

### Scientific Quality

- [ ] Every workflow step has a stated purpose.
- [ ] Non-obvious parameters have a rationale and reconsideration condition.
- [ ] Figures include labels, units, readable legends, and no hidden assumptions.
- [ ] Plotting uses a predefined GPAO function wherever one is available, and
      the surrounding text identifies it.
- [ ] Expected qualitative behavior and at least one validity check are given.
- [ ] Interpretation is separated from numerical and method limitations.

### Consistency and Maintenance

- [ ] PAOFLOW terms, units, admonitions, filenames, and plotting style are
      consistent with nearby documentation.
- [ ] Links use descriptive labels rather than "click here".
- [ ] The relevant tutorial or how-to index is updated.
- [ ] Details already maintained in reference docs are linked, not duplicated.
- [ ] `finish_execution()` is included when required by the workflow.

## Design Basis

This skill adapts the Diataxis distinction between learning-oriented tutorials
and goal-oriented how-to guides. Its emphasis on clear, concise, structured,
skimmable, consistent, current, and cumulative documentation is informed by:

- [Documentation done right: A developer's guide](https://github.blog/developer-skills/documentation-done-right-a-developers-guide/)
- [Write the Docs documentation principles](https://www.writethedocs.org/guide/writing/docs-principles/)

Repository-specific workflow, asset, and scientific-validation rules in this
skill take precedence when applying those general principles.
