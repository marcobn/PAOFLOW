# How-to template

A how-to is a Markdown page `docs/learn/howtos/<task>.md` for a reader who already has their own calculation and needs one change to it: how to set something up, which flags matter, and how to check that the result is right. It varies an existing workflow (a flag, a route, a plot) on its base case, one case of `examples/cases/`, assumes the [bands and DOS tutorial](../tutorials/tutorial01.ipynb), and shows only what differs from them. Its code is a workflow script kept beside the page, which the reader downloads into a copy of the base case (see "The scripts"). The file name is the task in kebab case (`adhoc-spin-orbit.md`, `projectability-threshold.md`).

A how-to has at most 1500 words of prose (see the table at the top of SKILL.md). When it needs more, it usually answers several questions: split it into one how-to per question, or, if it teaches a workflow on one system, make it a tutorial.

## Structure

The items below are the default order, not a list of required sections. The title, the opening paragraph and the change are always there. Every other section is written only when it adds information the reader does not already have from the sections above it; a section that repeats values or checks already stated next to their output is left out, and its content stays where it is.

1. **Title**: `# How to <verb> <object>` (for example "How to add spin–orbit coupling to a scalar-relativistic calculation").
2. **Opening paragraph**: what the reader does with the page, the situation in which they need it, and the physical approximation it introduces, if any. The first sentence says what the reader does (see "Openings" in SKILL.md). The page is written for every system the task applies to. The base case is the system the page runs on; it appears where a concrete example helps, and other systems are linked, never shown.
   Then the **case tip**, the same on every tutorial and how-to, whose first sentence the checks and the page test read:

   ```markdown
   :::{tip}
   This page is built on the case [`<case>`](https://github.com/marcobn/PAOFLOW/tree/develop/examples/cases/<case>): open it in PAOFLOW Studio with *Examples…* → `<case>`, or copy it with `paoflow-gen --example <case>`. <the scripts, the data, the run time>
   :::
   ```
3. **The change**: the page's workflow `<name>.py`, shown in full and split into blocks that each follow the text explaining them (see "The scripts"). Keep it to the workflow of this page: when it varies a base example, write the workflow as that example with the change applied, and say in the text which blocks differ from the base example. When the change is a single argument, the block that holds the call is the one to explain. In the tip, link both scripts, each with what the reader gains from it: ``{download}`<name>.py <<task>/<name>.py>` `` to run the workflow in the copy of the case or on their own calculation, and ``{download}`figures.py <<task>/figures.py>` `` to redraw the figures of the page from that run (see "Connectors" in SKILL.md: an instruction carries its reason).
4. **The parameters that matter**: a table when the flags are parallel (flag, meaning, units, default, when to change it), or one short subsection per flag when a flag needs its own figure or code block, giving the physical meaning, units, default, how to choose the value for a given material, and what goes wrong when it is too small or too large. Explain the physics of each flag in one or two sentences, without display equations: what it approximates and why the default is what it is, with a link to the section of the theory page that has the rest. Quote a measured number only where it drives a choice ("on Si, `1e-4` moves the bands near the gap by about 10 meV, the default `1e-3` by about 0.1 eV"). Tables of statistics from the test runs (bond counts, timings, memory, error sweeps) put a load on the reader without helping them choose, so they stay in the report to the developer. Flags that share their physics (two ways of setting the same cutoff) get a parent subsection that states the shared physics once, then one subsection per flag.
5. **Other cases** (when the change affects many properties): one or two sentences on how to apply it to the properties the page does not show, with a link to the list of supported methods.
6. **Checking the result** (only when it adds information): the concrete checks in order of cost, for example the log lines to read, a comparison with the dense pipeline or the DFT bands, and a convergence test. Each check states what a good result looks like. When each section already states what its output should show, as on a page whose figures and printed output carry the expected values, there is no separate checking section.
7. **Troubleshooting** (optional): a table `| message | meaning | what to do |` with the literal error or warning text the code prints. Every method or option the table names is introduced in the body above.
8. **Where to go next** (optional): links to the theory page behind the change, the internals page that explains the mechanism, and related how-tos, one clause each.

Leave out any physics background beyond what is needed to choose the flags, and link to the theory page that covers it. A how-to has no display equations (check key `howto-math`); when a flag cannot be explained without one, the equation belongs on the theory page.

## The scripts

A how-to has two scripts beside the page, never in the case, so that a reader who downloads the workflow gets the workflow and nothing else (see "The page holds its workflow" in SKILL.md):

- The workflow `docs/learn/howtos/<task>/<name>.py` is the script the page teaches, and every line of it appears on the page. The reader downloads it into the copy of the case, so its name is the task's, never a file the case has or writes (`main.py`); the page's extra QE inputs (a band run to compare with, a `projwfc.x` input) go beside it. The workflow reads its input where the case's own runs put it (`savedir = 'silicon.save'`), writes its output files in the case folder, and prints what the page quotes. It draws no figures, unless drawing them is what the page teaches.
- `docs/learn/howtos/<task>/figures.py`, kept beside the page and not in the case, reads the output files of the workflow and its printed output, saved as its log, draws every figure of the page with `fig.savefig(name, dpi=150)` into the working directory, and ends with the asserts. It never imports the workflow, so it also redraws the figures from a run that was made elsewhere, for example on a cluster.

The workflow's printed output goes to `<name>.log`. The page test copies the workflow into a copy of the case, runs it with its log (on a generator case after the generated `main.py`), and then runs the figures script in that folder. Run them the same way, then copy the PNGs to `docs/_static/images/howtos/<task>/`.

Split the workflow into blocks with plain step comments, one per step of the page, and pull each block onto the page with `literalinclude`. A step comment is an ordinary comment that also helps a reader of the downloaded script:

```python
from PAOFLOW import GPAO

plotter = GPAO.GPAO()

# Band structure and DOS
si = 'plotting/si/'
plotter.plot_bands(si + 'bands_0.dat', sym_points=si + 'kpath_points.txt')

# Projected DOS grouped by orbital
files = sorted(Path(si).glob('Si*_pdosdk_0.dat'))
```

````markdown
```{literalinclude} <task>/<name>.py
:language: python
:start-after: '# Band structure and DOS'
:end-before: '# Projected DOS grouped by orbital'
```
````

`:start-after:` starts the block on the line after the first line of the file that contains the text, so the comment itself is not shown, and `:end-before:` stops it before the next step comment. The first block takes only `:end-before:`, the last only `:start-after:`. Because `:start-after:` matches any line that contains the text, each step comment must be unique and must not be part of another line (`# Bands` also matches `# Bands and DOS`). Sphinx only warns when the text is missing and then leaves the block out, so the checks of SKILL.md report a step comment that no block uses and a block whose text is not in the file; together they guarantee that every line of `main.py` is on the page.

A how-to whose workflow runs longer than the budget of "Assets and budgets" in SKILL.md ships the outputs of the workflow with the case's release data, and the page test runs `figures.py` on them; ask the developer before writing such a page.

## Figures and output

A how-to's code is not executed by the docs build, so paste the output the page discusses from `main.log` as a fenced `text` block, trimmed to the lines the page discusses. Figures are the PNGs `figures.py` writes, copied to `docs/_static/images/howtos/<task>/`, made with the palette and at the size given in "Figures" in SKILL.md. Use a figure or a schematic wherever it replaces a paragraph the reader would otherwise have to picture (the geometry of a setting, the range a flag acts on, a before-and-after comparison; see "Where a picture carries the explanation" and "Choosing the form" in SKILL.md), and leave out a figure that repeats a neighbouring one.

## Registration

Add a list entry, a link whose text is the page title, and a toctree entry to `docs/learn/howtos/index.md`:

```markdown
- [How to <verb> <object>](<task>.md)
```

and `<task>` under the page's `{toctree}`. How-tos are not shown as grid cards.
