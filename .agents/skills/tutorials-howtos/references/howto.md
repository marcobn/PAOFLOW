# How-to template

A how-to is a Markdown page `docs/tutorials/howtos/<task>.md` for a reader who already has their own calculation and needs one change to it: how to set something up, which flags matter, and how to check that the result is right. It varies an existing workflow (a flag, a route, a plot), often one of the examples under `examples/`, assumes [Tutorial 01](../tutorial01.ipynb), and shows only what differs from them. Its code is the script `docs/tutorials/howtos/<task>/main.py`. The file name is the task in kebab case (`adhoc-spin-orbit.md`, `projectability-threshold.md`), and the script's directory has the same name.

A how-to has at most 1500 words of prose (see the table at the top of SKILL.md). When it needs more, it usually answers several questions: split it into one how-to per question, or, if it teaches a workflow on one system, make it a tutorial.

## Structure

The items below are the default order, not a list of required sections. The title, the opening paragraph and the change are always there. Every other section is written only when it adds information the reader does not already have from the sections above it; a section that repeats values or checks already stated next to their output is left out, and its content stays where it is.

1. **Title**: `# How to <verb> <object>` (for example "How to add spin–orbit coupling to a scalar-relativistic calculation").
2. **Opening paragraph**: what the reader does with the page, the situation in which they need it, and the physical approximation it introduces, if any. The first sentence says what the reader does (see "Openings" in SKILL.md). The page is written for every system the task applies to. The system used to test it (the base example, a supercell chosen for the runs) is not its subject and appears only where a concrete example helps. When the page builds on an example under `examples/`, follow with a line of the form
   `Base example: [examples/<family>/<name>/](https://github.com/marcobn/PAOFLOW/tree/master/examples/<family>/<name>)`
   (the checks verify that the path exists). A page whose code is complete on its own, such as one that builds its model in code, has no such line and says in its opening which inputs, if any, the reader needs. A page that draws on several workflows, such as plotting the output of different calculations, writes `Base examples:` followed by one such link per example, all on that line; the checks verify every path, and the text after each figure names the example its data come from.
3. **The change**: one screen of code at most, showing only the lines that differ from the base example's `main.py`, with enough context to place them. A page without a base example shows the complete script, split into blocks that each follow the text explaining them. Every block is pulled from the page's `main.py` (see "The script"). When the change is a single argument, show the call with that argument and nothing more. Below the first block, link the full script: ``{download}`Download the script <<task>/main.py>` ``.
4. **The parameters that matter**: one short subsection or a table per flag, giving the physical meaning, units, default, how to choose the value for a given material, and what goes wrong when it is too small or too large. Explain the physics of each flag: what it approximates and why the default is what it is. Quote a measured number only where it drives a choice ("on Si, `1e-4` moves the bands near the gap by about 10 meV, the default `1e-3` by about 0.1 eV"). Tables of statistics from the test runs (bond counts, timings, memory, error sweeps) put a load on the reader without helping them choose, so they stay in the report to the developer. Flags that share their physics (two ways of setting the same cutoff) get a parent subsection that states the shared physics once, then one subsection per flag.
5. **Other cases** (when the change affects many properties): one or two sentences on how to apply it to the properties the page does not show, with a link to the list of supported methods.
6. **Checking the result** (only when it adds information): the concrete checks in order of cost, for example the log lines to read, a comparison with the dense pipeline or the DFT bands, and a convergence test. Each check states what a good result looks like. When each section already states what its output should show, as on a page whose figures and printed output carry the expected values, there is no separate checking section.
7. **Troubleshooting** (optional): a table `| message | meaning | what to do |` with the literal error or warning text the code prints. Every method or option the table names is introduced in the body above.
8. **Where to go next** (optional): links to the internals page that explains the mechanism, and to related how-tos, one clause each.

Leave out any physics background beyond what is needed to choose the flags, and link to the tutorial or internals page that covers it.

## The script

`main.py` reproduces everything on the page: it runs the calculation, prints the output the page quotes, draws every figure, and ends with the asserts of the numbers the page's conclusions rest on (see "Running the code" in SKILL.md). The page test runs it on every pull request, so the code on the page is the code that runs.

Mark each part of the script that the page shows with a pair of comments, and pull it onto the page with `literalinclude`:

```python
# [build-cell]
geom, theta = twisted_bilayer(10, 9)
# [/build-cell]
```

````markdown
```{literalinclude} <task>/main.py
:language: python
:start-after: '# [build-cell]'
:end-before: '# [/build-cell]'
```
````

The part between the markers appears on the page as a code block with copy button. Parts the page does not show (imports needed only for plotting, the plotting code, the asserts) stay in the script outside the markers. Write the script so that it runs from any directory: it reads its input from the extracted asset next to it (`savedir = '<task>/<system>.save'`) and writes its output files and figures to the working directory. To refresh the figures of the page, run it from a scratch directory and copy the PNGs to `docs/_static/images/howtos/<task>/`.

A script that runs longer than the budget of "Assets and budgets" in SKILL.md reads precomputed outputs from its asset for the page test, and its case is marked slow; ask the developer before writing such a page.

## Figures and output

A how-to's code is not executed by the docs build, so paste the output the page discusses from a run of `main.py` as a fenced `text` block, trimmed to the lines the page discusses. Figures are the PNGs `main.py` writes, copied to `docs/_static/images/howtos/<task>/`, made with the palette and at the size given in "Figures" in SKILL.md. A how-to is short, so each figure has to earn its place: use one where it replaces a paragraph the reader would otherwise have to picture (the geometry of a setting, the range a flag acts on, a before-and-after comparison; see "Where a picture carries the explanation" in SKILL.md).

## Registration

Add a list entry, a link whose text is the page title, and a toctree entry to `docs/tutorials/howtos/index.md`:

```markdown
- [How to <verb> <object>](<task>.md)
```

and `<task>` under the page's `{toctree}`. How-tos are not shown as grid cards. The How-Tos list on `docs/tutorials/index.md` holds featured how-tos only; add the entry there too (`howtos/<task>.md`) if the user asks.
