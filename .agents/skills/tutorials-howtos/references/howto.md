# How-to template

A how-to is a Markdown page `docs/tutorials/howtos/<task>.md` that answers one practical question: how to set something up, which flags matter, and how to check that the result is right. It varies an existing example (a flag, a route, a plot) and assumes [Tutorial 01](../tutorial01.ipynb). The file name is the task in kebab case (`adhoc-spin-orbit.md`, `projectability-threshold.md`).

## Structure

1. **Title**: `# How to <verb> <object>` (for example "How to add spin–orbit coupling to a scalar-relativistic calculation").
2. **Opening paragraph**: the question the page answers, the situation in which we need it, and the physical approximation it introduces, if any. Then a line of the form
   `Base example: [examples/<family>/<name>/](https://github.com/marcobn/PAOFLOW/tree/master/examples/<family>/<name>)`
   (the checks look for `Base example:` and verify that the path exists).
3. **The change**: one screen of code at most, showing only the lines that differ from the base example's `main.py`, with enough context to place them. Use a fenced `python` block. When the change is a single argument, show the call with that argument and nothing more.
4. **The parameters that matter**: one short subsection or a table per flag, giving the physical meaning, units, default, how to choose the value for a given material, and what goes wrong when it is too small or too large. Measured numbers from our run (a convergence table, a timing, a memory figure) belong here.
5. **Checking the result**: the concrete checks in order of cost, for example the log lines to read, a comparison with the dense pipeline or the DFT bands, and a convergence test with the size of the change we found. Each check states what a good result looks like in numbers.
6. **Troubleshooting** (optional): a table `| message | meaning | what to do |` with the literal error or warning text the code prints.

Leave out any physics background beyond what is needed to choose the flags, and link to the tutorial or internals page that covers it.

## Figures and output

A how-to's code is not executed by the docs build. We run it as a script (the base example with the change applied) before publishing, and paste the relevant output as a fenced `text` block, trimmed to the lines the page discusses. Figures are PNGs from that run in `docs/_static/images/howtos/<task>/`, made with the palette in SKILL.md.

## Registration

Add a grid card and a toctree entry to `docs/tutorials/howtos/index.md`:

```markdown
:::{grid-item-card}
:link: <task>
:link-type: doc
:class-card: landing-card

{octicon}`<icon>;1.8em;sd-text-muted`

**<Title without "How to">**

<One sentence: what we can do after reading the page.>
:::
```

and `<task>` under the page's `{toctree}`. The How-Tos grid on `docs/tutorials/index.md` lists featured how-tos only; add the card there too if the user asks.

## Before and after

From the sparse how-to, the opening of the `rcut` section.

Before:

> `rcut` is a **second and physically different** truncation: it discards hoppings by physical bond length $|a_\mathrm{lat}\mathbf{R} + \boldsymbol{\tau}_\alpha - \boldsymbol{\tau}_\beta|$ in Bohr, rather than by amplitude. The statement behind it is that the interaction is short-ranged — a claim about the material, not about numerical noise.

After:

> `rcut` discards hoppings by bond length $|a_\mathrm{lat}\mathbf{R} + \boldsymbol{\tau}_\alpha - \boldsymbol{\tau}_\beta|$ (Bohr) instead of by amplitude. Where `hopping_threshold` removes couplings too weak to matter, `rcut` assumes the interaction is short-ranged, which is a statement about the material. We use it when the PAO Hamiltonian has long, weak tails that the threshold would keep.

What changed. There is no bold for emphasis and no em dash. The contrast with the other flag is stated once, as a fact, and the last sentence tells us when to reach for the flag.
