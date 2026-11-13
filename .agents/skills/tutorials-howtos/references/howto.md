# How-to template

A how-to is a Markdown page `docs/tutorials/howtos/<task>.md` that answers one practical question: how to set something up, which flags matter, and how to check that the result is right. It varies an existing example (a flag, a route, a plot) and assumes [Tutorial 01](../tutorial01.ipynb). The file name is the task in kebab case (`adhoc-spin-orbit.md`, `projectability-threshold.md`).

## Structure

1. **Title**: `# How to <verb> <object>` (for example "How to add spin–orbit coupling to a scalar-relativistic calculation").
2. **Opening paragraph**: the question the page answers, the situation in which we need it, and the physical approximation it introduces, if any. The page is written for every system the task applies to. The system used to test it (the base example, a supercell chosen for the runs) is not its subject and appears only where a concrete example helps. Then a line of the form
   `Base example: [examples/<family>/<name>/](https://github.com/marcobn/PAOFLOW/tree/master/examples/<family>/<name>)`
   (the checks look for `Base example:` and verify that the path exists).
3. **The change**: one screen of code at most, showing only the lines that differ from the base example's `main.py`, with enough context to place them. Use a fenced `python` block. When the change is a single argument, show the call with that argument and nothing more.
4. **The parameters that matter**: one short subsection or a table per flag, giving the physical meaning, units, default, how to choose the value for a given material, and what goes wrong when it is too small or too large. Explain the physics of each flag: what it approximates and why the default is what it is. Quote a measured number only where it drives a choice ("on Si, `1e-4` moves the bands near the gap by about 10 meV, the default `1e-3` by about 0.1 eV"). Tables of statistics from the test runs (bond counts, timings, memory, error sweeps) put a load on the reader without helping them choose, so they stay in the report to the developer. Flags that share their physics (two ways of setting the same cutoff) get a parent subsection that states the shared physics once, then one subsection per flag.
5. **Other cases** (when the change affects many properties): one or two sentences on how to apply it to the properties the page does not show, with a link to the list of supported methods.
6. **Checking the result**: the concrete checks in order of cost, for example the log lines to read, a comparison with the dense pipeline or the DFT bands, and a convergence test. Each check states what a good result looks like.
7. **Troubleshooting** (optional): a table `| message | meaning | what to do |` with the literal error or warning text the code prints. Every method or option the table names is introduced in the body above.
8. **Where to go next** (optional): links to the internals page that explains the mechanism, and to related how-tos, one clause each.

Leave out any physics background beyond what is needed to choose the flags, and link to the tutorial or internals page that covers it.

## Figures and output

A how-to's code is not executed by the docs build. We run it as a script (the base example with the change applied) before publishing, and paste the relevant output as a fenced `text` block, trimmed to the lines the page discusses. Figures are PNGs from that run in `docs/_static/images/howtos/<task>/`, made with the palette in SKILL.md. The plotting script is not kept; the text after each figure names the run and the settings it was made from (see "Figures" in SKILL.md).

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

From the sparse how-to, the steps that change role on the sparse backend.

Before:

> `pao_eigh()`, `gradient_and_momenta()` and `adaptive_smearing()` are optional and only record their options. The eigenvalues, band velocities and adaptive smearing widths are computed in one pass over the k-mesh, started by the first property that needs them.

After:

> On the dense pipeline, `pao_eigh()` diagonalises $H(\mathbf{k})$ at every k-point and keeps all eigenvectors, $N_k\, n_\mathrm{awf}^2$ complex numbers, for the steps that follow. For a large cell that array does not fit in memory. The sparse backend therefore diagonalises one k-point at a time, computes there everything the properties of the script need (eigenvalues, band velocities, smearing widths, projections for the PDOS), and discards the eigenvectors before the next k-point. We call one such loop over the mesh a pass. It runs when a property needs it, not where the three calls are written: `pao_eigh()`, `gradient_and_momenta()` and `adaptive_smearing()` compute nothing themselves. They store their arguments for the pass, and a script that keeps the defaults can leave them out.

What changed. The reader learns why the three calls behave differently, from the memory cost they already know on the dense pipeline. "Pass" is defined before it is used, and "only record their options" is replaced by what happens to the arguments and what that means for the script.
