# Learn

The Learn section has three kinds of pages, for three reasons to open the documentation. Tutorials take one material through a complete workflow. How-tos change one thing in a calculation you already have. Theory pages explain the physics behind the results: the equations PAOFLOW evaluates, where they come from, and where their approximations fail. Tutorials and how-tos keep the physics to what you need to run the workflow and read its output, and link to the theory pages for the rest.

::::{grid} 1 2 3 3
:gutter: 4
:class-container: sd-mt-4

:::{grid-item-card}
:link: tutorials/index
:link-type: doc
:class-card: landing-card

{octicon}`mortar-board;2.4em;sd-text-muted`

**Tutorials**

Complete workflows on one material, from the DFT output to an interpreted result.
:::

:::{grid-item-card}
:link: howtos/index
:link-type: doc
:class-card: landing-card

{octicon}`tools;2.4em;sd-text-muted`

**How-tos**

One change to your own calculation: a flag, a route, a plot, a check.
:::

:::{grid-item-card}
:link: theory/index
:link-type: doc
:class-card: landing-card

{octicon}`book;2.4em;sd-text-muted`

**Theory**

The physics behind each result: equations, derivations and approximations.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

tutorials/index
howtos/index
theory/index
```
