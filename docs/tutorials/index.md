# Tutorials

PAOFLOW tutorials are guided, workflows that teach both **how to use PAOFLOW** and **the physics behind the calculations**.

Each tutorial walks through a complete workflow end-to-end: from the required inputs, through each PAOFLOW step, to interpreting the results. Tutorials are written for researchers who are familiar with density functional theory but may be new to PAOFLOW.

---

::::{grid} 1 2 2 2
:gutter: 4
:class-container: tutorials-grid

:::{grid-item-card}
:link: tutorial01
:link-type: doc
:class-card: landing-card

{octicon}`book;1.8em;sd-text-muted`

**Tutorial 01**

Electronic structure: build a PAO Hamiltonian, interpolate band structure, and compute the density of states.
:::

:::{grid-item-card}
:link: tutorial02
:link-type: doc
:class-card: landing-card

{octicon}`graph;1.8em;sd-text-muted`

**Tutorial 02**

Transport properties: compute Boltzmann transport tensors and extend the workflow with relaxation-time models.
:::

::::

---

## How-Tos

How-Tos are shorter and more targeted than tutorials. Instead of teaching a workflow end-to-end, each one answers a specific practical question: how to set something up, which flags matter, and how to check that the result is right.

::::{grid} 1 2 2 2
:gutter: 4
:class-container: tutorials-grid

:::{grid-item-card}
:link: howtos/sparse-workflow
:link-type: doc
:class-card: landing-card

{octicon}`stack;1.8em;sd-text-muted`

**Building a sparse workflow**

Run PAOFLOW on systems and grids too large for the dense pipeline to hold in memory.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

tutorial01
tutorial02
howtos/index
```
