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

**Tutorial 01**

Electronic structure: build a PAO Hamiltonian, interpolate band structure, and compute the density of states.
:::

:::{grid-item-card}
:link: tutorial02
:link-type: doc
:class-card: landing-card

**Tutorial 02**

Transport properties: compute Boltzmann transport tensors and extend the workflow with relaxation-time models.
:::

:::{grid-item-card}
:link: tutorial03
:link-type: doc
:class-card: landing-card

**Tutorial 03**

Running linear response calculations
:::

:::{grid-item-card}
:link: tutorial04
:link-type: doc
:class-card: landing-card

**Tutorial 04**

Large cells on the sparse backend: truncate the PAO Hamiltonian, double the Si cell, compute DOS and transport in one pass, and check the result against the dense pipeline.
:::

:::{grid-item-card}
:link: tutorial05
:link-type: doc
:class-card: landing-card

**Tutorial 05**

Topology of tight-binding models: the quantised spin Hall conductivity of the Kane–Mele model, Berry phases around a gapped Dirac cone, and Chern numbers of gapped graphene and the Haldane model.
:::

::::

---

## How-Tos

How-Tos are shorter and more targeted than tutorials. Instead of teaching a workflow end-to-end, each one answers a specific practical question: how to set something up, which flags matter, and how to check that the result is right.


```{toctree}
:hidden:
:maxdepth: 2
:glob:

tutorial*
howtos/index
```
