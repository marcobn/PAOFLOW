# How-Tos

Where the tutorials teach a complete workflow end-to-end, a **How-To** answers a narrower, practical question: _how do I set this up, which flags matter, and how do I know the result is right?_

How-Tos assume you have already worked through [Tutorial 01](../tutorial01.ipynb) and are comfortable with the basic PAOFLOW call sequence.

---

::::{grid} 1 2 2 2
:gutter: 4
:class-container: tutorials-grid

:::{grid-item-card}
:link: sparse-workflow
:link-type: doc
:class-card: landing-card

{octicon}`stack;1.8em;sd-text-muted`

**Large supercells on the sparse backend**

Switch a script to the sparse backend, choose the truncation of the Hamiltonian and the energy range of the solve, and check the result against the dense pipeline.
:::

:::{grid-item-card}
:link: qo_analysis
:link-type: doc
:class-card: landing-card

{octicon}`pulse;1.8em;sd-text-muted`

**Comparing pyskeaf orbits with quantum oscillations**

Assign measured dHvA/SdH/torque frequencies to Fermi-surface sheets, with chance baselines, convergence and breakdown checks, and an agentic workflow.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2

sparse-workflow
qo_analysis
```
