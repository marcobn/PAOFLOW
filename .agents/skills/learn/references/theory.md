# Theory template

A theory page is a Markdown page `docs/learn/theory/<topic>.md` that explains the physics and the method behind one family of PAOFLOW results: which quantity is computed, from which equations, under which approximations, and what those approximations leave out. The reader arrives from a link on a tutorial or how-to, or reads the page to decide whether a result can be trusted. Tutorials and how-tos carry only the equations they need (see "Physics depth" in SKILL.md), so the derivations, the intuition and the limits of each approximation have their home here. How the code evaluates the equations (solvers, data layout, MPI, memory) is in `docs/internals/`, not here. The file name is the topic in kebab case (`boltzmann-transport.md`).

A theory page has at most 3000 words of prose; equations do not count. A topic that needs more is split into two pages that link to each other.

## Structure

1. **Title**: `# <Physical topic>`, naming the physics and, where it matters, the approximation: "Boltzmann transport in the relaxation-time approximation", "Berry curvature and topological invariants". Not "Theory of ...".
2. **Opening paragraph**: what the page explains, which PAOFLOW methods evaluate it, linked to their API entries (``{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.transport` ``), and what it builds on (a link to [The PAO Hamiltonian](pao-hamiltonian.md) for every page but that one).
3. **One section per concept**, ordered so that each builds on the ones above it. A section gives, in this order and as far as the concept needs:
   - the derivation, in steps a reader can follow with pen and paper, or a citation where a step is skipped. Every result not derived on the page cites its source;
   - the equation as the code evaluates it: signs, factors, unit conversions and normalisations, from the source (step 2 of the Workflow in SKILL.md). Where it differs from the textbook form, show the textbook form too and say where the difference comes from (a convention, a unit system, a prefactor absorbed elsewhere), and report it to the developer;
   - the intuition: what the quantity means for an electron in the crystal, and how it shows in a calculated spectrum or tensor;
   - the approximations: each one named, what it neglects, and the physical scale at which it fails;
   - the argument of the PAOFLOW method that controls the quantity, with a link to its API entry for the full list.
4. **Used in**: a list of the tutorials and how-tos that rely on the page, one link each with what they compute there. Every page that links to this one appears in the list.
5. **References**: the cited works as a numbered list (authors, title, journal, volume, page, year). Cite only what you could confirm; otherwise ask the user.

## Rules

- **Anchors are an interface.** Each equation that a tutorial or how-to links to sits under its own heading (`##` or `###`), so the link can target `theory/<topic>.md#heading-anchor`. Before renaming a heading, search `docs/` for its anchor and update the links.
- **No workflow code.** A theory page shows no code blocks and no call sequences; the workflow is on the tutorials and how-tos, which the "Used in" list links. A method or argument is named only where the page says which one evaluates or controls an equation.
- **No numbers from a run.** The page describes the method, not a result. A value that illustrates a scale (a band width, a relaxation time, a smearing width) comes from a cited source or is stated as an order of magnitude.
- **Notation** is the table in SKILL.md, so a link from a tutorial lands on the same symbols. A new symbol is defined at its first use and added to the table when other pages will reuse it.
- **Voice.** Instructions and choices address the reader as "you", as on the other pages. Derivations make the quantity or the step the subject ("Expanding $f$ to first order in $\mathbf{E}$ gives ..."), never "we".
- **Figures** follow "Figures" in SKILL.md. A schematic (a Brillouin zone, an energy window, a smearing function) is drawn by `docs/learn/theory/<topic>/figures.py` into `docs/_static/images/theory/<topic>/`, so a later developer can redraw it.

## Registration

Add a list entry, a link whose text is the page title, and a toctree entry to `docs/learn/theory/index.md`:

```markdown
- [<Title>](<topic>.md)
```

and `<topic>` under the page's `{toctree}`.
