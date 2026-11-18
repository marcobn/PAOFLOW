---
name: tutorials-howtos
description: "Write or revise the tutorials (docs/tutorials/tutorialNN.ipynb) and how-tos (docs/tutorials/howtos/<task>.md) of the Read the Docs site so they read like physics written for physicists. Use when: writing a new tutorial or how-to, turning an example into a how-to, revising, reviewing or proofreading an existing tutorial or how-to page, or adding one to the tutorials or how-to index, even if the user only says 'document this workflow' or 'write a page for this flag'."
user-invocable: true
---

# Tutorials and how-tos

The reader is a researcher who knows density functional theory but not PAOFLOW. They read the page on the Read the Docs website, and only some of them download the notebook to run it. Each page tells them which physical quantity a workflow gives, under which approximation, how to choose the parameters that control it, and how to tell whether the result can be trusted. Software detail goes in only when it changes what they can compute (memory that grows with cell size, MPI pools, the sparse backend).

The pages are written for people to read, not for agents to parse. The aim is to communicate ideas clearly to a colleague. That means no instructions addressed to tools, no checklists written for a model, and no prompt-like text. It also means no marketing language, which rarely communicates an idea. Write at the level of a good review article: the reader is an expert in a neighbouring field. Do not talk down to them, and do not make them decode in-house shorthand (see "Jargon").

There are two page types, defined by what the reader comes to the page for. Read the matching template before writing:

|              | Tutorial                                                                                                         | How-to                                                                                                       |
| ------------ | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Reader       | Is learning. Follows one fixed system from the inputs to the interpreted result, and runs the page as it is       | Already has their own calculation and needs one change to it: a flag, a route, a plot, a check               |
| File         | `docs/tutorials/tutorialNN.ipynb`                                                                                | `docs/tutorials/howtos/<task>.md`, with its scripts `docs/tutorials/howtos/<task>/main.py` and `figures.py`    |
| Template     | `references/tutorial.md`                                                                                         | `references/howto.md`                                                                                        |
| Builds on    | Tutorial 01 (except Tutorial 01 itself)                                                                          | Tutorial 01, and an example under `examples/` when one exists. Shows only what differs from them             |
| Length       | Up to 3000 words of prose                                                                                        | Up to 1500 words of prose                                                                                    |

The reader's intent decides the type, and the length is the signal that a page has drifted from it. A how-to also drifts when it needs more than one system or teaches physics that no tutorial covers. When the checks report a page above its length, split it into pages that each answer one question, or promote a how-to that teaches a workflow to a tutorial. A page that answers one question and is still longer, such as a gallery of plots, is kept whole with an override (see "When a request conflicts with these rules").

When a request fits neither type (a long derivation, a design note), say so: derivations belong in `docs/about/theory.md` or `docs/internals/`, and design notes in `docs/internals/`.

## Workflow

1. **Read before writing.** Read the base example, if the page has one (`examples/.../main.py`, inputs, README), the PAOFLOW methods the page calls and their docstrings, and the existing pages that already cover nearby concepts. You need the code to know what is computed, and the neighbouring pages to know what is already explained (see "Say it once").
2. **Find the equation the code evaluates** for each step that gets an equation. Write it down from the source, signs, factors, unit conversions and normalisations included. If it differs from the textbook form, the page shows what the code does and you report the discrepancy to the user. Never write the textbook version silently, and never change the code to match the page.
3. **Write the script and run it** (see "Running the code" and "Assets and budgets"): the notebook of a tutorial, the `main.py` and `figures.py` of a how-to. Every number, table and figure on the page comes from that run, and the asserts check the numbers the conclusions rest on.
4. **Plan the figures, then write the page** from the template, in the style below. Decide first what the reader must see at each step and draw it (see "Where a picture carries the explanation"); the text then says what the figure cannot. When the draft is done, review it in two passes. First, for each paragraph, state its point in one sentence; if you cannot, rewrite the paragraph. Then read the first sentences of each section alone and check that they outline its argument (see "Connecting the facts"). Finally, read the page paragraph by paragraph and ask whether a figure would carry that paragraph better.
5. **Register it**: a tutorial gets a grid card plus a toctree entry, a how-to a list entry (a link with its title) plus a toctree entry (see the templates). The page test collects the new page by itself; add its name (`tutorialNN`, or the how-to's `<task>`) to the `page` matrix of the `docs-page-tests` job in `.github/workflows/ci.yaml`, which runs one page per job.
6. **Run the checks** at the end of this file and fix what they report. They are a floor, not the whole review.
7. **Report** briefly: what you wrote, every place where code and textbook physics disagree, every number or reference you could not confirm, every cell left for the user to run, the run time of the page, every concept whose home page or internals page is missing (see "Say it once" and "Building the explanation"), and every place where a figure would help but you did not make one, with the reason (no data from the run, a heavy run still pending).

## Running the code

The docs build does not execute notebooks (`nb_execution_mode = 'off'`), so whatever the committed notebook and the committed figures show is what the website shows. A cell without stored output shows no figure on the website. To keep the pages from drifting as the code changes, the page test `tests/integration/docs/test_pages.py` re-runs every tutorial notebook and every how-to `main.py` on each pull request into `develop`, in a scratch directory with the release asset of the page. It fails when a call raises, which catches renamed methods and arguments, and when an assert of the page fails, which catches a number on the page that the code no longer gives. It does not compare figures or stored outputs: re-execute and commit them when the text or a result changes.

- **The scripts are the page's code.** A tutorial's code is its notebook: the website hides its plotting cells (`hide-input`) and its asserts (`remove-cell`), and the downloaded notebook holds both. A how-to has two scripts in `docs/tutorials/howtos/<task>/`, because a reader who downloads the workflow should get the workflow and nothing else. `main.py` is the workflow the page teaches, every line of it shown on the page, split into blocks with `literalinclude` at plain step comments (see `references/howto.md`), so the page cannot show code that the test does not run and the reader downloads exactly the code the page explains. `figures.py` holds everything else: it reads the files `main.py` writes and its printed output, saved as `main.log`, draws every figure of the page and asserts the numbers. A step the page explains as something the reader does (a fit, a grouping of orbitals, a comparison) belongs in `main.py`; code that only draws a figure or checks a number the page quotes belongs in `figures.py`.
- **Asserted numbers.** Each script asserts the few numbers, about five at most, that the conclusions of the page rest on, with a tolerance at the precision the prose quotes: a gap quoted as 0.754 eV is asserted to 1 meV. Each assert names its quantity in its message (`assert abs(gap - 0.754) < 1e-3, f'indirect gap {gap:.4f} eV'`), so that a failure tells the developer which sentence to revise. In a how-to the asserts are at the end of `figures.py`, which takes the values from the output files of `main.py`, or from `main.log` for a value that `main.py` only prints; in a tutorial they are in a last code cell with the tag `remove-cell`, which the website hides.
- **Data** comes from a GitHub release (`tutorial-assets-vN` or `integration-assets-vN`), never from a path inside the repository: the reader downloads the notebook or the script alone. Confirm that the asset exists with `gh release view <tag>`, then name the tag and the file in the download tip. Code sets `savedir` to the extracted directory next to the script (`savedir = 'tutorial01/silicon.save'`), never `../../examples/...` and never an absolute path. In a tutorial, the setup cell (see `references/tutorial.md`) also downloads and extracts the asset when `savedir` does not exist, so that the downloaded notebook runs as it is; a how-to names the asset in its tip and leaves the download to the reader.
- **Cheap runs** (within the runtime budget of "Assets and budgets"): run them yourself with the Python of the project environment. Execute a notebook in place with `jupyter execute --inplace <notebook>` from a scratch copy whose working directory holds the extracted dataset, then copy the executed notebook back. Run a how-to from a scratch directory as the page test does, `python main.py > main.log` and then `python figures.py`, and copy the figures to `docs/_static/images/howtos/<task>/`. Then run the page test for the page: `pytest tests/integration/docs -k <page>`.
- **Heavy runs** (larger grids, SOC supercells, anything that needs the cluster): time the first steps on a reduced grid to estimate the cost. Then hand the user the exact command to run, for example `jupyter execute --inplace tutorialNN.ipynb` or `python main.py > main.log` in their cluster environment, together with the output files and the log you need back. Mark every number that is still missing as `{TODO: <quantity> from run}` in the text, and fill the numbers in only from the user's outputs. A page whose run stays above the budget follows the rule for heavy pages in "Assets and budgets".
- A snippet you have not run does not go on a page. If a run fails, report the failure; do not reword the page around it.

## Assets and budgets

Both page types are made of the same three parts:

|         | Tutorial                                                                                                                           | How-to                                                                                                             |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| Inputs  | A minimal `.save` directory (`atomic_proj.xml`, `data-file-schema.xml`, the pseudopotential files) under `tutorialNN/` in `tutorial_assets.tar.gz` | The same form under `<task>/` in `howto_assets.tar.gz`. None for a page that builds its model in code              |
| Scripts | The notebook                                                                                                                       | `docs/tutorials/howtos/<task>/main.py` (the workflow) and `figures.py` (figures and asserts)                       |
| Outputs | Stored in the notebook                                                                                                             | PNGs in `docs/_static/images/howtos/<task>/`, written by `figures.py`, and log excerpts pasted on the page         |

The inputs are DFT data, so a change in PAOFLOW's code does not invalidate them, and the release tag is bumped only when an input changes. Their generators are `.github/assets_generation/tutorials/build_assets.py` and `.github/assets_generation/howtos/build_assets.py`. A release asset holds PAOFLOW output files only for a heavy page (below), and then `build_assets.py` records the run that made them, because output files go stale as the code changes and a later developer must be able to remake them.

The budgets keep the page test fast enough to run on every pull request and the repository small:

- **Run time**: at most 5 minutes per page on a 4-core runner. For a heavier how-to, the asset folder `<task>/` holds what `main.py` writes, its output files and `main.log`, at the paths `main.py` writes them, and the page's name goes into the `PRECOMPUTED` set of the page test. On pull requests the test then runs only `figures.py` inside that folder, which takes seconds; running `main.py` is a separate case, `<task>-workflow`, marked `@pytest.mark.slow`. The tip of the page tells the reader that `figures.py` run inside the extracted folder redraws the figures without the calculation.
- **Inputs**: at most 5 MB compressed per page.
- **Figures**: PNG at 150 dpi, at most 200 KB each.
- **Notebooks**: outputs stay stored. Re-execute and commit a notebook only when its text or one of its results changes, because every re-execution adds the full outputs to the history of the repository.

## Say it once

The website is read as a whole, and repeated paragraphs make it long and hard to scan. Each concept has one home, and every other page links to it in one clause.

- Tutorial 01 is the home of the core chain: projections, projectability, $H(\mathbf{R})$, interpolation, diagonalisation, DOS. Later pages link back to it instead of re-deriving.
- The home of a method's parameters is its docstring (rendered under `docs/api/`). A page explains how to choose a parameter for this material and links to the API entry for the full list: ``{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.bands` ``.
- Links: `[Tutorial 01](../tutorial01.ipynb)`, `[internal basis](../internals/internal-basis.md)`, headings with `[text](page.md#heading-anchor)` (MyST anchors are on down to level 4).
- A how-to does not repeat the workflow of its base example. It shows only what changes and links to the example.
- A fact that holds for a method in general (a default, the files it writes, the energy range it is clipped to) has its home in the method's docstring or in the tutorial that introduces the method. Another page mentions it only when its situation changes what the reader should do, and then as a recommendation: what to do, followed by the reason in this situation, with a link to the home. A bare fact ("`do_pdos=False` turns them off") leaves the reader to work out why it is on this page. Write instead: "For a cell with many atoms, pass `do_pdos=False` to `dos()` unless you need the projected DOS, because by default it writes one file per orbital, thousands of files for a large supercell."
- A link that says "see X" must land on text that explains the concept. Open the target and check. If the home page does not explain it, the explanation goes on the home page: tell the developer, and do not define the concept on the linking page instead.

## Building the explanation

The page is read top to bottom by someone who has not seen the code. Three things decide whether it can be followed: the order in which concepts arrive, whether each new concept is explained enough to be used, and whether each statement is about the reader's calculation or about the implementation.

**Order.** Each sentence must be understandable from what is above it. When a sentence needs a concept that a later section treats in full, give the concept a one-clause definition where it first appears and link forward for the detail: "the small hoppings are dropped at the end of `pao_hamiltonian()` (see [Truncating the Hamiltonian](#truncating-the-hamiltonian))". A forward link adds detail; it never supplies the meaning. If no one-clause definition is possible, reorder the sections. Keep the first mention short: a reader needs enough to follow the sentence, not the whole concept.

**Constraints on the order of calls** are shown as code, with the call in its place in the script and a short comment, instead of a sentence listing what must come before and after.

**New concepts.** A concept the page's workflow depends on gets a short paragraph the first time it appears: what happens, in which order, and how it differs from what the reader already knows (usually the dense pipeline of Tutorial 01). A concrete example or a schematic figure (see "Where a picture carries the explanation") does more than an abstract sentence. A method that behaves differently from its usual role, such as a call that computes nothing or a step that runs later than where it is written, is explained by why it behaves so and what that changes in the script. Every method, option and mechanism named anywhere on the page, the troubleshooting table included, is introduced in the body first.

**Reasons and consequences in the reader's terms.** Give the reason for a setting and the consequence of a condition in terms of the calculation: run time, memory, the system size at which it fails, which results change and by how much, whether output files stay comparable between runs. Steps of the implementation ("the probe is skipped", "the margin is not added") are not reasons. "A pass that contains a Hall property solves the full spectrum" tells the reader nothing until it continues: "so each k-point needs a full $n_\mathrm{awf} \times n_\mathrm{awf}$ diagonalisation, and above `dense_n_max` orbitals the run stops with an error".

**Software detail.** It stays on the page only when the reader can act on it: memory per MPI rank that decides how many ranks to start, a call that must precede another, a limitation that decides which route to take. Exception classes, internal function names and statements that only confirm default behaviour ("the k-points are divided among the ranks as usual") go. A limitation is stated by what it means for the calculation (which properties are unavailable, and how to compute them anyway); the literal error message belongs in the troubleshooting table.

**Validation statements.** Agreement with another code path or a reference ("reproduces the dense DOS exactly") goes on the page only where it tells the reader what controls the error of their result or what a check should show, and then once: in the checking section if the page has one, otherwise next to the output it concerns. Benchmarks and error sweeps stay in the report to the developer.

**Beyond the cases shown.** A page whose change affects many properties (a backend, a global flag) shows one or two and then tells the reader how to run any other: whether the call is unchanged, and where the list of supported methods is (the API page, or a table on the page).

**Internals.** How the implementation works (solvers, data layout, parallel distribution) is explained in `docs/internals/<topic>.md`, not on the page. Where a mechanism has a consequence for the result or the cost, the page states the consequence and links to the internals page in one clause, so the reader knows why it matters. The "Where to go next" section links the internals page again for readers who want the mechanism. If no internals page exists, tell the developer instead of writing the mechanism into the tutorial or how-to.

## Describe the code as it is now

The reader has never seen earlier versions and was not part of the conversation that produced the page. The page describes current behaviour as if it had always been so.

- No history: not "previously", "now uses", "no longer", "was changed", "the old implementation", "renamed from", "the fixed version".
- No traces of the request: not "as requested", "the new option", "added to support".
- No deprecation hints or migration notes. When the code changes, rewrite the affected text so it stands on its own.
- When a behaviour is easy to get wrong, state it as a fact: "`fused()` must be called before the first property that triggers the mesh pass."

This section is about the history of the code. Comparing with an earlier page or an earlier step of the same page ("in contrast to Tutorial 01, where ...") is a different matter and is often what the reader needs; "Openings" says where on the page such a comparison goes.

## Writing style

Write like a careful physicist writing a methods section that walks a colleague through the calculation: precise, plain, specific, and connected. A page is an argument, not a list of facts: the reader should never have to work out why a sentence follows the one before it, or why a paragraph follows the one before it. "Connecting the facts", after this section, says how to make those links explicit, from the opening of a page down to a single sentence.

**Voice.** Address the reader as "you" throughout: "You build the Hamiltonian from the projections", "Choose `nk=2000` so that the band crossings near $L$ are resolved." Use the imperative for instructions ("Download the dataset and set `savedir`"). Where the subject of a sentence is the code or the calculation, make it the subject: "`projectability()` keeps the states with $p_{n\mathbf{k}} > 0.95$". Do not switch to "we", "our", "let's" or "one can" partway through.

**Physics depth.** For each step, give the defining equation as the code evaluates it, then one to three sentences of intuition: what the quantity means for an electron in the crystal, and what the reader should see in the output. Name the approximations explicitly: constant relaxation time, rigid band, independent particle, PBE gap, no local-field effects, the projectability cutoff. A reader needs to know what a result is not valid for. Leave derivations out and link to a reference or an internals page for them. Quote experiment or literature values only when the citation is certain (authors, journal, volume, page, year); otherwise ask the user.

**Numbers.** Give quantities with units and the precision the run supports ("indirect gap 0.61 eV", not "about 0.6"). Every number comes from the run or from a cited source.

**Jargon.** Use a technical term only when the physics needs it and no plain phrase is as precise. Terms that every DFT user knows need no explanation: Kohn–Sham state, k-point, PBE, pseudopotential, Fermi level. Terms that are specific to PAOFLOW or to another field do. Examples are projectability, PAO shift, cell doubling, the energy window of the sparse backend, and Berry curvature in a transport page. Define such a term in the sentence where it first appears on the page, or link to its definition in Tutorial 01, the API page or a cited reference. Spell out each acronym at its first use on each page (PAO, PDOS, SHC, AHC, NEGF). Do not use in-house shorthand as nouns: code variable names standing for a concept ("the `nawf` grows"), developer labels ("the dense branch", "nev padding", "the probe", "the engine") or names of internal functions the reader never calls. Say what they mean ("the number of orbitals grows"). When the page needs a name for a mechanism it uses repeatedly, define it once in plain terms ("one loop over all k-points of the mesh, called a pass") before using it.

Avoiding jargon is not the same as simplifying the physics. Do not explain what the reader already knows (what a band structure or reciprocal space is). Do not use analogies that lose precision, and do not reassure ("don't worry", "it's simple", "just run"). Keep the equations, units and approximations. What changes is that every term on the page is one the reader knows or has just been given.

**Titles and headings.** A title says what the page computes, for which system or with which method: "Tutorial 02: Boltzmann transport in silicon", "How to add spin–orbit coupling to a scalar-relativistic calculation". A section heading says what the step does: "3. Projectability: which bands the PAO basis can represent". Avoid catchy or clever titles: no puns, no questions, no alliteration, no exclamations, no teaser subtitles ("Silicon: the workhorse of the semiconductor world"), no superlatives. The same goes for catchphrases in the text ("the secret sauce", "where the magic happens", "the heavy lifting", "say goodbye to"). Each says nothing a plain sentence would not say better.

**Math.** Use `$...$` inline and `$$...$$` for display equations, with standard LaTeX, never Unicode Greek or symbols in prose (write `$\Gamma$`, not Γ; `$\varepsilon_{n\mathbf{k}}$`, not ε). Use `\mathbf{k}`, `\mathbf{R}` for vectors, matching the notation below.

**Typography.**

- No em dashes in prose. Use a comma, a colon, parentheses or a new sentence.
- No bold for emphasis. Bold is reserved for short labels that open a line, such as `**Output:**` or `**What you will compute**`.
- No emojis, no exclamation marks.
- Lists only for genuinely parallel items (output files, allowed values of a flag, prerequisites), with as many items as the content has. Explanations are prose.

**Banned words and phrases** (the checks grep for them): "delve", "seamless(ly)", "it's worth noting", "it is worth noting", "note that", "it is important to", "crucial", "robust", "powerful", "leverage", "utilize", "facilitate", "comprehensive", "in essence", "essentially", "simply", "basically", "under the hood", "a wide range of", "cutting-edge", "state-of-the-art", "streamline", "elegant", "plays a key role", "in summary", "to summarize", "in short", "in conclusion", "let's", "dive into", "deep dive", "game changer", "game-changer", "unlock", "supercharge", "empower", "out of the box", "made easy", "a breeze", "heavy lifting", "secret sauce", "where the magic happens", "best of both worlds", "next level", "in a nutshell", "the power of", "don't worry", "Overall," opening a sentence.

**Banned habits.**

- A closing paragraph or section that recaps the page. The last section adds information; "Where to go next" with links is allowed, a recap is not.
- Restating the section heading as the first sentence of the section.
- Marketing adjectives ("efficient", "fast", "flexible", "easy") without a number or a comparison behind them.
- Hedging filler ("can be used to", "is designed to", "allows you to"). Say what it does.

## Connecting the facts

A page written as one true sentence after another still leaves the reader to rebuild the argument: which fact is the reason, which is the consequence, which is the exception. Each of the rules below makes one kind of link explicit. They apply to both page types.

**Openings.** The first sentence of a page, a section or a paragraph says what the reader does or gets there. Earlier material, such as Tutorial 01 or the previous section, follows as a subordinate clause or a second sentence: "In this section you compute $\mu(T)$ at fixed carrier density, in contrast to Section 2, where $\mu$ was a free parameter." Lead with the positive statement, not with what something is not. Opening with the earlier material ("In Tutorial 01 you did X. Here you do Y") makes the reader hold the old idea in mind before they know why it matters. The exception is a section that fixes a problem the reader would hit: then the problem comes first, because it is the reason for the section ("A 500-atom cell exhausts the memory of a node. To avoid this, ...").

**Paragraphs.** Each paragraph answers one question the reader has at that point. It opens with the answer, and each following sentence supports or extends the sentence before it: the reason, the evidence, the consequence, the limit. A paragraph whose sentences could be shuffled without loss is a list of facts; find the question it answers and rewrite it from there.

**Between paragraphs.** The first sentence of a paragraph says how it relates to the paragraph before it: the next step, a consequence, an exception, or an alternative. It does so by picking up a term the previous paragraph ended on ("These expressions leave the relaxation time open."). The test is that the first sentences of a section, read alone, give the outline of its argument.

**Connectors.** Within a paragraph, when a sentence stands in a logical relation to the one before it, open it with a word that names the relation, so that the reader does not have to infer it: a solution to the problem just stated ("To avoid this", "Instead"), a consequence ("Therefore", "As a result", "so"), a cause ("because", "since"), a contrast ("However", "In contrast", "On the dense pipeline ..., whereas"), an example ("For example", "On Si"), or a condition ("When", "If"). Two short sentences in a cause-and-effect relation can also be joined into one with "because" or "so", and an instruction carries its reason in the same sentence. A sentence that only continues a description or a sequence of steps needs no connector, and a connector must state the real relation: "however" before a sentence that does not contrast misleads the reader.

Before: "Defect supercells, slabs, interfaces and heterostructures therefore exhaust the memory of a node. The sparse backend never forms these arrays."

After: "Defect supercells, slabs, interfaces and heterostructures therefore exhaust the memory of a node. To avoid this, the sparse backend never forms these arrays."

**Comparisons.** State a comparison from the reader's side: what they keep doing, what they change, what they leave out. A sentence that only equates two things ("the calls are those of a DFT run") leaves the reader to work out what it means for their script. Write instead: "the calls remain the same as the ones you would use for a DFT-based workflow".

### Before and after

Pair A, an opening. From the tight-binding how-to, its first paragraph.

Before:

> Every PAOFLOW property is computed from a real-space Hamiltonian $H_{\alpha\beta}(\mathbf{R})$. In the DFT workflow of Tutorial 01 it comes from projecting the Kohn–Sham states on pseudo-atomic orbitals (PAOs). With the `model` argument of `PAOFLOW` you give it instead as a tight-binding model: one of the built-in models, a Slater–Koster model with your own parameters, or a model built with the PythTB package.

After:

> On this page you run PAOFLOW on a tight-binding model instead of a DFT calculation. The `model` argument of `PAOFLOW` supplies the real-space Hamiltonian $H_{\alpha\beta}(\mathbf{R})$ directly, as one of the built-in models, a Slater–Koster model with your own parameters, or a model built with the PythTB package, in contrast to Tutorial 01, where it comes from projecting the Kohn–Sham states on pseudo-atomic orbitals (PAOs). Every property is computed from $H_{\alpha\beta}(\mathbf{R})$ alone, so the calls that follow the constructor remain the same as the ones you would use for a DFT-based workflow.

What changed. The first sentence says what the reader does on the page. Tutorial 01 follows as the contrast. The last sentence says what the reader keeps doing.

Pair B, inside a paragraph. From Tutorial 02, Section 1.

Before:

> First, interpolate onto a $36\times36\times36$ mesh instead of the default $24\times24\times24$. Near the conduction-band minimum (CBM) the default mesh has few states: only 12 of the states on the $24^3$ mesh lie within 50 meV above the CBM, against 60 on the $36^3$ mesh. With $10^{19}$ electrons per cm$^3$, $\mu$ at 300 K is 0.649 eV on the $24^3$ mesh, 0.700 eV on the $36^3$ mesh and 0.714 eV on a $48^3$ mesh. The $36^3$ mesh is therefore within 14 meV of the $48^3$ result, and needs 2.9 GB of memory, against about 6 GB for $48^3$.

After:

> First, interpolate onto a $36^3$ mesh instead of the default $24^3$, because at this doping $\mu$ is set by the few states just above the conduction-band minimum (CBM), and the default mesh samples them too coarsely: only 12 of its states lie within 50 meV above the CBM, against 60 on the $36^3$ mesh. As a result, $\mu$ at 300 K for $10^{19}$ electrons per cm$^3$ moves from 0.649 eV on the $24^3$ mesh to 0.700 eV on $36^3$ and 0.714 eV on $48^3$. The $36^3$ mesh is the compromise between accuracy and memory: it is within 14 meV of the $48^3$ value and needs 2.9 GB, where $48^3$ needs about 6 GB.

What changed. The reason for the instruction is in the same sentence as the instruction. The numbers for $\mu$ are introduced as the consequence of the sampling. The last sentence says what the choice trades.

Pair C, between paragraphs. From Tutorial 02, Section 2, the paragraph after the equations for $\sigma$, $S$ and $\kappa$.

Before:

> Without scattering channels, `transport()` uses the same relaxation time for every state, $\tau_{n\mathbf{k}} = 1$ s. The files therefore contain $\sigma/\tau$, $\kappa/\tau$ and the power factor $S^2\sigma/\tau$, while $S$, a ratio of moments, does not depend on $\tau$. The bands are rigid: changing $\mu$ moves the electrons through the bands of the undoped crystal, as if Si were doped without changing its electronic structure.

After:

> These expressions leave the relaxation time $\tau_{n\mathbf{k}}$ open. Without scattering channels, `transport()` sets it to 1 s for every state, so the files contain $\sigma/\tau$, $\kappa/\tau$ and the power factor $S^2\sigma/\tau$, while $S$, a ratio of moments, does not depend on $\tau$. The dependence on $\mu$ is a second approximation, that of rigid bands: changing $\mu$ moves the electrons through the bands of the undoped crystal, as if Si were doped without changing its electronic structure.

What changed. The first sentence picks up the equations above it. The rigid-band statement is introduced as a second approximation instead of arriving unannounced.

## Figures

### Where a picture carries the explanation

A well-defined picture is worth a thousand words. A reader takes in a supercell, a Brillouin zone or an energy window at a glance, where the same content in prose is a paragraph they must rebuild in their head, and often rebuild wrongly. Treat figures as part of the explanation, not as decoration of the results: wherever the reader has to picture something to follow the text, show it. These are the places where a figure usually does better than prose:

- Geometry in real space: cells and supercells, neighbour shells and cutoff radii, atom mappings, stacking and twist, slabs and interfaces.
- Reciprocal space: Brillouin zones, high-symmetry paths, k-meshes, the folding of one zone into another.
- Ranges on an energy or frequency axis: energy windows, smearing widths, projectability cutoffs, the bands a setting keeps or drops.
- A paragraph of numbers read off a result (splittings, shifts, weights, gaps). Show a zoom of the region with the numbers annotated, and keep in the text only the one or two that carry the conclusion.
- A comparison of two methods, two parameter values, or a run before and after a change: overlay them in one panel, in the colours used for the same series elsewhere.
- The dependence of a result on a parameter (convergence, cost, error): a curve, not a table, unless the reader needs to look up exact values.
- An order of steps or a flow of data that differs from Tutorial 01: a small diagram.

A well-defined figure makes one point. It shows the quantity the text discusses, in the range where the point is visible (zoom in instead of making the reader search), with labelled axes and units, and with every colour, line style and marker explained in the legend or in the sentences after it. Draw it from the page's run, the actual cells, bands and windows of the system, rather than from an idealised sketch, wherever the concept has data behind it. A figure that needs a long paragraph to decode is not yet well defined: split it into panels, zoom it or annotate it. Figures that repeat what a neighbouring figure shows stay out.

### Rules for every figure

- Plot with GPAO (`from PAOFLOW import GPAO`) whenever it has a function for the quantity: bands, DOS and PDOS, conductivity, Seebeck coefficient, Berry curvature, Hall conductivities, dielectric and optical spectra, phonons and the electron–phonon outputs. The reader learns the plotting calls they will use on their own runs, and the parsing of PAOFLOW's output files stays in one place. Pass the palette below through the `col`/`cols` arguments, because the GPAO defaults are not in it. Write matplotlib code only for a quantity GPAO has no function for, and when a GPAO function lacks something the figure needs (an axis label, a color argument), fix it in `GPAO.py` instead of rewriting the plot by hand.
- Colors: Deep Blue `#004488`, Vermillion `#D55E00` and Yellow-Amber `#FFC20A` first, then Okabe–Ito `#009E73`, `#56B4E9`, `#CC79A7`, `#000000`. Use `#8A6800` for amber text and spines. One color per physical series, kept the same across all panels and pages (for example, the PAO bands are always `#004488`, and DFT reference points are always `#D55E00`).
- Axis labels carry the quantity and the unit: `$E - E_F$ (eV)`, `$\sigma_{xx}$ ($\Omega^{-1}\,\mathrm{m}^{-1}$)`. State the energy zero in the caption sentence that follows the figure.
- In notebooks, a plotting cell carries the cell tag `hide-input`. The website then shows the figure with a "Show code cell source" toggle, and the downloaded notebook still holds the full code. Each plotting cell is standalone (it reads the output files of the run) so that it works in the downloaded notebook.
- In how-tos, figures are PNGs in `docs/_static/images/howtos/<task>/`, written by the page's `figures.py`, and embedded with `![alt text describing the physics](../../_static/images/howtos/<task>/<name>.png)`. Alt text states what the figure shows ("Si bands along L–Γ–X, PAO vs DFT"). Because the script that draws a figure is kept, a later developer redraws every figure of the page by running it again when the code changes.
- Save figures as PNG at 150 dpi (`fig.savefig(..., dpi=150)`), at most 200 KB each. A larger file usually means too many points or markers; thin the data rather than lower the resolution.
- After the figure, a few sentences tell the reader what to see in it, with numbers, and which run it shows where that is not obvious from the page: the dataset and the options that shape it (truncation, energy ranges, windows, k-mesh).

## Notation

Use these symbols, taken from Tutorial 01 and Tutorial 04. Add a symbol here when a new page introduces one that other pages will reuse.

| Symbol                                                                              | Meaning                                                    |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| $\psi_{n\mathbf{k}}$, $\varepsilon_{n\mathbf{k}}$                                   | Kohn–Sham state and eigenvalue of band $n$ at $\mathbf{k}$ |
| $\phi_\alpha$                                                                       | pseudo-atomic orbital $\alpha$ (PAO)                       |
| $A_{n\alpha}(\mathbf{k}) = \langle\phi_\alpha\|\psi_{n\mathbf{k}}\rangle$           | projection amplitude                                       |
| $p_{n\mathbf{k}} = \sum_\alpha \|A_{n\alpha}(\mathbf{k})\|^2$                       | projectability (code threshold `pthr`, default 0.95)       |
| $S(\mathbf{k})$                                                                     | PAO overlap matrix                                         |
| $H_{\alpha\beta}(\mathbf{R})$, $\tilde H(\mathbf{k})$                               | real-space PAO Hamiltonian and its Fourier interpolation   |
| $N_k$                                                                               | number of k-points of the DFT grid                         |
| $N_\mathrm{mesh}$                                                                   | number of k-points of the interpolated mesh                |
| $n_\mathrm{awf}$                                                                    | number of PAOs in the cell (`nawf` in the code)            |
| $E_F$                                                                               | Fermi energy; plots use $E - E_F$ unless stated            |
| $\mathbf{v}_{n\mathbf{k}} = \hbar^{-1}\nabla_{\mathbf{k}}\varepsilon_{n\mathbf{k}}$ | band velocity                                              |

## When a request conflicts with these rules

Several developers use this skill, and a rule is sometimes broken on purpose. A rule is never broken silently, either by you or at someone's request.

1. If the developer asks for something that breaks a rule in this skill, or the task cannot be done without breaking one, stop before writing that part. Examples: a catchy title, an em dash, a tutorial on data inside the repository, a number we did not run, a figure in a non-palette color. Name the rule (its section here, and its check key below if it has one), say in one sentence why it exists, and offer the version that follows it.
2. Proceed with the exception only after the developer explicitly says to override that rule in their reply. Silence, an earlier override on another page or in an earlier session, or a general "do whatever" does not count.
3. Record every override only in the commit message, never on the page: one trailer line per page and rule, `Override: <page> <key> | <reason>`, for example `Override: docs/tutorials/howtos/plotting.md color | white midpoint of a diverging map`. You do not commit, so give these lines in the final report for the developer to paste into the commit message.

The same applies to a page that already breaks a rule when you are asked to revise it. Point out the violations the checks report, and fix them unless a commit of the page records an override for that rule (the checks read the `Override:` trailers from `git log` of the page). If the developer wants to keep one, they give the override and you report its trailer line.

Check keys: `banned-phrase`, `history`, `em-dash`, `bold`, `greek`, `voice`, `todo`, `repo-path`, `color`, `hide-input`, `no-output`, `registration`, `base-example`, `length`, `script`. Rules without a key (jargon, titles, physics depth, numbers from the run, the rules of "Connecting the facts") are checked by reading. Their overrides use the section name as the key, for example `Override: docs/tutorials/tutorial01.ipynb titles | ...`.

## Checks

Run from the repository root, giving the pages you wrote or revised. The checks waive a rule for a page when a commit of that page has its `Override:` trailer. For an override the developer gave in this session, which is not committed yet, add `--override=<page>:<key>`.

````bash
python - docs/tutorials/tutorial01.ipynb docs/tutorials/howtos/plotting.md <<'EOF'
import json, re, sys
from pathlib import Path

BANNED = ["delve", "seamless", "it's worth noting", "it is worth noting", "note that",
          "it is important to", "crucial", "robust", "powerful", "leverage", "utiliz",
          "facilitate", "comprehensive", "in essence", "essentially", "simply", "basically",
          "under the hood", "a wide range of", "cutting-edge", "state-of-the-art", "streamline",
          "elegant", "plays a key role", "in summary", "to summarize", "in short",
          "in conclusion", "let's", "dive into", "deep dive", "game changer", "game-changer",
          "unlock", "supercharge", "empower", "out of the box", "made easy", "a breeze",
          "heavy lifting", "secret sauce", "where the magic happens", "best of both worlds",
          "next level", "in a nutshell", "the power of", "don't worry"]
HISTORY = ["previously", "now uses", "no longer", "was changed", "has been changed",
           "the old ", "renamed", "as requested", "the new option", "deprecated", "used to be"]
PALETTE = {"#004488", "#D55E00", "#FFC20A", "#8A6800", "#009E73", "#56B4E9", "#CC79A7", "#000000"}
GREEK = re.compile(r"[Α-ωϑϕϵ]")
MAX_WORDS = {"howtos": 1500, "tutorials": 3000}  # prose words per page, see the table at the top
OVERRIDE = re.compile(r"^Override:\s*(\S+)\s+([\w-]+)\s*\|(.*)$", re.M)
PENDING = {tuple(a.split("=", 1)[1].rsplit(":", 1)) for a in sys.argv[1:] if a.startswith("--override=")}

def overrides(path):
    """Overrides of this page: Override: trailers in the commits of the page, plus --override flags."""
    import subprocess
    log = subprocess.run(["git", "log", "--format=%B", "--", str(path)], capture_output=True, text=True).stdout
    found = {key: why.strip() for page, key, why in OVERRIDE.findall(log) if Path(page) == path}
    found.update({key: "given in this session, not committed" for page, key in PENDING if Path(page) == path})
    return found

def blocks(path):
    """Yield (kind, text, tags, has_output) for markdown prose and code."""
    if path.suffix == ".ipynb":
        for c in json.loads(path.read_text())["cells"]:
            yield (c["cell_type"], "".join(c["source"]), c.get("metadata", {}).get("tags", []),
                   bool(c.get("outputs")))
    else:
        parts = re.split(r"^(```.*)$", path.read_text(), flags=re.M)
        code, info = False, ""
        for p in parts:
            if p.startswith("```"):
                code, info = not code, p.strip("`").strip()  # info: the language or directive
                continue
            yield ("code", info + p, [], True) if code else ("markdown", p, [], True)

def prose(text):
    """Markdown with comments, inline code, math and link targets removed."""
    text = re.sub(r"^%.*$", "", text, flags=re.M)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"`[^`]*`|\$[^$]*\$", "X", text)  # placeholder keeps **`flag`** intact

def words(p):
    """Number of prose words: no placeholders, MyST roles or directive lines."""
    p = re.sub(r"^:{3,}.*$|\{[\w:-]+\}", "", p, flags=re.M)
    return sum(1 for w in re.findall(r"[^\W\d_][\w'’-]*", p) if w != "X")

def check_code(text, say):
    """Checks that apply to any code the page shows or runs."""
    if re.search(r"(savedir|outputdir|open|loadtxt)\s*[=(][^\n]*['\"](/home|/Users|\.\./)", text):
        say("repo-path", "code points into the repository or an absolute path; use the downloaded dataset")
    for hexc in re.findall(r"#[0-9A-Fa-f]{6}\b", text):
        if hexc.upper() not in PALETTE:
            say("color", f"{hexc} is not in the palette")
    if re.search(r"\bcol(or|s)?\s*=\s*['\"](?!#)\w+['\"]|['\"]C\d['\"]", text):
        say("color", "named or cycle color in plot code; use palette hex codes")
    if re.search(r"\.plot_\w+\(", text) and not re.search(r"\bcols?\s*=", text):
        say("color", "GPAO call without col/cols falls back to non-palette defaults")

def check_blocks(page, lines, say):
    """Every line of main.py is shown by a literalinclude, split at unique step comments."""
    shown, comments = [False] * len(lines), set()
    pattern = r"^```\{literalinclude\}\s+(\S+)\n((?::[\w-]+:.*\n)*)```"
    for m in re.finditer(pattern, page.read_text(), re.M):
        if m.group(1) != f"{page.stem}/main.py":
            say("script", f"literalinclude of {m.group(1)}; the page shows only {page.stem}/main.py")
            continue
        opts = {k: v.strip().strip("'\"") for k, v in re.findall(r"^:([\w-]+):(.*)$", m.group(2), re.M)}
        ends = []
        for key, default in (("start-after", -1), ("end-before", len(lines))):
            if key not in opts:
                ends.append(default)
                continue
            hits = [i for i, line in enumerate(lines) if opts[key] in line]
            if len(hits) != 1:
                say("script", f"step comment {opts[key]!r} is on {len(hits)} lines of main.py, not 1")
                break
            ends.append(hits[0])
            comments.add(hits[0])
        else:
            for i in range(ends[0] + 1, ends[1]):
                shown[i] = True
    hidden = [i + 1 for i, line in enumerate(lines) if line.strip() and not shown[i] and i not in comments]
    if hidden:
        say("script", f"lines {hidden[:10]} of main.py are not on the page")

problems, overridden = [], []
for name in [a for a in sys.argv[1:] if not a.startswith("--")]:
    path = Path(name)
    found = []
    say = lambda key, msg: found.append((key, msg))
    allmd, nwords, fenced = "", 0, 0
    for kind, text, tags, has_out in blocks(path):
        if kind == "markdown":
            allmd += text + "\n"
            p = prose(text)
            nwords += words(p)
            low = p.lower()
            for w in BANNED:
                if w in low:
                    say("banned-phrase", f"'{w}': ...{p[max(0, low.find(w) - 40):low.find(w) + 40]!r}...")
            for w in HISTORY:
                if w in low:
                    say("history", f"'{w}': ...{p[max(0, low.find(w) - 40):low.find(w) + 40]!r}...")
            if re.search(r"^\s*overall,", p, flags=re.M | re.I):
                say("banned-phrase", "sentence opens with 'Overall,'")
            for m in re.finditer("—", p):
                say("em-dash", repr(p[max(0, m.start() - 40):m.start() + 20]))
            for m in re.finditer(r"\*\*[^*\n]+\*\*", p):
                line = p[p.rfind("\n", 0, m.start()) + 1:m.start()].strip()
                if not re.fullmatch(r"([-*|]|\d+\.)?", line):  # label opening a line or list item is allowed
                    say("bold", f"bold inside a sentence: {m.group()}")
            for m in GREEK.finditer(p):
                say("greek", f"Unicode '{m.group()}' in prose; write it as LaTeX")
            if re.search(r"\b(we|our|ours|us|let's)\b|\bone can\b", low):
                say("voice", "'we'/'our'/'one can' in prose; address the reader as 'you'")
            if "{todo" in low:
                say("todo", "unfilled {TODO: ...} placeholder")
        else:
            check_code(text, say)
            fenced += path.suffix == ".md" and bool(re.match(r"python\b", text))
            if path.suffix == ".ipynb":
                if re.search(r"plt\.|matplotlib|GPAO", text) and "hide-input" not in tags:
                    say("hide-input", "plotting cell without the 'hide-input' tag")
                setup = re.search(r"^savedir\s*=", text, re.M)  # the setup cell has no output
                quiet = setup or "remove-cell" in tags  # so has the hidden cell of asserts
                if text.strip() and not has_out and not quiet and not text.lstrip().startswith("#"):
                    say("no-output", f"code cell without stored output: {text.strip().splitlines()[0][:60]!r}")
    cap = MAX_WORDS.get(path.parent.name)
    if cap and nwords > cap:
        say("length", f"{nwords} words of prose, above {cap}: split, promote, or override")
    if fenced:
        say("script", f"{fenced} fenced python blocks; pull the code from main.py with literalinclude")
    if path.parent.name == "howtos":
        folder = path.parent / path.stem
        for script in (folder / "main.py", folder / "figures.py"):
            if script.is_file():
                check_code(script.read_text(), say)
            else:
                say("script", f"no {script}")
        if (folder / "main.py").is_file():
            check_blocks(path, (folder / "main.py").read_text().splitlines(), say)
        idx = (path.parent / "index.md").read_text()
        if not re.search(rf"^- \[.+\]\({path.stem}\.md\)$", idx, re.M) or not re.search(rf"^{path.stem}$", idx, re.M):
            say("registration", "missing list entry or toctree entry in docs/tutorials/howtos/index.md")
        m = re.search(r"^Base examples?:(.*)$", allmd, re.M)  # optional; checked when present
        paths = re.findall(r"(?<![/\w])examples/[\w./-]+", m.group(1)) if m else []
        if m and not paths:
            say("base-example", "'Base example:' line without an examples/... path")
        for ex in paths:
            if not Path(ex).exists():
                say("base-example", f"{ex} does not exist")
    elif path.name.startswith("tutorial"):
        if path.stem not in re.findall(r":link:\s*(\S+)", (path.parent / "index.md").read_text()):
            say("registration", "missing grid card in docs/tutorials/index.md")
    waived = overrides(path)
    problems += [f"{name}: [{k}] {msg}" for k, msg in found if k not in waived]
    overridden += [f"{name}: [{k}] overridden ({why})" for k, why in waived.items()]
print("\n".join(overridden + problems) or "no problems found")
sys.exit(1 if problems else 0)
EOF
````

Some hits are false positives. A quoted log line may contain "we", and "no longer" can be physics ("no longer projectable"). Do not record overrides for these: leave them and mention them in the report. Overrides are only for rules the developer chose to break.
