---
name: tutorials-howtos
description: "Write or revise the tutorials (docs/tutorials/tutorialNN.ipynb) and how-tos (docs/tutorials/howtos/<task>.md) of the Read the Docs site so they read like physics written for physicists. Use when: writing a new tutorial or how-to, turning an example into a how-to, revising, reviewing or proofreading an existing tutorial or how-to page, or adding one to the tutorials or how-to index, even if the user only says 'document this workflow' or 'write a page for this flag'."
user-invocable: true
---

# Tutorials and how-tos

The reader is a researcher who knows density functional theory but not PAOFLOW. They read the page on the Read the Docs website, and only some of them download the notebook to run it. Each page tells them which physical quantity a workflow gives, under which approximation, how to choose the parameters that control it, and how to tell whether the result can be trusted. Software detail goes in only when it changes what they can compute (memory that grows with cell size, MPI pools, the sparse backend).

The pages are written for people to read, not for agents to parse. The aim is to communicate ideas clearly to a colleague. That means no instructions addressed to tools, no checklists written for a model, and no prompt-like text. It also means no marketing language, which rarely communicates an idea. Write at the level of a good review article: the reader is an expert in a neighbouring field. Do not talk down to them, and do not make them decode in-house shorthand (see "Jargon").

There are two page types with different jobs. Read the matching template before writing:

|           | Tutorial                                                 | How-to                                                                               |
| --------- | -------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Job       | Teaches a complete workflow end to end, with its physics | Answers one practical question: how to set X up, which flags matter, how to check it |
| File      | `docs/tutorials/tutorialNN.ipynb`                        | `docs/tutorials/howtos/<task>.md`                                                    |
| Template  | `references/tutorial.md`                                 | `references/howto.md`                                                                |
| Builds on | Tutorial 01 (except Tutorial 01 itself)                  | an example under `examples/`, named on the page                                      |

When a request fits neither type (a long derivation, a design note), say so: derivations belong in `docs/about/theory.md` or `docs/internals/`, and design notes in `docs/internals/`.

## Workflow

1. **Read before writing.** Read the base example (`examples/.../main.py`, inputs, README), the PAOFLOW methods the page calls and their docstrings, and the existing pages that already cover nearby concepts. You need the code to know what is computed, and the neighbouring pages to know what is already explained (see "Say it once").
2. **Find the equation the code evaluates** for each step that gets an equation. Write it down from the source, signs, factors, unit conversions and normalisations included. If it differs from the textbook form, the page shows what the code does and you report the discrepancy to the user. Never write the textbook version silently, and never change the code to match the page.
3. **Run the code** (see "Running the code"). Every number, table and figure on the page comes from that run.
4. **Write the page** from the template, in the style below.
5. **Register it**: a grid card plus a toctree entry (see the templates).
6. **Run the checks** at the end of this file and fix what they report. They are a floor, not the whole review.
7. **Report** briefly: what you wrote, every place where code and textbook physics disagree, every number or reference you could not confirm, and every cell left for the user to run.

## Running the code

The docs build does not execute notebooks (`nb_execution_mode = 'off'`), so whatever the committed notebook contains is what the website shows. A cell without stored output shows no figure on the website, and a wrong cell ships unnoticed.

- **Data** comes from a GitHub release (`tutorial-assets-vN` or `integration-assets-vN`), never from a path inside the repository: the reader downloads the notebook alone. Confirm that the asset exists with `gh release view <tag>`, then name the tag and the file in the download tip. Code cells set `savedir` to the extracted directory next to the notebook (`savedir = 'silicon.save'`), never `../../examples/...` and never an absolute path.
- **Cheap runs** (finish in about 10 minutes on the laptop): run them yourself with `/home/anooja/miniconda3/envs/paoflow/bin/python`. Execute a notebook in place with `/home/anooja/miniconda3/envs/paoflow/bin/jupyter execute --inplace <notebook>`, from a scratch copy whose working directory holds the extracted dataset, and then copy the executed notebook back. Run a how-to's code blocks as one script.
- **Heavy runs** (larger grids, SOC supercells, anything that needs the cluster): time the first steps on a reduced grid to estimate the cost. Then hand the user the exact command to run, for example `jupyter execute --inplace tutorialNN.ipynb` in their cluster environment, or the how-to as a `main.py`, together with the output files you need back. Mark every number that is still missing as `{TODO: <quantity> from run}` in the text, and fill the numbers in only from the user's outputs.
- An `Examples`-style snippet you have not run does not go on a page. If a run fails, report the failure; do not reword the page around it.

## Say it once

The website is read as a whole, and repeated paragraphs make it long and hard to scan. Each concept has one home, and every other page links to it in one clause.

- Tutorial 01 is the home of the core chain: projections, projectability, $H(\mathbf{R})$, interpolation, diagonalisation, DOS. Later pages link back to it instead of re-deriving.
- The home of a method's parameters is its docstring (rendered under `docs/api/`). A page explains how to choose a parameter for this material and links to the API entry for the full list: ``{py:meth}`~PAOFLOW.PAOFLOW.PAOFLOW.bands` ``.
- Links: `[Tutorial 01](../tutorial01.ipynb)`, `[internal basis](../internals/internal-basis.md)`, headings with `[text](page.md#heading-anchor)` (MyST anchors are on down to level 4).
- A how-to does not repeat the workflow of its base example. It shows only what changes and links to the example.

## Describe the code as it is now

The reader has never seen earlier versions and was not part of the conversation that produced the page. The page describes current behaviour as if it had always been so.

- No history: not "previously", "now uses", "no longer", "was changed", "the old implementation", "renamed from", "the fixed version".
- No traces of the request: not "as requested", "the new option", "added to support".
- No deprecation hints or migration notes. When the code changes, rewrite the affected text so it stands on its own.
- When a behaviour is easy to get wrong, state it as a fact: "`fused()` must be called before the first property that triggers the mesh pass."

## Writing style

Write like a careful physicist writing a methods section that walks a colleague through the calculation: precise, plain, specific. Each sentence defines something, states a condition, tells the reader what to do, or tells them what to expect. Delete any sentence that does none of these.

**Voice.** Use the first person plural throughout: "We build the Hamiltonian from the projections", "We choose `nk=2000` so that the band crossings near $L$ are resolved." Do not switch to "you" or "one can" partway through. Use the imperative only in short instructions inside admonitions ("Download the dataset and set `savedir`").

**Physics depth.** For each step, give the defining equation as the code evaluates it, then one to three sentences of intuition: what the quantity means for an electron in the crystal, and what we should see in the output. Name the approximations explicitly: constant relaxation time, rigid band, independent particle, PBE gap, no local-field effects, the projectability cutoff. A reader needs to know what a result is not valid for. Leave derivations out and link to a reference or an internals page for them. Quote experiment or literature values only when the citation is certain (authors, journal, volume, page, year); otherwise ask the user.

**Numbers.** Give quantities with units and the precision the run supports ("indirect gap 0.61 eV", not "about 0.6"). Every number comes from the run or from a cited source.

**Jargon.** Use a technical term only when the physics needs it and no plain phrase is as precise. Terms that every DFT user knows need no explanation: Kohn–Sham state, k-point, PBE, pseudopotential, Fermi level. Terms that are specific to PAOFLOW or to another field do. Examples are projectability, PAO shift, cell doubling, the energy window of the sparse backend, and Berry curvature in a transport page. Define such a term in the sentence where it first appears on the page, or link to its definition in Tutorial 01, the API page or a cited reference. Spell out each acronym at its first use on each page (PAO, PDOS, SHC, AHC, NEGF). Do not use in-house shorthand as nouns: code variable names standing for a concept ("the `nawf` grows"), developer labels ("the dense branch", "nev padding") or names of internal functions the reader never calls. Say what they mean ("the number of orbitals grows").

Avoiding jargon is not the same as simplifying the physics. Do not explain what the reader already knows (what a band structure or reciprocal space is). Do not use analogies that lose precision, and do not reassure ("don't worry", "it's simple", "just run"). Keep the equations, units and approximations. What changes is that every term on the page is one the reader knows or has just been given.

**Titles and headings.** A title says what the page computes, for which system or with which method: "Tutorial 02: Boltzmann transport in silicon", "How to add spin–orbit coupling to a scalar-relativistic calculation". A section heading says what the step does: "3. Projectability: which bands the PAO basis can represent". Avoid catchy or clever titles: no puns, no questions, no alliteration, no exclamations, no teaser subtitles ("Silicon: the workhorse of the semiconductor world"), no superlatives. The same goes for catchphrases in the text ("the secret sauce", "where the magic happens", "the heavy lifting", "say goodbye to"). Each says nothing a plain sentence would not say better.

**Math.** Use `$...$` inline and `$$...$$` for display equations, with standard LaTeX, never Unicode Greek or symbols in prose (write `$\Gamma$`, not Γ; `$\varepsilon_{n\mathbf{k}}$`, not ε). Use `\mathbf{k}`, `\mathbf{R}` for vectors, matching the notation below.

**Typography.**

- No em dashes in prose. Use a comma, a colon, parentheses or a new sentence.
- No bold for emphasis. Bold is reserved for short labels that open a line, such as `**Output:**` or `**What we compute**`.
- No emojis, no exclamation marks.
- Lists only for genuinely parallel items (output files, allowed values of a flag, prerequisites), with as many items as the content has. Explanations are prose.

**Banned words and phrases** (the checks grep for them): "delve", "seamless(ly)", "it's worth noting", "it is worth noting", "note that", "it is important to", "crucial", "robust", "powerful", "leverage", "utilize", "facilitate", "comprehensive", "in essence", "essentially", "simply", "basically", "under the hood", "a wide range of", "cutting-edge", "state-of-the-art", "streamline", "elegant", "plays a key role", "in summary", "to summarize", "in short", "in conclusion", "let's", "dive into", "deep dive", "game changer", "game-changer", "unlock", "supercharge", "empower", "out of the box", "made easy", "a breeze", "heavy lifting", "secret sauce", "where the magic happens", "best of both worlds", "next level", "in a nutshell", "the power of", "don't worry", "Overall," opening a sentence.

**Banned habits.**

- A closing paragraph or section that recaps the page. The last section adds information; "Where to go next" with links is allowed, a recap is not.
- Restating the section heading as the first sentence of the section.
- Marketing adjectives ("efficient", "fast", "flexible", "easy") without a number or a comparison behind them.
- Hedging filler ("can be used to", "is designed to", "allows us to"). Say what it does.

## Figures

- Colors: Deep Blue `#004488`, Vermillion `#D55E00` and Yellow-Amber `#FFC20A` first, then Okabe–Ito `#009E73`, `#56B4E9`, `#CC79A7`, `#000000`. Use `#8A6800` for amber text and spines. One color per physical series, kept the same across all panels and pages (for example, the PAO bands are always `#004488`, and DFT reference points are always `#D55E00`).
- Axis labels carry the quantity and the unit: `$E - E_F$ (eV)`, `$\sigma_{xx}$ ($\Omega^{-1}\,\mathrm{m}^{-1}$)`. State the energy zero in the caption sentence that follows the figure.
- In notebooks, a plotting cell carries the cell tag `hide-input`. The website then shows the figure with a "Show code cell source" toggle, and the downloaded notebook still holds the full code. Each plotting cell is standalone (it reads the output files of the run) so that it works in the downloaded notebook.
- In how-tos, figures are PNGs in `docs/_static/images/howtos/<task>/`, produced by a script you ran, and embedded with `![alt text describing the physics](../../_static/images/howtos/<task>/<name>.png)`. Alt text states what the figure shows ("Si bands along L–Γ–X, PAO vs DFT").
- After the figure, a few sentences tell the reader what to see in it, with numbers.

## Notation

Use these symbols, taken from Tutorial 01 and the sparse how-to. Add a symbol here when a new page introduces one that other pages will reuse.

| Symbol                                                                              | Meaning                                                    |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| $\psi_{n\mathbf{k}}$, $\varepsilon_{n\mathbf{k}}$                                   | Kohn–Sham state and eigenvalue of band $n$ at $\mathbf{k}$ |
| $\phi_\alpha$                                                                       | pseudo-atomic orbital $\alpha$ (PAO)                       |
| $A_{n\alpha}(\mathbf{k}) = \langle\phi_\alpha\|\psi_{n\mathbf{k}}\rangle$           | projection amplitude                                       |
| $p_{n\mathbf{k}} = \sum_\alpha \|A_{n\alpha}(\mathbf{k})\|^2$                       | projectability (code threshold `pthr`, default 0.95)       |
| $S(\mathbf{k})$                                                                     | PAO overlap matrix                                         |
| $H_{\alpha\beta}(\mathbf{R})$, $\tilde H(\mathbf{k})$                               | real-space PAO Hamiltonian and its Fourier interpolation   |
| $N_k$                                                                               | number of k-points of the DFT grid                         |
| $n_\mathrm{awf}$                                                                    | number of PAOs in the cell (`nawf` in the code)            |
| $E_F$                                                                               | Fermi energy; plots use $E - E_F$ unless stated            |
| $\mathbf{v}_{n\mathbf{k}} = \hbar^{-1}\nabla_{\mathbf{k}}\varepsilon_{n\mathbf{k}}$ | band velocity                                              |

## When a request conflicts with these rules

Several developers use this skill, and a rule is sometimes broken on purpose. A rule is never broken silently, either by you or at someone's request.

1. If the developer asks for something that breaks a rule in this skill, or the task cannot be done without breaking one, stop before writing that part. Examples: a catchy title, an em dash, a tutorial on data inside the repository, a number we did not run, a figure in a non-palette color. Name the rule (its section here, and its check key below if it has one), say in one sentence why it exists, and offer the version that follows it.
2. Proceed with the exception only after the developer explicitly says to override that rule in their reply. Silence, an earlier override on another page or in an earlier session, or a general "do whatever" does not count.
3. Mention every override in the final report.

The same applies to a page that already breaks a rule when you are asked to revise it. Point out the violations the checks report, and fix them unless the page has an override line for that rule. If the developer wants to keep one, they give the override and you record it.

Check keys: `banned-phrase`, `history`, `em-dash`, `bold`, `greek`, `voice`, `todo`, `repo-path`, `color`, `hide-input`, `no-output`, `registration`, `base-example`. Rules without a key (jargon, titles, physics depth, numbers from the run) are checked by reading. Their overrides use the section name as the key, for example `% override: titles | ...`.

## Checks

Run from the repository root, giving the pages you wrote or revised:

````bash
/home/anooja/miniconda3/envs/paoflow/bin/python - docs/tutorials/tutorial01.ipynb docs/tutorials/howtos/sparse-workflow.md <<'EOF'
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
OVERRIDE = re.compile(r"^%\s*override:\s*([\w-]+)\s*\|(.*)$", re.M)

def blocks(path):
    """Yield (kind, text, tags, has_output) for markdown prose and code."""
    if path.suffix == ".ipynb":
        for c in json.loads(path.read_text())["cells"]:
            yield (c["cell_type"], "".join(c["source"]), c.get("metadata", {}).get("tags", []),
                   bool(c.get("outputs")))
    else:
        parts = re.split(r"^(```.*)$", path.read_text(), flags=re.M)
        code = False
        for p in parts:
            if p.startswith("```"):
                code = not code
                continue
            yield ("code" if code else "markdown", p, [], True)

def prose(text):
    """Markdown with comments, inline code, math and link targets removed."""
    text = re.sub(r"^%.*$", "", text, flags=re.M)
    text = re.sub(r"\$\$.*?\$\$", "", text, flags=re.S)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return re.sub(r"`[^`]*`|\$[^$]*\$", "X", text)  # placeholder keeps **`flag`** intact

problems, overridden = [], []
for name in sys.argv[1:]:
    path = Path(name)
    found = []
    say = lambda key, msg: found.append((key, msg))
    allmd = ""
    for kind, text, tags, has_out in blocks(path):
        if kind == "markdown":
            allmd += text + "\n"
            p = prose(text)
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
            if re.search(r"\b(you|your)\b|\bone can\b", low):
                say("voice", "'you'/'one can' in prose; use 'we' (imperative is fine in admonitions)")
            if "{todo" in low:
                say("todo", "unfilled {TODO: ...} placeholder")
        else:
            if re.search(r"(savedir|outputdir|open|loadtxt)\s*[=(][^\n]*['\"](/home|/Users|\.\./)", text):
                say("repo-path", "code points into the repository or an absolute path; use the downloaded dataset")
            for hexc in re.findall(r"#[0-9A-Fa-f]{6}\b", text):
                if hexc.upper() not in PALETTE:
                    say("color", f"{hexc} is not in the palette")
            if re.search(r"color\s*=\s*['\"](?!#)\w+['\"]|['\"]C\d['\"]", text):
                say("color", "named or cycle color in plot code; use palette hex codes")
            if path.suffix == ".ipynb":
                if re.search(r"plt\.|matplotlib", text) and "hide-input" not in tags:
                    say("hide-input", "plotting cell without the 'hide-input' tag")
                if text.strip() and not has_out and not text.lstrip().startswith("#"):
                    say("no-output", f"code cell without stored output: {text.strip().splitlines()[0][:60]!r}")
    if path.parent.name == "howtos":
        idx = (path.parent / "index.md").read_text()
        if path.stem not in re.findall(r":link:\s*(\S+)", idx) or not re.search(rf"^{path.stem}$", idx, re.M):
            say("registration", "missing grid card or toctree entry in docs/tutorials/howtos/index.md")
        m = re.search(r"Base example:.*?(examples/[\w./-]+)", allmd)
        if not m:
            say("base-example", "no 'Base example: examples/...' line")
        elif not Path(m.group(1)).exists():
            say("base-example", f"{m.group(1)} does not exist")
    elif path.name.startswith("tutorial"):
        if path.stem not in re.findall(r":link:\s*(\S+)", (path.parent / "index.md").read_text()):
            say("registration", "missing grid card in docs/tutorials/index.md")
    waived = {k: why.strip() for k, why in OVERRIDE.findall(allmd)}
    problems += [f"{name}: [{k}] {msg}" for k, msg in found if k not in waived]
    overridden += [f"{name}: [{k}] overridden ({why})" for k, why in waived.items()]
print("\n".join(overridden + problems) or "no problems found")
sys.exit(1 if problems else 0)
EOF
````

Some hits are false positives. A cell that only defines paths has no output, a quoted log line may contain "you", and "no longer" can be physics ("no longer projectable"). Do not write override lines for these: leave them and mention them in the report. Override lines are only for rules the developer chose to break.
