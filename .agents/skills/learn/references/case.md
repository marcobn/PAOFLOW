# Run card template

A run card is the `README.md` of a case, `examples/cases/<case>/README.md`. It answers *what to run*: what the case computes, the ways to run it, its steps with their run time, and its outputs. The physics goes to a theory page, parameter tables to the docstrings of the methods, and the choice of settings to a how-to; the run card links to them. GitHub shows it when a reader opens the case folder, and Studio's *Examples…* shows the `title` and `summary` of `case.json`, so the first paragraph agrees with `summary`.

## Structure

````markdown
# <title of case.json>

<one paragraph: the system, what PAOFLOW computes on it, and the method in one clause>

## Run it

- **PAOFLOW Studio**: *Examples…* → `<case>`. Studio copies the case, <fills the <workflow> form | lists its scripts> and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example <case>` copies the case into `./<case>` <and writes the drivers>; run the steps below in that folder.
- **Python**: <generator case: the `main.py` written by the command above is the workflow, a plain PAOFLOW script | scripts case: `main.py` is the workflow>; read it to change a call.

<case with data: To skip the QE runs, copy the case with `paoflow-gen --example <case> --with-data` (in Studio, *Use the precomputed data of the case*): it extracts <what> from `<case>-v<N>.tar.gz` of the release [case-data](https://github.com/marcobn/PAOFLOW/releases/tag/case-data).>

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| <label of case.json> | `pw.x -in scf.in` | <measured, with the core count> |
| ... | `mpirun -np 4 python main.py` | ... |

## Outputs

- `output/<file>`: <what it holds, its columns>

Explained in: [<page title>](https://paoflow.readthedocs.io/en/latest/<page path>.html)
````

## Rules

- The steps and their labels are those of `case.json` (`qe`, then the drivers, then `steps`), in that order. A run time is measured, with the core count; write "a few seconds" only below ten.
- The run card covers the case's own calculation. The inputs, scripts, steps and outputs of a page's variation are on that page, beside it in `docs/learn/`, and the run card links the page under "Explained in".
- Name the inputs only where the reader must act on them (a file to extract, a pseudopotential they must bring, a precomputed archive).
- "Explained in" lists every page of the case's `docs`, by title; a case with no page leaves the line out. Links go to `latest` until the docs-URL helper picks the version of the installed PAOFLOW.
- No equations, no parameter tables, no advice on choosing values: link the theory page or the how-to that has them.
