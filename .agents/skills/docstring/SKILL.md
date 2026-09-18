---
name: docstring
description: "Write and update PAOFLOW Python docstrings. Use when: adding docstrings, revising API documentation, documenting equations, explaining algorithms or physics, and aligning function signatures with type hints."
user-invocable: true
---

# PAOFLOW Docstring Style

## Scope

Use this skill whenever adding or updating Python docstrings in PAOFLOW.

Docstrings should follow the style: concise summary, NumPy-style sections, explicit array shapes and dtypes when useful, and a `Notes` section for equations, physics, or algorithmic context.

---

## Core Rules

1. **All functions must be type hinted.**
   Add type hints to every new or edited function signature, including return types. Use standard Python typing and local project types where available.

2. **Use NumPy docstring convention.**
   Prefer these sections, in this order when applicable:
   - One-line summary.
   - Extended summary, only if needed.
   - `Parameters`
   - `Returns`
   - `Raises`
   - `Notes`
   - `Examples`

3. **Do not rewrite existing docstrings unless behavior changed.**
   Preserve existing docstring content during refactors and mechanical edits. Update a docstring only when the function signature, logic, accepted inputs, returned values, side effects, physics, or numerical algorithm has changed.

4. **Document side effects clearly.**
   Functions that mutate `DataController`, arrays, attributes, files, MPI state, or process control should say so in `Returns`.

5. **Keep names and prose self documenting.**
   Prefer clear parameter and variable names so the docstring explains concepts, not avoidable ambiguity. Avoid one-letter names unless they are standard mathematical notation and the surrounding context makes them clear.

---

## Function Signatures

Use explicit type hints in all new or modified functions.

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_bands(
    hksp: NDArray[np.complex128],
    kpath: NDArray[np.float64],
    nk: int,
    spin_orbit: bool = False,
) -> tuple[NDArray[np.float64], dict[str, object]]:
    """Compute band energies along a k-path."""
```

Use `NDArray[np.float64]`, `NDArray[np.complex128]`, or a broader `NDArray[np.floating]`/`NDArray[np.complexfloating]` when the exact dtype is not guaranteed. If a function accepts a `DataController` and importing the class would cause a circular import, use a string annotation or `TYPE_CHECKING` import.

---

## NumPy Docstring Format

Write parameters as:

```python
def build_hamiltonian(
    data_controller: "DataController",
    shift: float,
) -> np.ndarray:
    """Construct the PAO Hamiltonian from projected DFT eigenstates.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``U`` (shape ``(nawf, nbnds, nkpnts, nspin)``),
        ``my_eigsmat`` (shape ``(nbnds, nkpnts, nspin)``).
    shift : float
        Energy shift used to separate the low-energy PAO subspace from the
        shifted complement.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf, nkpnts, nspin)``, complex
        The PAO Hamiltonian in k-space.
    """
```

Use double backticks for code names, array keys, attributes, filenames, and literal values inside docstrings.

For functions that return `None` but mutate shared state, document the stored arrays or attributes:

```python
Returns
-------
None
    Adds ``HRs`` to ``data_controller.data_arrays`` and broadcasts it to all
    MPI ranks.
```

---

## Equations For Read The Docs

Write equations in reStructuredText math so Sphinx and Read the Docs render them correctly.

Use display equations with `.. math::`, followed by a blank line and an indented LaTeX block:

```python
"""Construct the PAO Hamiltonian.

Notes
-----
The PAO Hamiltonian is built at each k-point as

.. math::

    H(\\mathbf{k}) = A_c \\varepsilon_c A_c^\\dagger
        + \\eta \\left(I - A_c (A_c^\\dagger A_c)^{-1} A_c^\\dagger\\right).
"""
```

Use inline math with `:math:` for short expressions, such as ``:math:`H(\\mathbf{k})``` or ``:math:`S^{1/2} H S^{1/2}```.

Equation rules:

- Put display equations in the `Notes` section unless the equation is the main return value definition.
- Leave one blank line before and after `.. math::` blocks.
- Indent every equation line by four spaces.
- Escape backslashes in Python docstrings, for example `\\mathbf{k}`, unless the docstring is a raw string.
- Prefer `\\left(` and `\\right)` for large grouped terms.
- Use `\\dagger`, `\\varepsilon`, `\\eta`, and similar LaTeX commands instead of Unicode symbols.
- Do not place unindented prose inside the math block.

---

## Notes Sections

Add a `Notes` section when it helps explain the physics, numerical method, assumptions, or important implementation detail in plain language.

Good `Notes` content includes:

- The physical meaning of the computed quantity.
- The algorithmic sequence used by the function.
- Why a numerical safeguard exists, such as enforcing Hermiticity or avoiding division by zero.
- How flags select different published formulas or computational branches.
- MPI behavior, symmetry expansion, basis transformations, or file-writing side effects.

Keep the explanation easy to read. A future contributor should understand what the code is doing before needing to parse every matrix operation.

Example style:

```python
Notes
-----
Only MPI rank 0 performs the inverse FFT.  The result is broadcast via
``DataController.broadcast_single_array`` so that all ranks carry an identical
copy of ``HRs``.
```

---

## Preservation Checklist

Before finishing docstring work, check that:

- [ ] Every new or edited function has type hints and a return annotation.
- [ ] Docstrings follow NumPy section headings and underline style.
- [ ] Existing docstring content was preserved unless behavior changed.
- [ ] Array shapes, dtypes, mutated keys, and side effects are documented when relevant.
- [ ] Equations use Read the Docs-compatible reStructuredText math.
- [ ] `Notes` explain the physics or algorithm when that would help future readers.
