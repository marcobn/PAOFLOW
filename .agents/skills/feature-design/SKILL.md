---
name: feature-design
description: "Design and refactor PAOFLOW features and subsystems. Use when: adding new compute modules, redesigning transport/, introducing new drivers, enforcing argument-driven APIs, deciding what belongs in DataController, and preserving behavior during refactors."
user-invocable: true
---

# PAOFLOW Feature Design

## Scope

This skill defines design rules for:

- General PAOFLOW feature additions.
- Transport redesign work under `src/PAOFLOW/transport/`.

The goal is consistency, maintainability, and behavior preservation.

---

## Core Rules (Applies Everywhere)

1. **Do not rename existing files.**
   Existing legacy modules (including `do_*`) stay unchanged unless explicitly requested.

2. **New module names must NOT use `do_*`.**
   Use descriptive names such as `bands_solver.py`, `transport_core.py`, `kspace_ops.py`, `current_pipeline.py`.

3. **No input-parameter files for runtime configuration.**
   Do not design new workflows around YAML/JSON/INI config files that are read and injected into code.
   Inputs are passed as direct Python arguments from `main.py` or user code.

4. **DataController is not a dumping ground.**
   Store data in `DataController` only if it is reused by later steps or needed for interoperability.
   Keep temporary and one-off intermediates local to function scope.

5. **No logic changes during redesign/refactor.**
   Reorganization is allowed; algorithmic behavior and numerical outputs must remain equivalent.

6. **All functions and variables must be self documenting.**
   Use descriptive function, parameter, local variable, and return-value names that reveal their role in the physics or workflow. Avoid abbreviations and one-letter names unless they are standard domain notation and the surrounding code/docstring makes their meaning clear.
   Prefer names such as `energy_grid`, `overlap_matrix`, `hamiltonian_kspace`, and `selected_band_indices` over opaque names such as `arr`, `tmp`, `x`, or `vals`.

---

## Architecture Pattern

Use a dual-driver orchestrator model:

- `src/PAOFLOW/PAOFLOW.py`: Existing orchestrator for core workflows.
- `src/PAOFLOW/Transport.py`: New orchestrator dedicated to green's function transport workflows.

Both orchestrators should call module-level procedural functions in subpackages.

---

## Part A: General Codebase Design (All New Features)

### Placement Rules

| Category                   | Location                                         | New-module naming style                     |
| -------------------------- | ------------------------------------------------ | ------------------------------------------- |
| Spectrum features          | `src/PAOFLOW/spectrum/`                          | `*_solver.py`, `*_pipeline.py`              |
| Hamiltonian features       | `src/PAOFLOW/hamiltonian/`                       | `*_ops.py`, `*_builder.py`                  |
| Topology/response features | `src/PAOFLOW/topology/`, `src/PAOFLOW/response/` | descriptive domain names                    |
| Transport redesign         | `src/PAOFLOW/transport/`                         | `*_core.py`, `*_pipeline.py`, `*_runner.py` |
| New driver                 | `src/PAOFLOW/Transport.py`                       | class `Transport`                           |

### Function Style for New Modules

Use procedural functions with explicit arguments, type hints, and self-documenting names.

```python
# src/PAOFLOW/spectrum/bands_solver.py
from numpy.typing import NDArray

def compute_bands(
   hamiltonian_kspace: NDArray,
   k_path: NDArray,
   number_of_kpoints: int,
   spin_orbit: bool = False,
) -> tuple[NDArray, dict[str, object]]:
    """Compute band energies along a k-path."""
    # local temporary arrays only
    # return values explicitly
    return energies, meta
```

### Orchestrator Integration

```python
# src/PAOFLOW/PAOFLOW.py

def bands(self, ibrav=None, band_path=None, nk=500, spin_orbit=False):
    from .spectrum.bands_solver import compute_bands

    arrays, attr = self.data_controller.data_dicts()
    energies, meta = compute_bands(arrays['Hksp'], band_path, nk, spin_orbit=spin_orbit)

    # Store only if reused downstream
    arrays['E_k'] = energies
```

### DataController Storage Policy

Store only if at least one is true:

- Needed by another public method later in the workflow.
- Needed for restart/checkpoint behavior.
- Needed for writer/exporter APIs.
- Needed across MPI stages where recomputation is expensive.

Do not store:

- Scratch tensors used once inside one method.
- Debug-only arrays unless explicitly requested.
- Duplicate views/copies of already stored data.

### Input Policy

Preferred usage:

```python
# examples/.../main.py
transport = Transport(pao.data_controller)
transport.conductor(
    emin=-2.0,
    emax=2.0,
    ne=500,
    delta=1e-5,
    nk=[12, 12],
    formula='landauer',
)
```

Avoid new interfaces like:

- `from_yaml(...)`
- `load_config_file(...)`
- `input_parameters.py`-only pipelines for new features

---

## Part B: Transport Redesign Rules (Specific to `transport/`)

### Design Goal

Refactor green's function transport from class-heavy internals to procedural modules while keeping behavior unchanged.

### Mandatory Constraints

1. Keep existing public imports working during migration.
   Example: existing `ConductorRunner`/`CurrentRunner` can remain as compatibility wrappers.

2. New transport compute modules must not use `do_*` names.
   Examples:

- `conductor_pipeline.py`
- `current_pipeline.py`
- `green_solver.py`
- `self_energy_ops.py`

3. New primary API uses direct arguments, not file-based config.

4. Keep local intermediates local; only persist reusable results.

### New Driver Structure

`src/PAOFLOW/Transport.py` should provide user-facing methods such as:

```python
class Transport:
    def __init__(self, data_controller):
        self.data_controller = data_controller

    def conductor(self, *, emin, emax, ne, delta, nk, formula='landauer', **kwargs):
        from .transport.conductor_pipeline import run_conductor
        return run_conductor(self.data_controller, emin=emin, emax=emax, ne=ne, delta=delta, nk=nk, formula=formula, **kwargs)

    def current(self, *, bias_min, bias_max, nbias, temperature, **kwargs):
        from .transport.current_pipeline import run_current
        return run_current(self.data_controller, bias_min=bias_min, bias_max=bias_max, nbias=nbias, temperature=temperature, **kwargs)
```

### Migration Strategy

1. Extract logic from class methods into module-level functions.
2. Keep thin wrapper classes for compatibility.
3. Add direct-argument APIs in `Transport` class.
4. Port examples to new direct-argument API.
5. Only then deprecate file-based pathways (if any remain).

---

## Part C: Behavior-Safety Checklist

Before merging any redesign patch:

- [ ] No algorithmic edits disguised as refactor.
- [ ] Numerical outputs match baseline within expected tolerance.
- [ ] Legacy entry points still execute (if not intentionally removed).
- [ ] New APIs accept arguments directly (no required config file).
- [ ] New modules avoid `do_*` naming.
- [ ] DataController additions are justified by reuse.
- [ ] Temporary arrays are not persisted.
- [ ] Tests updated for both compatibility path and new path.

---

## Quick Decision Guide

1. Is this a new module?
   Use descriptive filename, not `do_*`.

2. Is this input configuration?
   Pass function arguments from user code/main, not config files.

3. Should this result go into DataController?
   Only if reused later or required by shared workflow contracts.

4. Is this transport redesign?
   Route through `Transport.py` orchestrator with procedural backend modules.

5. Are outputs changing?
   If yes, stop and split into separate feature change; redesign patches must preserve logic.
