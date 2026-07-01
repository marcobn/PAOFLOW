---
name: feature-design
description: "Design and refactor PAOFLOW features and subsystems. Use when: adding new compute modules, redesigning transport/, introducing new drivers, enforcing argument-driven APIs, deciding what belongs in DataController, and preserving behavior during refactors."
user-invocable: true
---

# PAOFLOW Feature Design

## Scope

This skill defines design rules for:

- General PAOFLOW feature additions.
- Example and driver design where `main.py` should read like a physics workflow.

The goal is consistency, maintainability, behavior preservation, and user-facing workflows that are understandable when read from top to bottom.

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

7. **Design the public workflow before designing internals.**
   Start from the desired top-down `main.py` script. Then design the orchestrator methods and backend modules needed to make that script clean, physical, and maintainable.

---

## Architecture Pattern

Use a dual-driver orchestrator model:

- `src/PAOFLOW/PAOFLOW.py`: Existing orchestrator for core workflows.
- `src/PAOFLOW/Transport.py`: New orchestrator dedicated to Green's-function transport workflows.

Both orchestrators should call module-level procedural functions in subpackages.

The orchestrators are the user-facing physics layer. Backend modules are implementation details.

---

## Physics-Readable Workflow Design

### Design Goal

A PAOFLOW `main.py` should read like the physics workflow a computational materials scientist has in mind, not like a list of internal implementation details.

When read top-down, the script should clearly show:

1. What input electronic-structure data is being loaded.
2. How the PAO Hamiltonian is constructed.
3. What physical transformation or preparation is applied.
4. What observable is computed.
5. What results are written or plotted.

The user should not need to understand internal containers, low-level array keys, or implementation modules to follow the workflow.

### Main Script Rule

Design `main.py` as a sequence of high-level physics actions.

Preferred style:

```python
from PAOFLOW import PAOFLOW
from PAOFLOW.Transport import Transport

pao = PAOFLOW(savedir="al.save")

pao.read_atomic_proj_QE()
pao.projectability()
pao.pao_hamiltonian()
pao.doubling_Hamiltonian(nx=1, ny=1, nz=1)

transport = Transport(pao.data_controller)

transport.conductor(
    emin=-2.0,
    emax=2.0,
    ne=500,
    delta=1e-5,
    nk=[12, 12],
    formula="landauer",
)

transport.current(
    bias_min=-1.0,
    bias_max=1.0,
    nbias=101,
    temperature=300.0,
)
```

Avoid exposing implementation details in `main.py`:

```python
arrays, attr = pao.data_controller.data_dicts()
run_conductor_pipeline(arrays, attr, internal_flags, temporary_buffers)
```

Low-level access is acceptable only in tests, debugging scripts, or specialized developer workflows.

### Method Naming Rule

User-facing orchestrator methods should be named after physics actions or observables, not implementation steps.

Prefer:

- `read_atomic_proj_QE()`
- `projectability()`
- `pao_hamiltonian()`
- `doubling_Hamiltonian()`
- `bands()`
- `dos()`
- `fermi_surface()`
- `berry_curvature()`
- `conductor()`
- `current()`

Avoid names that expose plumbing:

- `prepare_arrays_for_step_3()`
- `run_pipeline_internal()`
- `populate_data_dict()`
- `compute_tmp_blocks()`
- `call_backend_runner()`

### Argument Naming Rule

Arguments exposed in `main.py` should be physical parameters or user-relevant numerical controls.

Prefer:

- `emin`, `emax`, `ne`, `delta`
- `energy_grid`
- `k_path`, `nk`
- `temperature`
- `bias_min`, `bias_max`, `nbias`
- `lead_coupling`, `self_energy`, `chemical_potential`
- `spin_orbit`, `smearing`, `adaptive_smearing`

Avoid exposing:

- raw `arrays`, `attr`, `data_dict`
- internal flags whose meaning is not physical
- temporary matrix buffers
- module-private lookup tables
- implementation-specific object graphs

### One Public Step, Many Backend Functions

One clean public method may call many backend functions internally.

Example:

```python
transport.conductor(...)
```

may internally perform:

- energy-grid construction
- k-point setup
- Hamiltonian block assembly
- lead self-energy construction
- Green's-function solution
- transmission accumulation
- output writing

Those internal stages should be split into small callable backend functions, but `main.py` should remain a readable physics workflow.

### Do Not Push Backend Complexity Upward

If a public method becomes difficult to implement, do not solve that by forcing `main.py` to call more internal steps.

Bad direction:

```python
transport.prepare_energy_grid(...)
transport.prepare_lead_blocks(...)
transport.prepare_self_energies(...)
transport.prepare_green_solver(...)
transport.compute_transmission(...)
```

Better direction:

```python
transport.conductor(...)
```

with the internal preparation split into backend modules.

Expose lower-level public methods only when they represent meaningful reusable physics operations.

### Public API Review Test

Before accepting a new feature or refactor, inspect the final `main.py` and ask:

- Can a physicist understand the workflow by reading it top-down?
- Are the visible method names physics actions or observables?
- Are the visible arguments physical parameters or user-facing numerical controls?
- Are internal arrays, scratch buffers, and backend modules hidden from the example script?
- Could this example become documentation without explaining PAOFLOW internals first?

If the answer is no, redesign the public API before refining backend code.

---

## General Codebase Design (All New Features)

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

Backend functions may access internal arrays through the orchestrator, but examples and user-facing scripts should not expose those internals unless explicitly debugging.

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

6. Does the example `main.py` read like a physics workflow?
   If no, redesign the public API before changing more backend code.

7. Is `main.py` exposing internal arrays or temporary objects?
   If yes, move that logic behind an orchestrator method or backend pipeline.
