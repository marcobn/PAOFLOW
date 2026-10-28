# Agentic non-linear optics: SHG and photocurrents vs experiment

`PAOFLOW.agentic.nlo` compares the second-order responses of
[nonlinear_optics](nonlinear-optics.md) (shift and injection currents, SHG) with
experiments and published calculations. The data toolkit is the one of
[thermo-optics](thermo-optics.md). The agent playbook, lessons learned, prompts
and templates live in `agentic/nlo/`.

## Modules

| module | role |
|---|---|
| `model` | `NonlinearModel(output_dir, two_d, slab_height_nm, system)`: the `<kind>_<abc>.dat` files of a run (`shift`, `shiftm`, `injectc`, `injectm`, `shgr`, `shgi`, `shga`); `spectrum`, `shg` (complex), `value(kind, abc, energy)`, `tensor(kind, energy)`, `static`, `peak`, `onset`, `symmetry_check(point_group, kind, energy)` (forbidden components present, their leakage, the relations between allowed components, the symmetry error), `kleinman_check`, `d_coefficient` (d = χ/2), `effective_3d` (sheet value per layer thickness), `unit`, `describe` |
| `symmetry` | `GENERATORS` / `ALIASES` of the crystal classes with an axis convention (`'-43m'`/`'Td'`, `'3m_x'`/`'C3v'`, `'-6m2_x'`/`'D3h'`, `'mm2_x'`, `'4mm'`, `'6mm'`, ...), `point_group_operations(name)` (closure from the generators), `symmetrize(tensor, ops)`, `forbidden_components`, `independent_relations` (e.g. `yyy -> {yxx: -1, xxy: -1, xyx: -1}` for 3m), `symmetry_error`, `kleinman_error`, `tensor_from_components` |

## Design choices

- **The symmetry audit is the error bar.** A DFT-derived PAO Hamiltonian is symmetric only to
  about a meV; the second-order responses amplify this into percent-level leakage into forbidden
  components and spread between equivalent ones. The adapter quantifies both from the computed
  tensor and the crystal class, and the playbook quotes them with every comparison.
- **Conventions are data.** Experimental coefficients are stored with their convention (d or χ,
  component, photon energy, reference crystal, 3D or sheet with a thickness); the conversions are
  explicit methods, never implicit.
- **Point groups from generators**, not from tables of allowed components: any axis convention can
  be added by listing its generators, and the forbidden components and relations follow.

## Tests

`tests/unit/agentic/nlo/`: group closure (T_d 24, D3h 12, mm2 4 operations), forbidden components
and relations of T_d, 3m and mm2, symmetrisation and the Kleinman error, the adapter on a synthetic
output (spectra, tensors, static values, peaks, onsets, symmetry and Kleinman checks, unit
conversions, 2D sheet units, spin-channel files ignored).

Since October 28 2026 the point-group code lives in `src/PAOFLOW/utils/point_groups.py` (also used by the engine's k-grid reduction); `PAOFLOW.agentic.nlo.symmetry` re-exports it.
