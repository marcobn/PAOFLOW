# Recipes

Scripts for what the generator (`paoflow-gen`, the Setup tab of `paoflow-studio`) does not write. Each folder
has a README: what the script shows, which inputs it needs, how to run it.

| Recipe | What it shows |
|---|---|
| `vasp/` | PAOFLOW on VASP runs (vasprun.xml, POTCAR projections): Si, Pt, Fe, MnF2, Mn3Ir, FeRh, CrI3 |
| `basis_presets/` | the `minimal`, `standard` and `extended` internal bases on NC and ultrasoft, scalar and fully relativistic pseudopotentials (Si, Pt) |
| `sparse_backend/` | `SparsePAOFLOW` on a doubled cell: the bond-list backend for large cells |
| `transport_defect/` | the low-level `PAOFLOW.Transport` calls of a conductor with a defect |

Studio-ready examples (inputs, answers and README) are in `../cases/`.
