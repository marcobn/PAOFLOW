# Slater-Koster and EDTB Module

The Slater-Koster with environmental dependence (EDTB) module gives you tight-binding model construction and fitting tools that go beyond the standard two-center approximation. By including environment-dependent screening corrections, it can capture configuration-dependent effects that matter in surfaces, defects, and strained structures.

## Module Layout

**`PAOFLOW.defs.models`** — Model builders:

- `SK_EDTB`: The environment-dependent extension. Modulates hopping integrals through a scalar screening factor that depends on the local atomic environment.
- `Slater_Koster`: Builds generalized two-center tight-binding models up to third nearest neighbors.
- Pre-built models for common systems (graphene, cubium variants) to get you started quickly.

**`PAOFLOW.defs.sk_fitting`** — Parameter fitting:

- `SKFitter`: Single-geometry, eigenvalue-based fitting.
- `SKFitterEDTB`: Extends fitting to include the screening parameters (γ values).
- `MultiGeomEDTB`: Fits shared parameters across multiple atomic configurations simultaneously — useful when you want a model that transfers across different geometries.

**`PAOFLOW.defs.edtb_params`** — Model serialization via the `EDTBModel` class. Models serialize to JSON, so they're easy to store, share, and load across different simulation setups.

**`PAOFLOW.defs.surface_project`** — Projects bulk band structures onto surface planes, identifying absolute gaps and lens-shaped features.

**`PAOFLOW.defs.dual_params`** — Dual-parameter models that give bulk and surface atoms independent parameter sets, which is useful when the bonding at the surface differs significantly from the bulk.

**`PAOFLOW.defs.band_unfold`** — Supercell-to-primitive-cell band unfolding with spectral weight analysis.

## The EDTB Idea

Traditional two-center Slater-Koster models assume that the hopping integral between two atoms depends only on their separation and orbital character. The EDTB formalism relaxes this by introducing a screening factor that accounts for the atoms in between — the mediating environment. This makes it possible to build models that stay accurate across configurations where the local environment changes, without needing a separate parameterization for each one.
