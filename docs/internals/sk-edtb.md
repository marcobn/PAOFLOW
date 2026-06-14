# Slater-Koster and EDTB Module

The Slater-Koster with environmental dependence (EDTB) module provides tight-binding model construction and fitting tools that extend the standard two-center approximation with environment-dependent screening corrections.

## Module Layout

**`PAOFLOW.defs.models`** — Model builders:

- `SK_EDTB`: Extends two-center Slater-Koster with environment-dependent screening corrections via a scalar modulation factor that depends on the local atomic environment.
- `Slater_Koster`: Constructs generalized two-center tight-binding models up to third nearest neighbors.
- Pre-built models for common systems (graphene, cubium variants).

**`PAOFLOW.defs.sk_fitting`** — Parameter fitting:

- `SKFitter`: Single-geometry, eigenvalue-based fitter.
- `SKFitterEDTB`: Extends fitting to include screening parameters (γ values).
- `MultiGeomEDTB`: Fits shared parameters across multiple atomic configurations simultaneously.

**`PAOFLOW.defs.edtb_params`** — Model serialization via the `EDTBModel` class, enabling JSON-based storage and transferability across different geometries.

**`PAOFLOW.defs.surface_project`** — Projects bulk band structures onto surface planes, identifying absolute gaps and lens-shaped features.

**`PAOFLOW.defs.dual_params`** — Dual-parameter models that distinguish bulk and surface atoms with independent parameter sets.

**`PAOFLOW.defs.band_unfold`** — Supercell-to-primitive-cell band unfolding with spectral weight analysis.

## Key Concept: EDTB Formalism

The EDTB approach modulates hopping integrals through a screening factor that accounts for mediating atoms in the local environment. This captures configuration-dependent effects beyond traditional two-center approximations, making it suitable for systems where the bonding environment varies significantly (surfaces, defects, strained structures).
