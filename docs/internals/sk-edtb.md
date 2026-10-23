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

Both routes feed `unfold_bands` with a model dict: `EDTBModel.to_model_dict()` for a tight-binding model and `paoflow_to_model_dict(data_controller)` for a projected DFT run (after `pao_hamiltonian()`). The two store the real-space Hamiltonian in opposite Fourier conventions. Model Hamiltonians hold the hopping to `tau_B + R`, so `H(k) = sum_R H(R) exp(+2 pi i k.R)`; the projected Hamiltonian is `HRs = ifftn(Hks)`, so `H(k) = sum_R H(R) exp(-2 pi i k.R)` (hopping at `tau_B - R`). Band energies are blind to the difference (with time reversal the two give `H(k)` and `H(-k)`), spectral weights are not: with the sign wrong the weights select the folded branches of the mirror point `-k`, with the sum rule still satisfied. `unfold_bands` therefore reads the `fourier_sign` entry of a `_paoflow` dict (`-1`, set by `paoflow_to_model_dict`) and flips `R` before the transform; a dict without the entry is classified from the decay of the hoppings with bond length (`_detect_fourier_sign`, the same test as `sparse_hamiltonian.detect_bond_sign`). Note that `PAOFLOW.bands()` along a path uses `exp(+2 pi i k.R)` on the projected `HRs` and so evaluates `H(-k)`: identical energies for time-reversal-symmetric scalar systems, mirrored paths otherwise.

## The EDTB Idea

Traditional two-center Slater-Koster models assume that the hopping integral between two atoms depends only on their separation and orbital character. The EDTB formalism relaxes this by introducing a screening factor that accounts for the atoms in between — the mediating environment. This makes it possible to build models that stay accurate across configurations where the local environment changes, without needing a separate parameterization for each one.
