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

Both routes feed `unfold_bands` with a model dict: `EDTBModel.to_model_dict()` for a tight-binding model and `paoflow_to_model_dict(data_controller)` for a projected DFT run (after `pao_hamiltonian()`) or any other PAOFLOW run. **Convention (15 November 2026).** Every PAOFLOW Hamiltonian stores the hopping to the orbital at `tau_B + R` at the `HRs` index `-R`, so that `H(k) = sum_n exp(-2 pi i k.n) HRs[n]`: the projected Hamiltonian (`HRs = ifftn(Hks)`), the hopping-list and PythTB models, and `Slater_Koster`, `SK_EDTB` and the dual-parameter builders, which stored it at `+R` until 15 November 2026. The unfolding transforms with `exp(+2 pi i k.R)`, so `_extract_hamiltonian` orients the grid: `build_model_hamiltonian` returns `R = -R_n` for a model dict, and a `_paoflow` dict is flipped by its `fourier_sign` (`-1` from `paoflow_to_model_dict`) or, without it, by `_detect_fourier_sign` (the bond-length test of `sparse_hamiltonian.detect_bond_sign`). Band energies are blind to the sign (with time reversal `H(k)` and `H(-k)` have the same spectrum), spectral weights are not: with the sign wrong the weights select the folded branches of the mirror point `-k`, with the sum rule still satisfied. This is the mismatch that appeared between a DFT and a Slater-Koster unfolding: with the old `+R` storage the same SK supercell gave weights that differed by up to 1 through `paoflow_to_model_dict` and through the model runner; they now agree (`tests/unit/models/test_builder_conventions.py`), and the unfolding through the runner is bit-identical to before. Since the same date `PAOFLOW.bands()` and the path routines of `topology/` evaluate `H(k)` with `exp(-2 pi i k.R)` (they evaluated `H(-k)` before: the same energies with time reversal, mirrored paths without it).

## The EDTB Idea

Traditional two-center Slater-Koster models assume that the hopping integral between two atoms depends only on their separation and orbital character. The EDTB formalism relaxes this by introducing a screening factor that accounts for the atoms in between — the mediating environment. This makes it possible to build models that stay accurate across configurations where the local environment changes, without needing a separate parameterization for each one.
