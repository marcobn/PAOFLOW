# Si: band unfolding of the 8-atom cubic cell onto the fcc zone

An environment-dependent Slater-Koster (EDTB) model of bulk silicon with s, p and d orbitals
(`Si_SK_params.json`, nine orbitals per atom), built on the 2-atom fcc cell (`Si_geometry.json`) and on
the 8-atom conventional cubic cell (`Si_geometry_sc8.json`). `PAOFLOW.models.band_unfold.unfold_bands`
finds the supercell matrix, the translations and the atom mapping by itself and unfolds the 72 bands of
the cubic cell onto the fcc Brillouin zone: every supercell state gets a spectral weight $W_n(\mathbf k)$
on the path of the primitive cell. No DFT run.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name. Studio copies
  the files and lists the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_supercell_unfold` copies the case into `./si_supercell_unfold` and prints
  the steps; run them in that folder.

## Steps

1. `python silicon.py`: builds both models, unfolds along Γ-X-W-K-Γ-L-U-W-L-K|U-X (80 points per
   segment) and writes `si_sc8_unfolded.png`.

Runtime: a few seconds on one core.

## What to look at

- The sum rule printed at the end: $\sum_n W_n(\mathbf k) = 18$, the orbitals of the primitive cell, at
  every k (to about 1e-14).
- `si_sc8_unfolded.png`: the unfolded supercell states, coloured by weight, fall on the bands of the
  primitive cell (the grey reference lines underneath): dark red for weight 1, lighter where states
  share the weight. The folded copies, with weight below 0.02, are not drawn.

## Notes

The same unfolding applies to projected DFT Hamiltonians: build the model dicts with
`PAOFLOW.models.band_unfold.paoflow_to_model_dict(paoflow.data_controller)` for a primitive-cell run and a
supercell run (both after `pao_hamiltonian()`, same pseudopotential and basis). The helper records the
Fourier convention of the projected Hamiltonian (`fourier_sign = -1`), to which the spectral weights are
sensitive. Since 15 November 2026 the Slater-Koster and EDTB builders store their hoppings in the same
convention as DFT, so models and DFT runs mix freely (`docs/internals/sk-edtb.md`).

The parameter and geometry files are also a unit-test fixture (`tests/data/siliconSC/`).

Former location: `examples/unfold_examples/siliconSC`.
