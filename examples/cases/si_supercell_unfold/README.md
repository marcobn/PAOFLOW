# Si: band unfolding of the 8-atom cubic cell onto the fcc zone

An environment-dependent Slater-Koster (EDTB) model of silicon with s, p and d orbitals on the
8-atom conventional cubic cell: `PAOFLOW.models.band_unfold.unfold_bands` unfolds its bands onto the
Brillouin zone of the 2-atom fcc cell along L-Γ-X, giving each supercell state its spectral weight,
for the perfect cell and for a cell with a +1 eV impurity potential on one atom. No DFT run: the
models are built in code from the parameters in `Si_SK_params.json`.

## Run it

- **PAOFLOW Studio**: *Examples…* → `si_supercell_unfold`. Studio copies the case, lists its scripts and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example si_supercell_unfold` copies the case into `./si_supercell_unfold`; run the steps below in that folder.
- **Python**: `main.py` is the workflow; read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Unfold the bands of the perfect and the perturbed 8-atom cell onto the fcc zone | `python main.py > main.log` | under a minute on 4 cores |

## Outputs

- `main.log`: the supercell matrix, the atom mapping and the sum rule of the weights.
- `si_unfolded.npz`: the path (`kdist`, `kpath`, `ticks`), the primitive bands `E_pc`, and the supercell bands and weights of the perfect cell (`E_sc`, `W`) and of the cell with the impurity (`E_imp`, `W_imp`), in eV.

The parameter file is also a unit-test fixture (`tests/data/siliconSC/`).

Explained in: [How to unfold supercell bands onto the primitive Brillouin zone](https://paoflow.readthedocs.io/en/latest/learn/howtos/band-unfolding.html)
