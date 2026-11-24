# Gapped graphene model: shift current and SHG

The honeycomb tight-binding model with a 1 eV sublattice gap (point group D3h): PAOFLOW computes
the shift current, the injection currents and the second-harmonic susceptibility of the sheet
with `PAOFLOW.nonlinear_optics`, checks the selection rules yyy = -yxx = -xxy and xxx = 0, and
repeats the run on the model without the gap, where inversion makes every response vanish. No DFT
run: the model is built in code.

## Run it

- **PAOFLOW Studio**: *Examples…* → `graphene_nlo_model`. Studio copies the case, lists its scripts and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example graphene_nlo_model` copies the case into `./graphene_nlo_model`; run the steps below in that folder.
- **Python**: `main.py` is the workflow; read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Shift current, injection currents and SHG of gapped graphene, with and without inversion | `python main.py > main.log` | under a minute on one core |

## Outputs

- `output/shift_<abc>.dat`, `output/shiftm_<abc>.dat`: shift current and magnetic shift current (photon energy in eV, nm µA/V²), for `abc` = `yyy`, `yxx`, `xxy`, `xxx`.
- `output/injectc_<abc>.dat`, `output/injectm_<abc>.dat`: circular and magnetic injection currents (nm A V⁻² s⁻¹).
- `output/shgr_<abc>.dat`, `output/shgi_<abc>.dat`, `output/shga_<abc>.dat`: real part, imaginary part and modulus of the SHG susceptibility (pm²/V).
- `output_inv/`: the same files for the centrosymmetric model, component `yyy` only.
- `main.log`: the symmetry checks printed by `main.py`.

The case `gaas_nlo` draws Fig. 1 of its validation report from `output/` when the two cases sit in
the same folder.

Explained in: [How to compute the non-linear optical response of a tight-binding model](https://paoflow.readthedocs.io/en/latest/learn/howtos/nonlinear-optics-model.html)
