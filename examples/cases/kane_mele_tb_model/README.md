# Kane-Mele model: bands and spin Hall conductivity

The Kane-Mele model of the quantum spin Hall effect (graphene with intrinsic spin-orbit coupling,
`t = 1` eV, `soc_par = 0.1` eV), built in code with `PAOFLOW(model=...)`: no DFT run. PAOFLOW
computes its bands, its density of states, and the spin Hall conductivity of the sheet with the
Kubo formula, together with the map of the spin Berry curvature of the occupied bands.

## Run it

- **PAOFLOW Studio**: *Examples…* → `kane_mele_tb_model`. Studio copies the case, lists its
  scripts and runs the steps below in the Runs tab.
- **Command line**: `paoflow-gen --example kane_mele_tb_model` copies the case into
  `./kane_mele_tb_model`; run the steps below in that folder.
- **Python**: `main.py` is the workflow; read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Bands, DOS and spin Hall conductivity of the Kane-Mele model | `python main.py` | 11 s on 1 core |

## Outputs

- `kane_mele/bands_0.dat`, `kane_mele/kpath_points.txt`: the bands along Γ-M-K-Γ and the
  positions of the labels.
- `kane_mele/dosdk_0.dat`: the DOS, energy (eV) and states per eV.
- `kane_mele/shcEf_z_xy.dat`: the spin Hall conductivity σ^z_xy of the sheet against the Fermi
  energy, in units with ħ = e = 1, for the last broadening of `main.py` (0.05 eV).
- `kane_mele/Spin_Berry_z_xy.bxsf`: the spin Berry curvature of the occupied bands on the k-mesh
  (bohr²).

Explained in: [Spin Hall conductivity of the Kane–Mele model](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial05.html),
[How to run PAOFLOW on a tight-binding model](https://paoflow.readthedocs.io/en/latest/learn/howtos/tight-binding-model.html)
