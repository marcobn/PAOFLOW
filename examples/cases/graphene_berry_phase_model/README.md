# Gapped graphene model: Berry phases and Chern number

Spinless graphene with a sublattice potential (`{'label': 'graphene', 't': -1.0, 'delta': -0.2}`,
eV), built in code with `PAOFLOW(model=...)`: no DFT run. PAOFLOW computes its bands, the Berry
phases of its bands on circles around the valley K', from the overlaps of the Bloch states along
each loop, their dependence on the radius of the loop for the gapped and the gapless model, and
the Chern number of the lower band from the Berry phases of the plaquettes that tile the zone.

## Run it

- **PAOFLOW Studio**: *Examples…* → `graphene_berry_phase_model`. Studio copies the case, lists
  its scripts and runs the step below in the Runs tab.
- **Command line**: `paoflow-gen --example graphene_berry_phase_model` copies the case into
  `./graphene_berry_phase_model`; run the step below in that folder.
- **Python**: `main.py` is the workflow; read it to change a call.

## Steps

| step | command | run time |
| ---- | ------- | -------- |
| Bands, Berry phases and Chern number of gapped graphene | `python main.py` | 70 s on 1 core |

## Outputs

- `graphene/bands_0.dat`, `graphene/kpath_points.txt`: the bands along Γ-M-K-Γ and the positions
  of the labels.
- `graphene/circle_{lower,upper,both}.dat`: the Berry phase (last line) of the lower band, the
  upper band and both on the circle of radius 0.05 around K'.
- `graphene/patch_lower.dat` with `patch_lower_kgrid_corners.dat`: the Berry phase per plaquette
  (columns k_x, k_y, phase) of the lower band on a square around K', and the plaquette corners.
- `graphene/radius_<r>.dat`, `gapless/radius_<r>.dat`: the Berry phase of the lower band on
  circles of radius r, gapped and gapless graphene.
- `graphene/zone_lower.dat` with `zone_lower_kgrid_corners.dat`: the plaquette phases over the
  whole zone; their sum divided by 2π is the Chern number.

Explained in: [Berry phases and Chern number of gapped graphene](https://paoflow.readthedocs.io/en/latest/learn/tutorials/tutorial06.html)
