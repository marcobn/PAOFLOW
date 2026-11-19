# Cubium: band structure of the one-band cubic model

The simplest tight-binding model of `PAOFLOW(model=...)`: one `s` orbital on the simple-cubic
lattice with the nearest-neighbour hopping `t = 1 eV` (`{'label': 'cubium', 't': 1.0}`). No DFT
run, no projection and no basis: `bands()` draws the cosine band along Γ-X-M-Γ-R.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example cubium_tb_model` copies the case into
  `./cubium_tb_model`; then run the step in that folder.

## Steps

1. `python cubium.py` (about a second): writes `output/bands_0.dat` and
   `output/kpath_points.txt`.

`Reference/` holds the expected outputs. Plot with `GPAO.plot_bands('output/bands_0.dat',
'output/kpath_points.txt')`. The commented line in `cubium.py` runs the PythTB version of the
model (`cubium_pythtb`); the other models of `PAOFLOW(model=...)` are listed in the README of the
case `graphene_tb_model`.
