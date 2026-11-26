# Gapped graphene model: Berry phase and Berry flux around K

The honeycomb tight-binding model with a sublattice potential (`{'label': 'graphene', 't': -1.0,
'delta': -0.2}`, eV) passed to `PAOFLOW(model=...)`: no DFT run. The script computes

- the band structure along Γ-M-K-Γ;
- the Berry phase of the lower band, of the upper band and of both on a circle of radius 0.05
  around K (`berry_phase(kspace_method='circle', ...)`);
- the Berry flux of the same bands through a square patch of side 0.1 around K
  (`berry_phase(kspace_method='square', ...)`), and a map of the Berry curvature of the lower
  band on the patch.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example graphene_berry_phase_model` copies the case into
  `./graphene_berry_phase_model`; then run the step in that folder.

## Steps

1. `python graphene2.py` (a few seconds): prints the Berry phases and fluxes and writes
   `output/` (`bands_0.dat`, `circle_berry_phase_*.dat`, `square_berry_phase_*.dat` with their
   `*_kgrid_corners.dat`, and the curvature map `berry_phase_k.pdf`).

## What to look at

The Berry phase around K is −2.07 rad for the lower band and +2.07 rad for the upper one, and
vanishes for the two together; the Berry flux through the square patch is ∓2.18 rad. The gap
`delta` spreads the π Berry phase of the Dirac cone over the valley, so a small loop catches part
of it, with opposite signs for the two bands. `Reference/` holds the expected outputs. The
commented line in `graphene2.py` runs the PythTB version of the model (`graphene_pythtb`); the
other models of `PAOFLOW(model=...)` are listed in the README of the case `graphene_tb_model`.
