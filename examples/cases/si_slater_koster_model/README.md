# Si: Slater-Koster sp3 tight-binding bands

A Slater-Koster model of silicon built by `PAOFLOW(model={'label': 'Slater_Koster', 'model': ...})`:
two Si atoms on the diamond lattice with s and p orbitals, on-site energies and the two-centre
parameters (ssσ, spσ, ppσ, ppπ) of D. A. Papaconstantopoulos, *Handbook of the Band Structure of
Elemental Solids* (Springer, 2015) (`model1`; `model0` is a one-atom fcc toy model). No DFT run:
`bands()` draws the band structure along L-Γ-X.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example si_slater_koster_model` copies the case into
  `./si_slater_koster_model`; then run the step in that folder.

## Steps

1. `python slater_koster.py` (about a second): writes `output/bands_0.dat`,
   `output/kpath_points.txt` and the figure `output/bands.pdf`.

The `model` dictionary sets `a_vectors`, the `atoms` (name, position `tau`, `orbitals` and their
on-site energies) and the `hoppings`; set `model = model0` for the toy model. Fitting the
Slater-Koster parameters to a PAO Hamiltonian is `PAOFLOW.models.sk_fitting.SKFitter`; the other
models of `PAOFLOW(model=...)` are listed in the README of the case `graphene_tb_model`.
