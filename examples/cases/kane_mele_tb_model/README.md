# Kane-Mele model: bands and spin Hall conductivity

The Kane-Mele model of the quantum spin Hall effect (C. L. Kane and E. J. Mele, Phys. Rev. Lett.
95, 226801 (2005)) in its PythTB version (`{'label': 'Kane_Mele_Pythtb', 't': 1.0, 'soc': 0.1,
'rashba': 0.0, 'delta': 0.0}`), passed to `PAOFLOW(model=...)`: no DFT run. The script draws the
band structure along Γ-M-K-Γ, interpolates the Hamiltonian on a 60 × 60 grid and computes the
spin Hall conductivity σ^z_xy of the sheet as a function of the Fermi energy.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example kane_mele_tb_model` copies the case into
  `./kane_mele_tb_model`; then run the step in that folder.

## Steps

1. `python kane_mele.py` (a few seconds): opens the band-structure plot (`GPAO.plot_bands`;
   close the window to go on), then writes `output/` (`bands_0.dat`, `shcEf_z_xy.dat`, the spin
   Berry curvature `Spin_Berry_z_xy.bxsf`) and the plot `output/shc_z_xy.png`.

## What to look at

`shc_z_xy.png` / `shcEf_z_xy.dat`: σ^z_xy = 1 e/2π (0.999 on this grid) across the spin-orbit
gap, the quantum spin Hall plateau. `Reference/` holds the expected outputs. The first `model`
dictionary of `kane_mele.py` is the legacy `'Kane_Mele'` label (`kane_mele_model` of
`PAOFLOW.models.models`, the same bands); the other models of `PAOFLOW(model=...)` are listed in
the README of the case `graphene_tb_model`. The case `kane_mele_nlo_model` computes the shift
current of the same model.
