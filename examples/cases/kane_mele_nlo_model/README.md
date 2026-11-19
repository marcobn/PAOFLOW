# Kane-Mele model: shift current across the topological transition

The Kane-Mele model with a staggered potential λ_v (`delta`), intrinsic spin-orbit coupling
λ_SO = 0.06 eV and Rashba coupling λ_R = 0.05 eV (t = 1 eV), built by
`PAOFLOW.models.models.kane_mele_model` and passed to `PAOFLOW(model=...)`: no DFT run. Without
Rashba the gap at K closes at λ_v = 3√3 λ_SO (quantum spin Hall below, trivial above); the script
computes the shift current of the sheet at λ_v = 0.5 and 1.5 times that value. The staggered
potential breaks inversion (D3h, C3v with Rashba; `a_1` along x): σ^{yyy} = −σ^{yxx} = −σ^{xxy},
σ^{xxx} = 0.

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example kane_mele_nlo_model` copies the case into
  `./kane_mele_nlo_model`; then run the step in that folder.

## Steps

1. `python kane_mele_nonlinear.py`: two runs on 300 × 300 k-points (shift current only, 701
   photon energies up to 7 eV, η = 0.03 eV), about 10 s; writes `output_qsh/` and
   `output_trivial/` (`shift_<abc>.dat`, `shiftm_<abc>.dat` in nm µA/V²).

## What to look at

`shift_yyy.dat` in the two folders: the onset at the gap and the change of the spectrum across
the band inversion (peaks of about 15 and 19 nm µA/V²); `shift_yxx.dat` = −`shift_yyy.dat` and
`shift_xxx.dat` vanishes to machine precision in both phases. The dict label
`{'label': 'Kane_Mele', 't', 'soc_par', 'r_par', 'v_par' (units of t), 'alat' (angstrom)}` builds
the same model; the case `kane_mele_tb_model` has its bands and spin Hall conductivity, and the
README of `graphene_tb_model` lists the other models. Formulas and conventions:
`docs/internals/nonlinear-optics.md`.
