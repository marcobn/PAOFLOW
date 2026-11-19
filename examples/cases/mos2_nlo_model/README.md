# Monolayer MoS2 (three-band model): shift current and second-harmonic generation

The nearest-neighbour three-band model of G.-B. Liu et al., Phys. Rev. B 88, 085433 (2013), for
monolayer MoS2 (GGA parameters, on-site spin-orbit coupling λ L_z S_z): the d_z², d_xy, d_x²−y²
orbitals of the metal, `a_1` along x (`{'label': 'tmd', 'material': 'MoS2', 'functional': 'GGA',
'soc': True}` passed to `PAOFLOW(model=...)`; no DFT run). Point group D3h with the mirror
x → −x, so σ^{yyy} = −σ^{yxx} = −σ^{xxy} and σ^{xxx} = 0 (shift current and SHG alike).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example mos2_nlo_model` copies the case into
  `./mos2_nlo_model`; then run the step in that folder.

## Steps

1. `python tmd_nonlinear.py`: 300 × 300 k-points, 501 photon energies up to 5 eV, η = 0.05 eV;
   about 15 s on one core. Writes `output/` with the sheet responses (`twoD=True`, the slab
   height being the cell height, 10 alat by default or `'thickness'` in angstrom):
   `shift_<abc>.dat` (nm µA/V²), `injectc_/injectm_<abc>.dat`, `shgr_/shgi_/shga_<abc>.dat`
   (pm²/V).

## What to look at

`shift_yyy.dat` (peak about 95 nm µA/V²), with `shift_yxx.dat` = `shift_xxy.dat` = −`shift_yyy.dat`
and `shift_xxx.dat` zero to machine precision; the SHG files obey the same relations. Other
materials: `'WS2'`, `'MoSe2'`, `'WSe2'`, `'MoTe2'`, `'WTe2'`; `'functional': 'LDA'` (with `'soc':
<lambda in eV>`, the tabulated lambda being fitted to GGA); `'soc': False` for the
spin-degenerate model. The README of `graphene_tb_model` lists the other models; formulas and
conventions: `docs/internals/nonlinear-optics.md`.
