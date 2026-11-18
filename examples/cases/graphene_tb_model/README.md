# Graphene: band structure of the tight-binding model

The nearest-neighbour honeycomb model of graphene (`{'label': 'graphene', 't': 2.7, 'delta': 0.0}`,
eV) passed to `PAOFLOW(model=...)`: no DFT run, no projection and no basis. `bands()` draws the
band structure along Γ-M-K-Γ, with the Dirac cone at K (`delta` opens a gap `delta` at K).

## Run it

- **PAOFLOW Studio**: Setup tab, *Examples…*, pick this case and a folder name; the Runs tab
  lists the step below.
- **Command line**: `paoflow-gen --example graphene_tb_model` copies the case into
  `./graphene_tb_model`; then run the step in that folder.

## Steps

1. `python graphene.py` (about a second): writes `output/bands_0.dat` and
   `output/kpath_points.txt`.

`Reference/` holds the expected outputs. Plot with `GPAO.plot_bands('output/bands_0.dat',
'output/kpath_points.txt')`. The commented line in `graphene.py` runs the PythTB version of the
same model (`graphene_pythtb`). For the model properties of the generator (DOS, Hall, optics,
non-linear optics on the hopping-list models below), use the workflow `tb_model` of
`paoflow-gen` / PAOFLOW Studio.

## The models of `PAOFLOW(model=...)`

`PAOFLOW(model=tbmodel)` takes a dictionary (the predefined models below) or a PythTB model.

Predefined hard-coded models:

| label | parameters |
|---|---|
| `cubium` | `{'label': 'cubium', 't': ...}` |
| `cubium2` | `{'label': 'cubium2', 't': ..., 'Eg': ...}` |
| `graphene` | `{'label': 'graphene', 't': ..., 'delta': ... (default 0), 'nelec': ... (default 2)}` |
| `kane_mele` | `{'label': 'kane_mele', 't': ..., 'soc_par': ..., 'r_par': ..., 'v_par': ..., 'alat': ...}`: now `kane_mele_model` below; `r_par` and `v_par` (units of t) default 0, `alat` in angstrom (default 1) |

Predefined PythTB models (PythTB 2.0.0; converted to hopping lists, so they obey the conventions
of the optical responses: h(i → j, R) at the `HRs` index −R, one centre per orbital,
spin-orbitals in block order):

| model | parameters |
|---|---|
| cubium | `{'label': 'cubium_pythtb', 't': ...}` |
| cubium2 | `{'label': 'cubium2_pythtb', 't': ..., 'Eg': ...}` |
| SSH | `{'label': 'ssh_pythtb', 'v': ..., 'w': ...}` |
| checkerboard | `{'label': 'checkerboard_pythtb', 'delta': ..., 't': ...}` |
| graphene | `{'label': 'graphene_pythtb', 'delta': ..., 't': ...}` |
| Haldane | `{'label': 'haldane_pythtb', 'delta': ..., 't1': ..., 't2': ..., 'phi': ...}` |
| Kane-Mele | `{'label': 'kane_mele_pythtb', 'delta': ..., 't': ..., 'soc': ..., 'rashba': ...}` |
| Fu-Kane-Mele | `{'label': 'fu_kane_mele_pythtb', 't': ..., 'soc': ..., 'dt': ...}` |

PythTB models carry no filling: add `'nelec'` (electrons per cell) to any `*_pythtb` label.
`haldane_pythtb(delta, t1, t2, phi)` = `honeycomb_model(2 delta, t1, t2, -phi)` (on-site ±delta,
opposite chirality). See <https://pythtb.readthedocs.io/en/latest/> or
`src/PAOFLOW/models/models.py` for details.

Hopping-list models (the validation models of the optical responses; `alat` in bohr, energies
in eV):

| label | parameters and use |
|---|---|
| `honeycomb` | `{'label': 'honeycomb', 'delta', 't', 't2', 'phi', 'strain', 'nelec', 'alat'}`: gapped graphene (t2 = 0), Haldane model (t2 ≠ 0); `strain` lowers D3h (all in-plane components) |
| `rice_mele` | `{'label': 'rice_mele', 't', 'dt', 'delta', 'nelec', 'alat'}`: 1D ferroelectric chain, shift current xxx with the sign set by delta and dt |
| `weyl` | `{'label': 'weyl', 't', 'b', 'alat'}`: two Weyl nodes at energies ±b, one spinful orbital, one electron: quantized circular injection |
| `single_band` | `{'label': 'single_band', 'hoppings': {(1, 0, 0): ..., ...}, 'lattice_vectors', 'alat', 'nelec', 'real_space_grid'}`: one band + a decoupled empty band, Drude terms only |
| `tmd` | `{'label': 'tmd', 'material': 'MoS2' \| 'WS2' \| 'MoSe2' \| 'WSe2' \| 'MoTe2' \| 'WTe2', 'functional': 'GGA' \| 'LDA', 'soc': True \| False \| lambda, 'thickness'}`: three-band d-orbital model of the MX2 monolayers (Liu et al., PRB 88, 085433 (2013)): D3h SHG and shift current, spin-valley locking with SOC |

`kane_mele_model(t, soc, rashba, delta, nelec, alat)`: the Kane-Mele model (spinful), QSH below
delta = 3√3 soc; pass the `HoppingModel` itself: `PAOFLOW(model=kane_mele_model(...))` (or the
`'kane_mele'` label above). The same models are Python functions (`honeycomb_model`,
`rice_mele_model`, `weyl_model`, `single_band_model`, `tmd_model`, `kane_mele_model` in
`PAOFLOW.models.models`) returning a `HoppingModel`, which `PAOFLOW(model=...)` also accepts;
metals need `nonlinear_optics(fermi_energy=..., temperature=...)`. A Slater-Koster model is
the label `'Slater_Koster'` (case `si_slater_koster_model`).

The model cases: `cubium_tb_model`, `graphene_tb_model` (this one), `graphene_berry_phase_model`,
`kane_mele_tb_model`, `si_slater_koster_model`, and the non-linear optics of the models
`graphene_nlo_model`, `mos2_nlo_model`, `kane_mele_nlo_model`
(`docs/internals/nonlinear-optics.md`).
