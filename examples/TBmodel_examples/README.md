PAOFLOW can be called with PAOFLOW(model=tbmodel) where tbmodel is a dictionary (predefined models below) or a user defined PythTB object.

Predefined hard-coded models
cubium: tbmodel = {'label':'cubium', 't':?}
cubium2: tbmodel = {'label':'cubium2', 't':?, 'Eg':?}
graphene: tbmodel = {'label':'graphene', 'delta':?, 't':?}
kane_mele: tbmodel = {'label':'kane_mele', 'v_par':?, 't':?, 'soc_par':?, 'r_par':?}

Predefined PythTB models (PythTB V2.0.0)
cubium: tbmodel = {'label':'cubium_pythtb', 't':?}
cubium2: tbmodel = {'label':'cubium2_pythtb', 't':?, 'Eg':?}
ssh: tbmodel = {'label':'ssh_pythtb', 'v':?, 'w':?}
checkerboard: tbmodel = {'label':'checkerboard_pythtb', 'delta':?, 't':?}
graphene: tbmodel = {'label':'graphene_pythtb', 'delta':?, 't':?}
haldane: tbmodel = {'label':'haldane_pythtb', 'delta':?, 't1':?, 't2':?, 'phi':?}
kane_mele: tbmodel = {'label':'kane_mele_pythtb', 'delta':?, 't':?, 'soc':?, 'rashba':?}
fu_kane_mele: tbmodel = {'label':'fu_kane_mele_pythtb', 't':?, 'soc':?, 'dt':?}
See https://pythtb.readthedocs.io/en/latest/ or src/PAOFLOW/models/models.py for details

Hopping-list models (the validation models of the optical responses; alat in bohr, energies in eV)
honeycomb: tbmodel = {'label':'honeycomb', 'delta':?, 't':?, 't2':?, 'phi':?, 'strain':?, 'nelec':?, 'alat':?}
           gapped graphene (t2 = 0), Haldane model (t2 != 0), strain lowers D3h (all in-plane components)
rice_mele: tbmodel = {'label':'rice_mele', 't':?, 'dt':?, 'delta':?, 'nelec':?, 'alat':?}
           1D ferroelectric chain: shift current xxx, sign set by delta and dt
weyl:      tbmodel = {'label':'weyl', 't':?, 'b':?, 'alat':?}
           two Weyl nodes at energies +-b, one spinful orbital, one electron: quantized circular injection
single_band: tbmodel = {'label':'single_band', 'hoppings':{(1,0,0):?, ...}, 'lattice_vectors':?, 'alat':?,
           'nelec':?, 'real_space_grid':?} one band + a decoupled empty band: Drude terms only
The same models are Python functions (honeycomb_model, rice_mele_model, weyl_model, single_band_model in
PAOFLOW.models.models) returning a HoppingModel; metals need nonlinear_optics(fermi_energy=..., temperature=...).

Graphene_nonlinear/: shift current, injection currents and SHG of gapped graphene (tbmodel = {'label':'graphene', 'delta':1.0, 't':2.7}) with PAOFLOW.nonlinear_optics (docs/internals/nonlinear-optics.md).
