PAOFLOW can be called with PAOFLOW(model=tbmodel) where tbmodel is a dictionary (predefined models below) or a user defined PythTB object.

Predefined hard-coded models
cubium: tbmodel = {'label':'cubium', 't':?}
cubium2: tbmodel = {'label':'cubium2', 't':?, 'Eg':?}
graphene: tbmodel = {'label':'graphene', 'delta':?, 't':?}
kane_mele: tbmodel = {'label':'kane_mele', 't':?, 'soc_par':?, 'r_par':?, 'v_par':?, 'alat':?}
           now kane_mele_model below: r_par and v_par (units of t) default 0, alat in angstrom (default 1)

Predefined PythTB models (PythTB V2.0.0; converted to hopping lists, so they obey the conventions of the optical
responses: h(i -> j, R) at the HRs index -R, one centre per orbital, spin-orbitals in block order)
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
tmd:       tbmodel = {'label':'tmd', 'material':'MoS2'|'WS2'|'MoSe2'|'WSe2'|'MoTe2'|'WTe2', 'functional':'GGA'|'LDA',
           'soc':True|False|lambda, 'thickness':?} three-band d-orbital model of the MX2 monolayers (Liu et al.,
           PRB 88, 085433 (2013)): D3h SHG and shift current, spin-valley locking with SOC
kane_mele_model(t, soc, rashba, delta, nelec, alat): Kane-Mele model (spinful), QSH below delta = 3 sqrt(3) soc;
           pass the HoppingModel itself: PAOFLOW(model=kane_mele_model(...)) (or the 'kane_mele' label above)
The same models are Python functions (honeycomb_model, rice_mele_model, weyl_model, single_band_model, tmd_model,
kane_mele_model in PAOFLOW.models.models) returning a HoppingModel, which PAOFLOW(model=...) also accepts; metals need
nonlinear_optics(fermi_energy=..., temperature=...).

Graphene_nonlinear/: shift current, injection currents and SHG of gapped graphene (tbmodel = {'label':'graphene', 'delta':1.0, 't':2.7}) with PAOFLOW.nonlinear_optics (docs/internals/nonlinear-optics.md).
TMD_nonlinear/: shift current, injection currents and SHG of monolayer MoS2 from the three-band model with spin-orbit coupling.
Kane_Mele_nonlinear/: shift current of the Kane-Mele model in the quantum-spin-Hall and trivial phases.
