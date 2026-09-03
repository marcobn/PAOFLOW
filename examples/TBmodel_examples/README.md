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