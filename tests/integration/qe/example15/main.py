from PAOFLOW import PAOFLOW

paoflow = PAOFLOW.PAOFLOW(
    savedir='./silicon.save/', outputdir='./output/', smearing=None, verbose=True
)
paoflow.jdos(delta=0.1, emin=0.0, emax=4.0, ne=100, jdos_smeartype='gauss')

basis_path = '../../../BASIS/'
basis_config = {'Si': ['3S', '3P', '3D', '4S', '4P']}
paoflow.projections(configuration=basis_config, basispath=basis_path, internal=True)
paoflow.projectability()
paoflow.pao_hamiltonian(expand_wedge=True)

paoflow.pao_eigh()
paoflow.gradient_and_momenta()
paoflow.dielectric_tensor(delta=0.1, emax=4.0, ne=100, d_tensor='diag')
