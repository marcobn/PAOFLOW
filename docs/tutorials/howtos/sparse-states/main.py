from PAOFLOW.PAOFLOW import PAOFLOW

savedir = 'tutorial01/silicon.save'


def si_hamiltonian(outputdir: str, sparse_config: dict) -> PAOFLOW:
    """The truncated PAO Hamiltonian of Si on the sparse backend."""
    paoflow = PAOFLOW(
        savedir=savedir,
        outputdir=outputdir,
        smearing='gauss',
        sparse=True,
        sparse_config=sparse_config,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()
    return paoflow


# The window derived from the property ranges
paoflow = si_hamiltonian('output_auto', {'bond_order': 36})
paoflow.bands(ibrav=2, nk=500)
paoflow.interpolated_hamiltonian()
with paoflow.sparse.fused():
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
paoflow.finish_execution()

# A fixed number of bands
sparse_config = {'bond_order': 36, 'nev': 8}
paoflow = si_hamiltonian('output_nev', sparse_config)
paoflow.interpolated_hamiltonian()
with paoflow.sparse.fused():
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
paoflow.finish_execution()

# An interior window with fixed bounds
sparse_config = {'bond_order': 36, 'interior_window': {'elo': -1.5, 'ehi': 2.5}}
paoflow = si_hamiltonian('output_interior', sparse_config)
paoflow.bands(ibrav=2, nk=500)
paoflow.finish_execution()

# An interior window around the property ranges
sparse_config = {'bond_order': 36, 'interior_window': {}}
paoflow = si_hamiltonian('output_interior_auto', sparse_config)
paoflow.interpolated_hamiltonian()
with paoflow.sparse.fused():
    paoflow.dos(emin=-1.0, emax=1.5, ne=500)
    paoflow.transport(emin=-1.0, emax=1.5)
    paoflow.doping(emin=-12.0, emax=2.0, doping_conc=-1.0e19)  # needs every occupied state
paoflow.bands(ibrav=2, nk=500)  # after a property: the window comes from its range
paoflow.finish_execution()
