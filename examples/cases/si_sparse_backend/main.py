from PAOFLOW import PAOFLOW


def main():
    # input: the QE runs of this case (scf.in, nscf.in, proj.in) write silicon.save
    savedir = 'silicon.save'
    outputdir = 'output'

    # truncated PAO Hamiltonian on the sparse backend
    paoflow = PAOFLOW.PAOFLOW(
        savedir=savedir,
        outputdir=outputdir,
        smearing='gauss',
        sparse=True,
        sparse_config={'bond_order': 36},
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()  # H(R) becomes a list of hoppings, truncated at the 36th shell
    paoflow.save_sparse_hamiltonian('silicon_bo36.npz')  # to outputdir, for the dense check

    # 16-atom cell by doubling, its folded bands, then DOS and transport in one pass over the mesh
    paoflow.doubling_Hamiltonian(1, 1, 1)  # before bands() and every property
    paoflow.bands(ibrav=2, nk=500)
    paoflow.interpolated_hamiltonian(
        nfft1=12, nfft2=12, nfft3=12
    )  # the density of 24^3 for 2 atoms
    with paoflow.sparse.fused():
        paoflow.dos(emin=-12.0, emax=2.2, ne=1000, do_pdos=False)
        paoflow.transport(emin=-12.0, emax=2.2)
    paoflow.finish_execution()

    # check: the same truncated Hamiltonian on the dense pipeline, in the 2-atom cell
    dense = PAOFLOW.PAOFLOW(outputdir='output_dense', restart=True, smearing='gauss')
    dense.load_sparse_hamiltonian('output/silicon_bo36.npz')  # replaces the first three steps
    dense.interpolated_hamiltonian()
    dense.pao_eigh()
    dense.gradient_and_momenta()
    dense.adaptive_smearing()
    dense.dos(emin=-12.0, emax=2.2, ne=1000, do_pdos=False)
    dense.transport(emin=-12.0, emax=2.2)
    dense.finish_execution()


if __name__ == '__main__':
    main()
