from PAOFLOW import PAOFLOW


def main():
    # input: tutorial01/silicon.save from tutorial_assets.tar.gz, extracted here
    savedir = 'tutorial01/silicon.save'
    outputdir = 'output'

    paoflow = PAOFLOW.PAOFLOW(
        savedir=savedir,
        outputdir=outputdir,
        smearing='gauss',
        verbose=True,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    # bands along the default fcc path
    paoflow.bands(ibrav=2, nk=2000)

    # DOS and PDOS on the 24x24x24 mesh, with adaptive smearing
    paoflow.interpolated_hamiltonian()
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing(min_width=0.01)
    paoflow.dos(emin=-12.0, emax=2.0, ne=2801)
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
