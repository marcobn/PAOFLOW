from PAOFLOW.PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW(
        savedir='silicon.save',
        outputdir='output_sparse',
        smearing='gauss',
        npool=1,
        verbose=True,
        sparse={'threshold': 1.0e-4},
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.doubling_Hamiltonian(nx=1, ny=1, nz=1)
    paoflow.sparse.energy_window(emin=-12.0, emax=2.2)

    paoflow.bands(ibrav=2, nk=2000)
    paoflow.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
    # eigenvalues, velocities, smearing widths, PDOS and the band curvature
    # of the Hall term are one mesh pass for the whole block; the curvature's
    # interband sum makes it solve every state per k-point
    with paoflow.sparse.fused():
        paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
        paoflow.transport(emin=-12.0, emax=2.2, do_hall=True)

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
