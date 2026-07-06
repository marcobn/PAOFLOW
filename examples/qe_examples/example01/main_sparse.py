from PAOFLOW import SparsePAOFLOW


def main():

    paoflow = SparsePAOFLOW.SparsePAOFLOW(
        savedir='silicon.save',
        outputdir='output_sparse',
        smearing='gauss',
        npool=1,
        verbose=True,
        sparse_threshold=1.0e-3,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.doubling_Hamiltonian(nx=1, ny=1, nz=1)

    # Selected-eigenvalue band structure on the default ibrav=2 FCC path.
    paoflow.bands(ibrav=2, nk=2000)
    paoflow.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)

    # Selected-window eigenpairs over the full BZ grid (sparse eigsh).
    paoflow.pao_eigh()

    # Band-diagonal group velocities via Hellmann-Feynman (sparse dH/dk).
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
