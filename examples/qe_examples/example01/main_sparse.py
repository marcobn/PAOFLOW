# Sparse counterpart of main.py: identical call sequence, purely sparse
# backend from pao_hamiltonian onward.  All parameters live here — edit
# `threshold` (eV) to trade accuracy for memory; the conversion prints a
# rigorous bound on the eigenvalue shift the truncation can cause.
from PAOFLOW.SparsePAOFLOW import SparsePAOFLOW


def main():
    paoflow = SparsePAOFLOW(
        savedir='silicon.save',
        outputdir='output_sparse',
        smearing='gauss',
        npool=1,
        verbose=True,
        threshold=1.0e-4,
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.doubling_Hamiltonian(nx=1, ny=1, nz=1)
    paoflow.energy_window(emin=-12.0, emax=2.2)

    paoflow.bands(ibrav=2, nk=2000)
    paoflow.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
