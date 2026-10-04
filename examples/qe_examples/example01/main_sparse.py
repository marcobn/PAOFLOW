from PAOFLOW.PAOFLOW import PAOFLOW
from PAOFLOW.sparse import SparseConfig


def main():
    paoflow = PAOFLOW(
        savedir='silicon.save',
        outputdir='output_sparse',
        smearing='gauss',
        npool=1,
        verbose=True,
        sparse=SparseConfig(threshold=1.0e-4),
    )
    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.doubling_Hamiltonian(nx=1, ny=1, nz=1)
    paoflow.sparse.energy_window(emin=-12.0, emax=2.2)

    paoflow.bands(ibrav=2, nk=2000)
    paoflow.interpolated_hamiltonian(nfft1=12, nfft2=12, nfft3=12)
    # eigenvalues, velocities and smearing widths are one fused mesh pass,
    # run by the first property that needs it
    paoflow.dos(emin=-12.0, emax=2.2, ne=1000)
    paoflow.transport(emin=-12.0, emax=2.2)

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
