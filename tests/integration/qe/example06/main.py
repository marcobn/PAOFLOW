import numpy as np

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='alp.save')

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    correction_Hubbard = np.zeros(32, dtype=float)
    correction_Hubbard[1:4] = 0.1
    correction_Hubbard[17:20] = 2.31
    paoflow.add_external_fields(HubbardU=correction_Hubbard)

    paoflow.bands(ibrav=2, nk=2000)
    paoflow.interpolated_hamiltonian(nfft1=24, nfft2=24, nfft3=24)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(do_pdos=False, emin=-8.0, emax=5.0, delta=0.1, ne=100)
    paoflow.transport(emin=-8.0, emax=5.0, ne=100)
    paoflow.dielectric_tensor(emax=6.0, d_tensor=[[0, 0]], ne=100)
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
