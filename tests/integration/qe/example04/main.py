import numpy as np

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='./fe.save')
    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian()
    paoflow.bands(ibrav=3, nk=100)
    paoflow.topology(spol=2, ipol=1, jpol=2, eff_mass=True, Berry=True)
    paoflow.interpolated_hamiltonian()
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(do_pdos=False, ne=100)
    paoflow.anomalous_Hall(do_ac=True, emin=-8.0, emax=4.0, ne=100, a_tensor=np.array([[0, 1]]))
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
