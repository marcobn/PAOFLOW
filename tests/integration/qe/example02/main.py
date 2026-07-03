from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='al.save', verbose=True)
    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.97)
    paoflow.pao_hamiltonian()
    paoflow.interpolated_hamiltonian()
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(do_pdos=False, delta=0.1, emin=-12.0, emax=3.0, ne=100)
    paoflow.transport(emin=-12.0, emax=3.0, ne=100)
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
