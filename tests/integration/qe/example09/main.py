from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='MoP2.save', verbose=False, outputdir='./output')
    paoflow.read_atomic_proj_QE()
    paoflow.projectability(pthr=0.95)
    paoflow.pao_hamiltonian(symmetrize=True, thresh=1.0e-10, max_iter=2)
    paoflow.write_Hamiltonian()
    paoflow.find_weyl_points(search_grid=[2, 2, 2])
    paoflow.finish_execution()


if __name__ == '__main__':
    main()
