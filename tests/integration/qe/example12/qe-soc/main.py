import numpy as np

from PAOFLOW import GPAO, PAOFLOW


def main():
    pplt = GPAO.GPAO()
    paoflow = PAOFLOW.PAOFLOW(savedir='./Bi.save')

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    path = 'xX-G-X'
    special_points = {'xX': [-0.5, 0.0, 0.0], 'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.0]}
    paoflow.bands(ibrav=0, nk=100, band_path=path, high_sym_points=special_points)

    # Projection on the outmost sites of the nanoribbon
    # index of the sites to obtain the projection.
    paoflow.site_projected_bands(site_proj=np.array([0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23]))

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
