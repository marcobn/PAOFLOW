import numpy as np

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='./pt.save')

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    # Inplane band-structure. Notice the oscilations around -10eV. This is due to few k-points. K-mesh 5x5x3 is only for testing.
    path = 'G-X-Y-G'
    special_points = {'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.0, 0.0], 'Y': [0.0, 0.5, 0.0]}
    paoflow.bands(ibrav=0, nk=100, band_path=path, high_sym_points=special_points)

    # Interpolate Bands. K-mesh 10x10x6 is only for testing.
    paoflow.interpolated_hamiltonian(nfft1=10, nfft2=10, nfft3=6)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    # Calculating total SHC
    print('               Calculating total SHC')
    paoflow.spin_Hall(emin=-1.0, emax=1.0, ne=100, s_tensor=[[0, 1, 2]])

    shc_total = np.loadtxt('./output/shcEf_z_xy.dat')

    # SHC contribution from the first layer (atomic site = 0)
    # shc_proj is an array with the sites to project indices. Here we are projeting on site zero. First layer.

    print('               Calculating First Layer SHC')
    paoflow.spin_Hall(twoD=False, emin=-1.0, emax=1.0, ne=100, s_tensor=[[0, 1, 2]], shc_proj=[0])

    # Calculating SHC for each Layer
    print('               Calculating SHC for each Layer')
    layers = 4  # Number of layers

    shc_layer = np.zeros((layers, 100, 2), dtype=float)
    shc_ef = np.zeros(layers, dtype=float)

    for i in range(layers):
        paoflow.spin_Hall(emin=-1, emax=1, ne=100, s_tensor=[[0, 1, 2]], shc_proj=[i])

        shc_layer[i] = np.loadtxt('./output/shcEf_z_xy.dat')
        shc_ef[i] = shc_layer[i, 50, 1]  # SHC at Fermi Level

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
