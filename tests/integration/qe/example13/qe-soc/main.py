import tbmodels
import z2pack

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='./Bi.save')

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.write_Hamiltonian(fname='Bi_bilayer_HRs.dat')
    path = 'M-G-K-M'

    special_points = {'M': [0.0, 0.5, 0.0], 'G': [0.0, 0.0, 0.0], 'K': [1.0 / 3.0, 1.0 / 3.0, 0.0]}
    paoflow.bands(ibrav=0, nk=100, band_path=path, high_sym_points=special_points)

    print('#######################################################')
    print('                     Z2PACK                            ')
    print('#######################################################')

    model = tbmodels.Model.from_wannier_files(hr_file='./output/Bi_bilayer_HRs.dat')
    system = z2pack.tb.System(model, bands=30)

    result = z2pack.surface.run(system=system, surface=lambda t1, t2: [t1 / 2, t2, 0], load=False)

    print('Z2 topological invariant :   {0}'.format(z2pack.invariant.z2(result)))

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
