import numpy as np

from PAOFLOW import PAOFLOW


def main():
    paoflow = PAOFLOW.PAOFLOW(savedir='./pt.save')
    arry, attr = paoflow.data_controller.data_dicts()

    paoflow.read_atomic_proj_QE()
    paoflow.projectability()
    paoflow.pao_hamiltonian()

    paoflow.adhoc_spin_orbit(
        phi=0.0,
        theta=0.0,
        naw=np.array([9]),  # number of orbitals for each atom
        lambda_p=np.array([0.0]),  # p orbitals SOC strengh for each atom
        lambda_d=np.array([0.5534]),  # d orbitals SOC strengh for each atom
        orb_pseudo=['spd'],
    )  # type of pseudo potential for each atom

    path = 'gG-X-W-K-gG-L-U-W-L-K|U-X'
    special_points = {
        'gG': (0.0, 0.0, 0.0),
        'K': (0.375, 0.375, 0.750),
        'L': (0.5, 0.5, 0.5),
        'U': (0.625, 0.250, 0.625),
        'W': (0.5, 0.25, 0.75),
        'X': (0.5, 0.0, 0.5),
    }

    paoflow.bands(ibrav=2, nk=1000, band_path=path, high_sym_points=special_points)

    paoflow.topology(Berry=True, eff_mass=True, spin_Hall=True, spol=2, ipol=0, jpol=1)
    paoflow.interpolated_hamiltonian()
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.dos(do_pdos=False, emin=-8.0, emax=4.0, ne=100)
    paoflow.spin_Hall(emin=-8.0, emax=4.0, ne=100, s_tensor=[[0, 1, 2]])

    paoflow.finish_execution()


if __name__ == '__main__':
    main()
