import matplotlib.pyplot as plt
import numpy as np
from PAOFLOW import GPAO, PAOFLOW


def main():

    pplt = GPAO.GPAO()
    model = {
        "label": "Kane_Mele",
        "t": 1.0,
        "r_par": 0.0,
        "v_par": 0.0,
        "soc_par": 0.1,
        "alat": 1.0,
    }
    model = {
        "label": "Kane_Mele_Pythtb",
        "t": 1.0,
        "rashba": 0.0,
        "delta": 0.0,
        "soc": 0.1,
    }  # Use PythTB model
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir="./output", verbose=True)
    arrays, attributes = paoflow.data_controller.data_dicts()

    path = "G-M-K-G"
    special_points = {
        "G": [0.0, 0.0, 0.0],
        "K": [2.0 / 3.0, 1.0 / 3.0, 0.0],
        "M": [1.0 / 2.0, 0.0 / 2.0, 0.0],
    }
    paoflow.bands(ibrav=4, nk=100, band_path=path, high_sym_points=special_points)

    # Bandstructure plot
    outputdir = "./output/"
    f_band = outputdir + "bands_0.dat"
    f_symp = outputdir + "kpath_points.txt"

    pplt.plot_bands(f_band, f_symp, None, None, y_lim=(-4, 4))

    paoflow.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=1)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing()
    paoflow.spin_Hall(twoD=True, emin=-4.0, emax=4.0, s_tensor=[[0, 1, 2]])

    # Spin Hall conductivity plot
    shc = np.loadtxt(outputdir + "shcEf_z_xy.dat")

    fig = plt.figure(figsize=(8, 5))
    plt.plot(shc[:, 0], shc[:, 1] * 2 * np.pi, color="green", linewidth=3)
    plt.xlim(-4, 4)
    plt.xlabel(r" E- E$_f$ (eV)")
    plt.ylabel(r"$\sigma^{z}_{xy}\;(e/2\pi)$")
    plt.title(r"SHC Kane-Mele Models")
    plt.savefig(outputdir + "shc_z_xy.png", bbox_inches="tight")

    paoflow.finish_execution()


if __name__ == "__main__":
    main()
