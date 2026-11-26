from PAOFLOW import PAOFLOW


def main():
    # The model replaces the projection steps of a DFT run
    model = {'label': 'Kane_Mele', 't': 1.0, 'soc_par': 0.1, 'r_par': 0.0, 'v_par': 0.0}
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='kane_mele', verbose=False)

    # Bands along a path in crystal coordinates of the reciprocal vectors; ibrav=4 is hexagonal
    points = {'G': [0.0, 0.0, 0.0], 'M': [0.5, 0.0, 0.0], 'K': [2 / 3, 1 / 3, 0.0]}
    paoflow.bands(ibrav=4, nk=100, band_path='G-M-K-G', high_sym_points=points)

    # Bands and velocities on the mesh, and the DOS
    paoflow.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=1)  # 2D: one k-point along z
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing(min_width=0.02)  # floor: twice the 0.01 eV spacing of the DOS grid
    paoflow.dos(emin=-3.5, emax=3.5, ne=701, do_pdos=False)

    # Spin Hall conductivity of the sheet at two broadenings
    for delta in (0.005, 0.05):  # the default 0.05 eV last: its files stay in kane_mele/
        paoflow.spin_Hall(
            twoD=True,
            emin=-4.0,
            emax=4.0,
            s_tensor=[[0, 1, 2]],
            delta=delta,
            fermi_dw=-4.0,  # below the lowest band: the curvature map sums all occupied bands
            fermi_up=0.0,
        )


if __name__ == '__main__':
    main()
