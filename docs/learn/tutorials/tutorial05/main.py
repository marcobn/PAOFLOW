import numpy as np

from PAOFLOW import PAOFLOW


def main():
    # Kane-Mele model: bands, then the spin Hall conductivity at two broadenings
    model = {'label': 'Kane_Mele', 't': 1.0, 'soc_par': 0.1, 'r_par': 0.0, 'v_par': 0.0}
    paoflow = PAOFLOW.PAOFLOW(model=model, outputdir='kane_mele', verbose=False)

    points = {'G': [0.0, 0.0, 0.0], 'M': [0.5, 0.0, 0.0], 'K': [2 / 3, 1 / 3, 0.0]}
    paoflow.bands(ibrav=4, nk=100, band_path='G-M-K-G', high_sym_points=points)

    paoflow.interpolated_hamiltonian(nfft1=60, nfft2=60, nfft3=1)
    paoflow.pao_eigh()
    paoflow.gradient_and_momenta()
    paoflow.adaptive_smearing(min_width=0.02)

    for delta in (0.005, 0.05):
        paoflow.spin_Hall(
            twoD=True,
            emin=-4.0,
            emax=4.0,
            s_tensor=[[0, 1, 2]],
            delta=delta,
            fermi_dw=-4.0,  # below the lowest band: the curvature map sums all occupied bands
            fermi_up=0.0,
        )

    # gapped graphene: Berry phases on loops around the valley K'
    graphene = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': -1.0, 'delta': -0.2}, outputdir='graphene', verbose=False
    )
    k_prime = [1 / 3, 2 / 3, 0.0]

    for sub, tag in (([0], 'lower'), ([1], 'upper'), ([0, 1], 'both')):
        graphene.berry_phase(
            kspace_method='circle',
            kcenter=k_prime,
            kradius=0.05,
            nk1=51,
            sub=sub,
            closed=True,
            method='berry',
            fname=f'circle_{tag}',
        )

    graphene.berry_phase(
        kspace_method='square',
        nk1=51,
        nk2=51,
        kxlim=[k_prime[0] - 0.05, k_prime[0] + 0.05],
        kylim=[k_prime[1] - 0.05, k_prime[1] + 0.05],
        sub=[0],
        method='berry',
        fname='patch_lower',
    )

    # Berry phase against the radius of the loop, gapped and gapless graphene
    radii = [0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18, 0.25]
    gapless = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': -1.0, 'delta': 0.0}, outputdir='gapless', verbose=False
    )
    for model in (graphene, gapless):
        for radius in radii:
            model.berry_phase(
                kspace_method='circle',
                kcenter=k_prime,
                kradius=radius,
                nk1=201,
                sub=[0],
                closed=True,
                method='berry',
                fname=f'radius_{radius}',
            )

    # Chern numbers: Berry flux through the whole zone, gapped graphene and the Haldane model
    graphene.berry_phase(
        kspace_method='square',
        nk1=60,
        nk2=60,
        kxlim=[0.0, 1.0],
        kylim=[0.0, 1.0],
        sub=[0],
        method='berry',
        fname='zone_lower',
    )

    haldane_model = {
        'label': 'haldane_pythtb',
        'delta': 0.0,
        't1': -1.0,
        't2': 0.15,
        'phi': np.pi / 2,
    }
    haldane = PAOFLOW.PAOFLOW(model=haldane_model, outputdir='haldane', verbose=False)
    haldane.bands(ibrav=4, nk=600, band_path='G-M-K-G', high_sym_points=points)
    haldane.berry_phase(
        kspace_method='square',
        nk1=60,
        nk2=60,
        kxlim=[0.0, 1.0],
        kylim=[0.0, 1.0],
        sub=[0],
        method='berry',
        fname='zone_lower',
    )


if __name__ == '__main__':
    main()
