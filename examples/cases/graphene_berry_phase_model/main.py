from PAOFLOW import PAOFLOW


def main():
    # Gapped graphene: bands and Berry phases on loops around the valley K'
    graphene = PAOFLOW.PAOFLOW(
        model={'label': 'graphene', 't': -1.0, 'delta': -0.2}, outputdir='graphene', verbose=False
    )
    points = {'G': [0.0, 0.0, 0.0], 'M': [0.5, 0.0, 0.0], 'K': [2 / 3, 1 / 3, 0.0]}
    graphene.bands(ibrav=4, nk=300, band_path='G-M-K-G', high_sym_points=points)
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

    # Chern number: Berry flux of the lower band through the whole zone
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


if __name__ == '__main__':
    main()
