from __future__ import annotations

from PAOFLOW import PAOFLOW


def main() -> None:
    model1 = {
        'label': 'Slater_Koster',
        'model': {
            'a_vectors': [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
            'atoms': {
                '0': {
                    'name': 'Si',
                    'tau': [0.0, 0.0, 0.0],
                    'orbitals': ['s', 'px', 'py', 'pz'],
                    's': -5.19278,
                    'px': 1.05825,
                    'py': 1.05825,
                    'pz': 1.05825,
                },
                '1': {
                    'name': 'Si',
                    'tau': [0.25, 0.25, 0.25],
                    'orbitals': ['s', 'px', 'py', 'pz'],
                    's': -5.19278,
                    'px': 1.05825,
                    'py': 1.05825,
                    'pz': 1.05825,
                },
            },
            'hoppings': {
                'sss': -2.36233,
                'sps': 1.86401,
                'pps': 2.85882,
                'ppp': -0.94687,
            },
        },
    }

    paoflow = PAOFLOW.PAOFLOW(
        savedir=None,
        model=model1,
        outputdir='slaterkoster',
        smearing='gauss',
        verbose=True,
    )

    path = 'L-G-X'
    special_points = {'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.5, 0.0], 'L': [0.5, 0.5, 0.5]}
    paoflow.bands(ibrav=2, nk=500, band_path=path, high_sym_points=special_points)


if __name__ == '__main__':
    main()
