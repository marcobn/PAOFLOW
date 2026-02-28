from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from PAOFLOW import PAOFLOW

Ry2eV = 13.60569193


def main() -> None:
    # FCC crystal with one atom per cell as in tightbinding.ipynb notebook
    model0 = {
        'label': 'Slater_Koster',
        'model': {
            'a_vectors': [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]],
            'atoms': {
                '0': {
                    'name': 'Si',
                    'tau': [0.0, 0.0, 0.0],
                    'orbitals': ['s', 'px', 'py', 'pz'],
                    's': -2,
                    'px': -1,
                    'py': -1,
                    'pz': -1,
                },
            },
            'hoppings': {
                'sss': -0.2,
                'sps': 0.01,
                'pps': 0.2,
                'ppp': -0.02,
            },
        },
    }

    # Slater and Koster parameter for Silicon from Dimitris A. Papaconstantopoulos
    # Handbook of the Band Structure of Elemental Solids From Z 1 To 112
    # Springer (2015)
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

    # choose the model
    model = model1

    paoflow = PAOFLOW.PAOFLOW(
        savedir=None,
        model=model,
        outputdir='slaterkoster',
        smearing='gauss',
        verbose=True,
    )

    path = 'L-G-X'
    special_points = {'G': [0.0, 0.0, 0.0], 'X': [0.5, 0.5, 0.0], 'L': [0.5, 0.5, 0.5]}
    paoflow.bands(ibrav=2, nk=500, band_path=path, high_sym_points=special_points)

    _, attr = paoflow.data_controller.data_dicts()
    bands = np.loadtxt(f'{attr["outputdir"]}/bands_0.dat')

    fig1, ax1 = plt.subplots(1, figsize=(6, 4))
    for ib in range(1, bands.shape[1]):
        ax1.plot(bands[:, 0] / Ry2eV / np.pi, bands[:, ib], 'k-')
    ax1.set_xticks([])
    ax1.set_ylabel(r'$\varepsilon$ [eV]')
    fig1.tight_layout()


if __name__ == '__main__':
    main()
