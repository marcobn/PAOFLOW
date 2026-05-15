import numpy as np


def orbital_array(data_controller):
    arry, attr = data_controller.data_dicts()

    naw = np.zeros(len(arry['atoms']), dtype=int)

    if attr['dftSO'] == True:
        for i in range(len(arry['atoms'])):
            n_atom = 0
            for j in arry['shells'][arry['atoms'][i]]:
                if j == 0:
                    n_atom += 2
                elif j == 1:
                    n_atom += 3
                elif j == 2:
                    n_atom += 5
                elif j == 3:
                    n_atom += 7
            naw[i] = n_atom
    return naw


def do_projection_operator(data_controller, proj_array):
    arry, attr = data_controller.data_dicts()

    P = np.zeros((attr['nawf'], attr['nawf']), dtype=float)

    for i in range(proj_array.shape[0]):
        idx = np.sum(arry['naw'][0 : proj_array[i]])
        fdx = idx + arry['naw'][proj_array[i]]

        P[idx:fdx, idx:fdx] = np.eye(arry['naw'][proj_array[i]])

    return P
