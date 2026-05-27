import numpy as np


def orbital_array(data_controller):
    """Count the number of spin-orbit orbitals per atom.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required array: ``atoms`` (list of element labels).
        Required dictionary: ``shells`` (maps element label to a list of
        angular momentum quantum numbers :math:`l`).
        Required attribute: ``dftSO`` (bool).

    Returns
    -------
    np.ndarray, shape ``(natoms,)``, int
        Number of spinor orbitals per atom:  :math:`2(2l+1)` per shell
        (:math:`l=0 \\to 2`, :math:`l=1 \\to 6`, :math:`l=2 \\to 10`,
        :math:`l=3 \\to 14`).

    Notes
    -----
    When ``dftSO`` is ``False`` the function returns an array of zeros.
    """
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
    """Build the site-projected identity operator for selected atoms.

    Parameters
    ----------
    data_controller : DataController
        Object providing ``data_arrays`` and ``data_attributes``.
        Required arrays: ``naw`` (shape ``(natoms,)`` — number of orbitals
        per atom).
        Required attribute: ``nawf``.
    proj_array : np.ndarray of int
        Indices (0-based) of the atoms whose orbitals are to be included
        in the projection operator.

    Returns
    -------
    np.ndarray, shape ``(nawf, nawf)``, float
        Block-diagonal projection matrix :math:`P` with identity blocks
        in the orbital sub-spaces of the selected atoms and zeros elsewhere.

    Notes
    -----
    For each atom index ``i`` in ``proj_array`` the corresponding
    :math:`n_{aw}[i] \\times n_{aw}[i]` identity block is placed at the
    appropriate diagonal position in :math:`P`.
    """
    arry, attr = data_controller.data_dicts()

    P = np.zeros((attr['nawf'], attr['nawf']), dtype=float)

    for i in range(proj_array.shape[0]):
        idx = np.sum(arry['naw'][0 : proj_array[i]])
        fdx = idx + arry['naw'][proj_array[i]]

        P[idx:fdx, idx:fdx] = np.eye(arry['naw'][proj_array[i]])

    return P
