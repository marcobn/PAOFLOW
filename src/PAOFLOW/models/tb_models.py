#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
# F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli,
# Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np

## Predefined PythTB models


def cubium_pythtb(t):
    from pythtb import Lattice, TBModel

    cell = np.eye(3)
    orb = [[0, 0, 0]]
    lat = Lattice(cell, orb, periodic_dirs='all')
    model = TBModel(lattice=lat, spinful=False)
    # on-site
    onsite = -6 * t
    model.set_onsite(onsite, ind_i=0)
    # hopping
    model.set_hop(t, 0, 0, [1, 0, 0])
    model.set_hop(t, 0, 0, [0, 1, 0])
    model.set_hop(t, 0, 0, [0, 0, 1])

    return model


def cubium2_pythtb(t, Eg):
    from pythtb import Lattice, TBModel

    cell = np.eye(3)
    orb = [[0, 0, 0], [0, 0, 0]]
    lat = Lattice(cell, orb, periodic_dirs='all')
    model = TBModel(lattice=lat, spinful=False)
    # on-site
    onsite = -Eg / 2 - 6.0 * t
    model.set_onsite(onsite, ind_i=0)
    model.set_onsite(-onsite, ind_i=1)
    # hopping
    model.set_hop(t, 0, 0, [1, 0, 0])
    model.set_hop(t, 0, 0, [0, 1, 0])
    model.set_hop(t, 0, 0, [0, 0, 1])
    model.set_hop(-t, 1, 1, [1, 0, 0])
    model.set_hop(-t, 1, 1, [0, 1, 0])
    model.set_hop(-t, 1, 1, [0, 0, 1])

    return model


def ssh_pythtb(v, w):
    from pythtb import Lattice, TBModel

    r"""Su-Schrieffer-Heeger (SSH) model.

    This function constructs the SSH model with the specified hopping parameters.
    The SSH model is a one-dimensional tight-binding model that describes a chain of atoms
    with alternating hopping parameters. The tight-binding Hamiltonian for the SSH model can be
    written as:

    .. math::
       H = v \sum_{i} (c_{i, 1}^{\dagger} c_{i, 2} + \text{h.c.}) + w \sum_{i} (c_{i, 2}^{\dagger} c_{i+1, 1} + \text{h.c.})


    Parameters
    ----------
    v : float
        The intercell hopping within the unit cell.
    w : float
        The intracell hopping to neighboring unit cells.

    Returns
    -------
    TBModel
        The tight-binding model for the SSH lattice.
    """
    lat_vecs = [[1]]
    orb_vecs = [[0], [1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0])

    model = TBModel(lattice=lat, spinful=False)

    model.set_hop(v, 0, 1, [0])
    model.set_hop(w, 1, 0, [0])

    return model


def checkerboard_pythtb(delta, t):
    from pythtb import Lattice, TBModel

    r"""Checkerboard tight-binding model.

    .. versionadded:: 2.0.0

    This function creates a checkerboard tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D square
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = (0, 1)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \left(0, 0\right), \quad \mathbf{\tau}_2 = \left(\frac{1}{2}, \frac{1}{2}\right)

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.}) + \Delta \sum_i n_i

    Parameters
    ----------
    t : float
        Nearest neighbor hopping amplitude.

    delta : float
        On-site energy. Positive for one sublattice, negative for the other.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [0, 1]]
    orb_vecs = [[0, 0], [1 / 2, 1 / 2]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    model = TBModel(lattice=lat, spinful=False)

    # set on-site energies
    model.set_onsite([-delta, delta], mode='set')

    model.set_hop(t, 1, 0, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])
    model.set_hop(t, 1, 0, [1, 1])

    return model


def graphene_pythtb(delta: float, t: float):
    from pythtb import Lattice, TBModel

    r"""Graphene tight-binding model.

    This function creates a graphene tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i n_i + t \sum_{\langle i,j \rangle} (c_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        On-site energy difference between the two orbitals.
    t : float
        Hopping parameter between nearest neighbor orbitals.

    Returns
    -------
    TBModel
        An instance of the model.
    """

    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    model = TBModel(lattice=lat, spinful=False)

    model.set_onsite([delta / 2, -delta / 2])
    model.set_hop(t, 0, 1, [0, 0])
    model.set_hop(t, 1, 0, [1, 0])
    model.set_hop(t, 1, 0, [0, 1])

    return model


def haldane_pythtb(delta: float, t1: float, t2: float, phi: float = np.pi / 2):
    from pythtb import Lattice, TBModel

    r"""Haldane tight-binding model.

    This function creates a Haldane tight-binding model with the specified
    hopping parameters and on-site energy. The model is defined on a 2D honeycomb
    lattice with two sublattices. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (1, 0), \quad \mathbf{a}_2 = \left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{\tau}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = \Delta \sum_i (-)^i c_i^\dagger c_i + t_1 \sum_{\langle i,j \rangle} (c_i^\dagger c_j
        + \text{h.c.}) + t_2 \sum_{\langle\langle i,j \rangle\rangle} (ic_i^\dagger c_j + \text{h.c.})

    Parameters
    ----------
    delta : float
        Onsite mass term. Opposite sign for the two sublattices.
    t1 : float
        Nearest neighbor hopping amplitude.
    t2 : float
        Next-nearest neighbor hopping amplitude. Peierls phase is included.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Haldane model describes a two-dimensional topological insulator with a
    non-trivial band structure. It is characterized by a finite Chern number
    and exhibits edge states that are protected by time-reversal symmetry [haldane]_.

    References
    ----------
    .. [haldane] Haldane, F. D. M. (1988).
        O(3) Nonlinear :math:`\sigma` Model and the Quantum Hall Effect in Two Dimensions.
        *Physical Review Letters*, 61(20), 2015–2018.
    """

    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])
    model = TBModel(lattice=lat, spinful=False)

    model.set_onsite([-delta, delta], mode='set')

    for lvec in ([0, 0], [-1, 0], [0, -1]):
        model.set_hop(t1, 0, 1, lvec, mode='set')

    for lvec in ([1, 0], [-1, 1], [0, -1]):
        model.set_hop(t2 * np.exp(1j * phi), 0, 0, lvec, mode='set')
    for lvec in ([-1, 0], [1, -1], [0, 1]):
        model.set_hop(t2 * np.exp(1j * phi), 1, 1, lvec, mode='set')

    return model


def kane_mele_pythtb(delta, t, soc, rashba):
    from pythtb import Lattice, TBModel

    r"""Kane-Mele tight-binding model.

    This function creates a Kane-Mele tight-binding model with the specified
    parameters. The model is defined on a 2D honeycomb lattice with two sublattices.
    The lattice vectors are given by:

    .. math::

        \mathbf{a}_1 = a(1, 0), \quad \mathbf{a}_2 = a\left(\frac{1}{2}, \frac{\sqrt{3}}{2}\right),

    and the orbital positions are given by:

    .. math::

        \mathbf{r}_1 = \frac{1}{3} \mathbf{a}_1 + \frac{1}{3} \mathbf{a}_2,
        \quad \mathbf{r}_2 = \frac{2}{3} \mathbf{a}_1 + \frac{2}{3} \mathbf{a}_2

    The Hamiltonian in second-quantized form is given by:

    .. math::

        H = \Delta \sum_{i} c_i^\dagger c_i +
        t \sum_{\langle i,j \rangle} ( c_i^\dagger c_j + h.c.) +
        \lambda_{SO} \sum_{\langle \langle i,j \rangle \rangle} ( c_i^\dagger \sigma_z c_j + \text{h.c.}) + \\
        \lambda_{R} \sum_{\langle i,j \rangle} ( c_i^\dagger \mathbf{\sigma} \times
        \mathbf{\hat{d}}_{\langle i,j \rangle} c_j + \text{h.c.})

    Parameters
    ----------
    onsite : float
        On-site energy.
    t : float, complex
        Hopping parameter.
    soc : float, complex
        Spin-orbit coupling strength.
    rashba : float, complex
        Rashba coupling strength.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    The Kane-Mele model describes a two-dimensional topological insulator with spin-orbit coupling.
    It is defined on a honeycomb lattice and includes both intrinsic and Rashba spin-orbit coupling [kane-mele]_.

    References
    ----------
    .. [kane-mele] Kane, C. L., & Mele, E. J. (2005). Quantum Spin Hall Effect in Graphene. *Physical Review Letters*, 95(22), 226801.
    """

    # define lattice vectors
    lat_vecs = [[1, 0], [1 / 2, np.sqrt(3) / 2]]
    # define coordinates of orbitals
    orb_vecs = [[1 / 3, 1 / 3], [2 / 3, 2 / 3]]

    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1])

    # make two dimensional tight-binding Kane-Mele model
    ret_model = TBModel(lattice=lat, spinful=True)

    # set on-site energies
    ret_model.set_onsite([delta, -delta])

    # useful definitions
    sigma_x = np.array([0, 1, 0, 0])
    sigma_y = np.array([0, 0, 1, 0])
    sigma_z = np.array([0, 0, 0, 1])

    # set hoppings (one for each connected pair of orbitals)
    # (amplitude, i, j, [lattice vector to cell containing j])

    # spin-independent first-neighbor hoppings
    ret_model.set_hop(t, 0, 1, [0, 0])
    ret_model.set_hop(t, 0, 1, [0, -1])
    ret_model.set_hop(t, 0, 1, [-1, 0])

    # second-neighbour spin-orbit hoppings (s_z)
    nnn_hop = 1j * soc * sigma_z
    ret_model.set_hop(nnn_hop, 0, 0, [0, 1])
    ret_model.set_hop(-nnn_hop, 0, 0, [1, 0])
    ret_model.set_hop(nnn_hop, 0, 0, [1, -1])
    ret_model.set_hop(-nnn_hop, 1, 1, [0, 1])
    ret_model.set_hop(nnn_hop, 1, 1, [1, 0])
    ret_model.set_hop(-nnn_hop, 1, 1, [1, -1])

    # Rashba first-neighbor hoppings: (s_x)(dy)-(s_y)(d_x)

    # bond unit vectors are (np.sqrt(3) / 2, 1/2) then (0,-1) then (-np.sqrt(3) / 2, 1/2)
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x - (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [0, 0],
        mode='add',
    )
    ret_model.set_hop(1j * rashba * -sigma_x, 0, 1, [0, -1], mode='add')
    ret_model.set_hop(
        1j * rashba * ((1 / 2) * sigma_x + (np.sqrt(3) / 2) * sigma_y),
        0,
        1,
        [-1, 0],
        mode='add',
    )

    return ret_model


def fu_kane_mele_pythtb(t, soc, dt=[0, 0, 0, 0]):
    from pythtb import Lattice, TBModel

    r"""Fu-Kane-Mele tight-binding model.

    This function creates a Fu-Kane-Mele tight-binding model on a diamond
    lattice. The lattice vectors are given by,

    .. math::

        \mathbf{a}_1 = (0, 1, 1), \quad \mathbf{a}_2 = (1, 0, 1),
        \quad \mathbf{a}_3 = (1, 1, 0)

    and the orbital positions are given by,

    .. math::

        \mathbf{\tau}_1 = (0, 0, 0),
        \quad \mathbf{\tau}_2 = \frac{1}{4} \mathbf{a}_1 + \frac{1}{4} \mathbf{a}_2
        + \frac{1}{4} \mathbf{a}_3

    The second-quantized Hamiltonian can be written as:

    .. math::

        H = t \sum_{\langle ij \rangle} c_i^{\dagger} c_j
        + i \lambda_{SO} \sum_{\langle\langle ij \rangle\rangle} c_i^{\dagger}
        \vec{\sigma} \cdot (\mathbf{d}_{ij}^{1} \times \mathbf{d}_{ij}^{2}) c_j

    where the first term is a nearest-neighbor hopping term connecting the two fcc sublattices
    of the diamond lattice, and the second term is a spin-orbit coupling term connecting
    second-neighbor sites within the same sublattice. Here, :math:`\mathbf{d}_{ij}^{1,2}`
    are the two nearest-neighbor bond vectors connecting sites :math:`i` and :math:`j`.

    Due to inversion symmetry, each band is doubly degenerate. The degeneracy is lifted by symmetry
    lowering perturbations of the four nearest-neighbor hoppings :math:`t \rightarrow t + \delta t_p`
    with :math:`p = 1, 2, 3, 4` indexing the four bonds connected to each site.

    .. versionadded:: 2.0.0

    Parameters
    ----------
    t : float
        Spin-independent nearest-neighbor hopping amplitude.
    soc : float
        Spin-orbit coupling strength. Modulates next-nearest neighbor
        hopping amplitudes.
    dt : list[float, float, float, float], optional
        Offsets added to the four nearest-neighbor hoppings along the
        bonds connected to each site. The entries are applied in the
        following order:

        - `dt[0]` : bond along ``R = [0, 0, 0]``
        - `dt[1]` : bond along ``R = [-1, 0, 0]``
        - `dt[2]` : bond along ``R = [0, -1, 0]``
        - `dt[3]` : bond along ``R = [0, 0, -1]``

        The default is ``[0, 0, 0, 0]``, which corresponds to uniform
        hopping amplitudes. This parameter allows for symmetry-lowering
        perturbations to the nearest-neighbor hoppings.

    Returns
    -------
    TBModel
        An instance of the model.

    Notes
    -----
    - The Fu-Kane-Mele model describes a three-dimensional topological insulator with a
      non-trivial band structure. It is characterized by a strong :math:`\mathbb{Z}_2` invariant
      and exhibits surface Dirac cones that are protected by time-reversal and inversion
      symmetry [1]_.

    References
    ----------
    .. [1] \ L. Fu, C. L. Kane, and E. J. Mele, *Phys. Rev. Lett.*, **98**, 106803
        (2007).
    """

    lat_vecs = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    orb_vecs = [[0, 0, 0], [0.25, 0.25, 0.25]]
    lat = Lattice(lat_vecs, orb_vecs, periodic_dirs=[0, 1, 2])

    model = TBModel(lattice=lat, spinful=True)

    # spin-independent first-neighbor hops
    for idx, lvec in enumerate([[0, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
        model.set_hop(t + dt[idx], 0, 1, lvec)

    # spin-dependent second-neighbor hops
    lvec_list = ([1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 1, 0], [0, -1, 1], [1, 0, -1])
    dir_list = ([0, 1, -1], [-1, 0, 1], [1, -1, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1])
    for j in range(6):
        spin = np.array([0.0] + dir_list[j])
        model.set_hop(1j * soc * spin, 0, 0, lvec_list[j])
        model.set_hop(-1j * soc * spin, 1, 1, lvec_list[j])

    return model


# Predefined hard-coded models


def graphene(data_controller, params):
    from ..utils.constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 1

    attr['nawf'] = 2
    attr['nspin'] = 1
    attr['natoms'] = 2

    arry['naw'] = np.array([1, 1])

    attr['alat'] = 2.46 * ANGSTROM_AU

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H00
    arry['HRs'][0, 0, 0, 0, 0, 0] = params['delta'] / 2
    arry['HRs'][1, 1, 0, 0, 0, 0] = -params['delta'] / 2

    # H00
    arry['HRs'][0, 1, 0, 0, 0, 0] = params['t']
    arry['HRs'][1, 0, 0, 0, 0, 0] = params['t']

    # H10
    arry['HRs'][1, 0, 1, 0, 0, 0] = params['t']

    # H20
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H01
    arry['HRs'][1, 0, 0, 1, 0, 0] = params['t']

    # H02
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry['a_vectors'] = arry['a_vectors']

    # Atomic coordinates
    arry['tau'] = np.zeros((2, 3), dtype=float)

    arry['tau'][0, 0] = 0.50000
    arry['tau'][0, 1] = 0.28867
    arry['tau'][1, 0] = 1.00000
    arry['tau'][1, 1] = 0.57735

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['C', 'C']


def cubium(data_controller, params):
    from ..utils.constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 3
    attr['Efermi'] = 6 * params['t']
    attr['nawf'] = 1
    attr['nspin'] = 1
    attr['natoms'] = 1
    attr['bnd'] = 1
    attr['shift'] = 0
    attr['dftSO'] = False
    attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
    attr['nbnds'] = 1
    attr['nelec'] = 2

    attr['alat'] = 1.0 * ANGSTROM_AU
    attr['omega'] = attr['alat'] ** 3

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H000
    arry['HRs'][0, 0, 0, 0, 0, 0] = 0.0 - attr['Efermi']

    # H100
    arry['HRs'][0, 0, 1, 0, 0, 0] = params['t']

    # H200
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H010
    arry['HRs'][0, 0, 0, 1, 0, 0] = params['t']

    # H020
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # H001
    arry['HRs'][0, 0, 0, 0, 1, 0] = params['t']

    # H002
    arry['HRs'][:, :, 0, 0, 2, 0] = np.conj(arry['HRs'][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry['tau'] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['Cu']


def cubium2(data_controller, params):
    from ..utils.constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 3

    attr['nawf'] = 2
    attr['nspin'] = 1
    attr['natoms'] = 1
    attr['bnd'] = 2
    attr['shift'] = 0
    attr['dftSO'] = False
    attr['nkpnts'] = attr['nk1'] * attr['nk2'] * attr['nk3']
    attr['nbnds'] = 2
    attr['nelec'] = 2
    attr['alat'] = 1.0 * ANGSTROM_AU
    attr['omega'] = attr['alat'] ** 3

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H000
    arry['HRs'][0, 0, 0, 0, 0, 0] = -params['Eg'] / 2 - 6.0 * params['t']
    arry['HRs'][1, 1, 0, 0, 0, 0] = params['Eg'] / 2 + 6.0 * params['t']

    # H100
    arry['HRs'][0, 0, 1, 0, 0, 0] = params['t']
    arry['HRs'][1, 1, 1, 0, 0, 0] = -params['t']

    # H200
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T

    # H010
    arry['HRs'][0, 0, 0, 1, 0, 0] = params['t']
    arry['HRs'][1, 1, 0, 1, 0, 0] = -params['t']

    # H020
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T

    # H001
    arry['HRs'][0, 0, 0, 0, 1, 0] = params['t']
    arry['HRs'][1, 1, 0, 0, 1, 0] = -params['t']

    # H002
    arry['HRs'][:, :, 0, 0, 2, 0] = np.conj(arry['HRs'][:, :, 0, 0, 1, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    # Atomic coordinates
    arry['tau'] = np.zeros((1, 3), dtype=float)

    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    arry['atoms'] = ['Cu']


def Kane_Mele(data_controller, params):
    from ..utils.constants import ANGSTROM_AU

    arry, attr = data_controller.data_dicts()

    attr['nk1'] = 3
    attr['nk2'] = 3
    attr['nk3'] = 1

    attr['nawf'] = 4
    attr['bnd'] = 4
    attr['shift'] = np.inf
    attr['Efermi'] = 0.0
    attr['nspin'] = 1
    attr['natoms'] = 2

    arry['naw'] = [2, 2]

    if 'alat' not in params:
        alat = 1.0
    else:
        alat = params['alat']

    attr['alat'] = alat * ANGSTROM_AU

    t = params['t']
    soc_par = params['soc_par']
    if soc_par > 0.0:
        attr['dftSO'] = True
    r_par = params['r_par']
    v_par = params['v_par']

    arry['HRs'] = np.zeros(
        (
            attr['nawf'],
            attr['nawf'],
            attr['nk1'],
            attr['nk2'],
            attr['nk3'],
            attr['nspin'],
        ),
        dtype=complex,
    )

    # H00
    arry['HRs'][0, 0, 0, 0, 0, 0] = t * v_par
    arry['HRs'][1, 1, 0, 0, 0, 0] = t * v_par
    arry['HRs'][2, 2, 0, 0, 0, 0] = -t * v_par
    arry['HRs'][3, 3, 0, 0, 0, 0] = -t * v_par

    # H00
    arry['HRs'][0, 2, 0, 0, 0, 0] = t
    arry['HRs'][1, 3, 0, 0, 0, 0] = t
    arry['HRs'][2, 0, 0, 0, 0, 0] = t
    arry['HRs'][3, 1, 0, 0, 0, 0] = t

    # H10
    arry['HRs'][2, 0, 1, 0, 0, 0] = t
    arry['HRs'][3, 1, 1, 0, 0, 0] = t

    arry['HRs'][0, 0, 1, 0, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][1, 1, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][2, 2, 1, 0, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][3, 3, 1, 0, 0, 0] = -complex(0.0, soc_par)

    ##H20
    # arry['HRs'][:,:,2,0,0,0] = np.conj(arry['HRs'][:,:,1,0,0,0]).T

    # H01
    arry['HRs'][2, 0, 0, 1, 0, 0] = t
    arry['HRs'][3, 1, 0, 1, 0, 0] = t

    arry['HRs'][0, 0, 0, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][1, 1, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][2, 2, 0, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][3, 3, 0, 1, 0, 0] = complex(0.0, soc_par)

    ##H02
    # arry['HRs'][:,:,0,2,0,0] = np.conj(arry['HRs'][:,:,0,1,0,0]).T

    # H21
    arry['HRs'][0, 0, 2, 1, 0, 0] = -complex(0.0, soc_par)
    arry['HRs'][1, 1, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][2, 2, 2, 1, 0, 0] = complex(0.0, soc_par)
    arry['HRs'][3, 3, 2, 1, 0, 0] = -complex(0.0, soc_par)

    ##H12
    ##arry['HRs'][:,:,1,2,0,0] = np.conj(arry['HRs'][:,:,2,1,0,0]).T

    r3h = np.sqrt(3.0) / 2.0

    arry['HRs'][0, 3, 0, 0, 0, 0] += r_par * complex(
        -r3h, 0.5
    )  # 1j * r_par * (0.5 * 1 - r3h * -1j)
    arry['HRs'][1, 2, 0, 0, 0, 0] += r_par * complex(r3h, 0.5)  # 1j * r_par * (0.5 * 1 - r3h * 1j)
    arry['HRs'][3, 0, 0, 0, 0, 0] += r_par * complex(-r3h, -0.5)
    arry['HRs'][2, 1, 0, 0, 0, 0] += r_par * complex(r3h, -0.5)

    arry['HRs'][0, 3, 1, 0, 0, 0] += -r_par * complex(
        r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * -1j)
    arry['HRs'][1, 2, 1, 0, 0, 0] += -r_par * complex(
        -r3h, 0.5
    )  # -1j * r_par * (0.5 * 1 + r3h * 1j)

    arry['HRs'][0, 3, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1
    arry['HRs'][1, 2, 0, 1, 0, 0] += complex(0.0, r_par)  # -1j * r_par * -1 * 1

    # H02
    arry['HRs'][:, :, 0, 2, 0, 0] = np.conj(arry['HRs'][:, :, 0, 1, 0, 0]).T
    # H20
    arry['HRs'][:, :, 2, 0, 0, 0] = np.conj(arry['HRs'][:, :, 1, 0, 0, 0]).T
    # H12
    arry['HRs'][:, :, 1, 2, 0, 0] = np.conj(arry['HRs'][:, :, 2, 1, 0, 0]).T

    # Lattice Vectors
    arry['a_vectors'] = np.zeros((3, 3), dtype=float)
    arry['a_vectors'] = np.array([[1.0, 0, 0], [0.5, 3**0.5 / 2, 0], [0, 0, 10]])
    arry['a_vectors'] = arry['a_vectors']

    # Spin properties
    arry['Sj'] = np.zeros((3, 4, 4), dtype=complex)

    arry['Sj'][2, 0, 0] = 0.5
    arry['Sj'][2, 1, 1] = -0.5
    arry['Sj'][2, 2, 2] = 0.5
    arry['Sj'][2, 3, 3] = -0.5

    arry['Sj'][0, 0, 1] = 0.5
    arry['Sj'][0, 1, 0] = 0.5
    arry['Sj'][0, 2, 3] = 0.5
    arry['Sj'][0, 3, 2] = 0.5

    arry['Sj'][1, 0, 1] = -complex(0.0, 0.5)
    arry['Sj'][1, 1, 0] = complex(0.0, 0.5)
    arry['Sj'][1, 2, 3] = -complex(0.0, 0.5)
    arry['Sj'][1, 3, 2] = complex(0.0, 0.5)
    # Atomic coordinates
    arry['tau'] = np.zeros((2, 3), dtype=float)
    arry['tau'][0] = np.dot([1 / 3, 1 / 3, 0.0], arry['a_vectors'])
    arry['tau'][1] = np.dot([2 / 3, 2 / 3, 0.0], arry['a_vectors'])

    arry['Dnm'] = np.zeros((4, 4, 3), dtype=float)
    # Reciprocal Lattice
    arry['b_vectors'] = np.zeros((3, 3), dtype=float)
    volume = np.dot(
        np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :]),
        arry['a_vectors'][2, :],
    )
    arry['b_vectors'][0, :] = (np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])) / volume
    arry['b_vectors'][1, :] = (np.cross(arry['a_vectors'][2, :], arry['a_vectors'][0, :])) / volume
    arry['b_vectors'][2, :] = (np.cross(arry['a_vectors'][0, :], arry['a_vectors'][1, :])) / volume

    attr['omega'] = alat**3 * arry['a_vectors'][0, :].dot(
        np.cross(arry['a_vectors'][1, :], arry['a_vectors'][2, :])
    )

    arry['species'] = ['KM', 'KM']


def build_from_pythTB(data_controller, my_model):
    # Basis ( 1up, 2up, 3up ... | 1dn, 2dn, 3dn ... )
    from scipy import fftpack as FFT

    arry, attr = data_controller.data_dicts()
    Lattice = my_model._lattice
    hoptable = my_model._hoptable
    site_energy = my_model._site_energies  # shape = norb*2*2
    ndim = int(hoptable.dim_r)
    norb = int(Lattice._orb_vecs_cart.shape[0])

    spinful = my_model._spinful
    attr['dftSO'] = spinful
    attr['adhoc_SO'] = spinful
    attr['do_spin_orbit'] = spinful
    arry['naw'] = [int(spinful) + 1] * norb  # Number of orbitals on each atom
    attr['norb'] = norb * 2 if spinful else norb
    attr['nawf'] = attr['bnd'] = attr['nbnds'] = attr['natoms'] = attr['norb']
    attr['nspin'] = 1
    attr['Efermi'] = 0.0
    attr['shift'] = np.inf
    attr['alat'] = 1.0
    attr['omega'] = Lattice._cell_vol

    arry['a_vectors'] = np.eye(3, 3)
    arry['a_vectors'][:ndim, :ndim] = Lattice._lat_vectors
    arry['b_vectors'] = np.eye(3, 3)
    arry['b_vectors'][:ndim, :ndim] = Lattice._recip_lat
    arry['b_vectors'] /= 2 * np.pi
    tau = arry['tau'] = np.zeros((norb, 3))
    arry['tau'][:, :ndim] = Lattice._orb_vecs_cart

    from_idx = hoptable.from_idx
    to_idx = hoptable.to_idx
    dR = hoptable.lattice_vecs
    dR3 = np.zeros((dR.shape[0], 3), dtype=int)
    dR3[:, :ndim] = dR
    hopping = hoptable.amplitudes
    nks = np.max(dR3, axis=0) * 2 + 1
    attr['nk1'] = nks[0]
    attr['nk2'] = nks[1]
    attr['nk3'] = nks[2]
    attr['nkpnts'] = nks[0] * nks[1] * nks[2]

    HRs = np.zeros((attr['norb'], attr['norb'], nks[0], nks[1], nks[2], 1), dtype=complex)
    if spinful:
        for i, t in enumerate(hopping):
            t_conj = np.conjugate(t)
            HRs[from_idx[i], to_idx[i], dR3[i, 0], dR3[i, 1], dR3[i, 2], 0] = t[0, 0]
            HRs[from_idx[i] + norb, to_idx[i], dR3[i, 0], dR3[i, 1], dR3[i, 2], 0] = t[1, 0]
            HRs[from_idx[i], to_idx[i] + norb, dR3[i, 0], dR3[i, 1], dR3[i, 2], 0] = t[0, 1]
            HRs[from_idx[i] + norb, to_idx[i] + norb, dR3[i, 0], dR3[i, 1], dR3[i, 2], 0] = t[1, 1]
            HRs[to_idx[i], from_idx[i], -dR3[i, 0], -dR3[i, 1], -dR3[i, 2], 0] = t_conj[0, 0]
            HRs[to_idx[i], from_idx[i] + norb, -dR3[i, 0], -dR3[i, 1], -dR3[i, 2], 0] = t_conj[1, 0]
            HRs[to_idx[i] + norb, from_idx[i], -dR3[i, 0], -dR3[i, 1], -dR3[i, 2], 0] = t_conj[0, 1]
            HRs[
                to_idx[i] + norb,
                from_idx[i] + norb,
                -dR3[i, 0],
                -dR3[i, 1],
                -dR3[i, 2],
                0,
            ] = t_conj[1, 1]
        for i, e in enumerate(site_energy):
            HRs[i, i, 0, 0, 0, 0] = e[0, 0]
            HRs[i + norb, i, 0, 0, 0, 0] = e[1, 0]
            HRs[i, i + norb, 0, 0, 0, 0] = e[0, 1]
            HRs[i + norb, i + norb, 0, 0, 0, 0] = e[1, 1]
    else:
        for i, t in enumerate(hopping):
            HRs[from_idx[i], to_idx[i], dR3[i, 0], dR3[i, 1], dR3[i, 2], 0] = t
            HRs[to_idx[i], from_idx[i], -dR3[i, 0], -dR3[i, 1], -dR3[i, 2], 0] = np.conjugate(t)
        for i, e in enumerate(site_energy):
            HRs[i, i, 0, 0, 0, 0] = e

    arry['HRs'] = HRs
    arry['Hks'] = np.zeros_like(arry['HRs'])
    arry['Hks'] = FFT.fftn(arry['HRs'], axes=[2, 3, 4])

    temp = np.zeros((norb, norb, 3))
    for n in range(norb):
        for m in range(norb):
            temp[n, m, :] = tau[n, :] - tau[m, :]
    if spinful:
        Dnm = np.zeros((2 * norb, 2 * norb, 3))
        for i in range(3):
            Dnm[:, :, i] = np.matlib.repmat(temp[:, :, i], 2, 2)
    else:
        Dnm = temp
    arry['Dnm'] = Dnm


def predefined_models(data_controller, params):
    model_name = params['label']
    match model_name.upper():
        case 'CUBIUM_PYTHTB':
            my_model = cubium_pythtb(params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'CUBIUM2_PYTHTB':
            my_model = cubium2_pythtb(params['t'], params['Eg'])
            build_from_pythTB(data_controller, my_model)
        case 'SSH_PYTHTB':
            my_model = ssh_pythtb(params['v'], params['w'])
            build_from_pythTB(data_controller, my_model)
        case 'CHECKERBOARD_PYTHTB':
            my_model = checkerboard_pythtb(params['delta'], params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'GRAPHENE_PYTHTB':
            my_model = graphene_pythtb(params['delta'], params['t'])
            build_from_pythTB(data_controller, my_model)
        case 'HALDANE_PYTHTB':
            my_model = haldane_pythtb(params['delta'], params['t1'], params['t2'], params['phi'])
            build_from_pythTB(data_controller, my_model)
        case 'KANE_MELE_PYTHTB':
            my_model = kane_mele_pythtb(
                params['delta'], params['t'], params['soc'], params['rashba']
            )
            build_from_pythTB(data_controller, my_model)
        case 'FU_KANE_MELE_PYTHTB':
            my_model = fu_kane_mele_pythtb(params['t'], params['soc'], params['dt'])
            build_from_pythTB(data_controller, my_model)
        case 'GRAPHENE':
            graphene(data_controller, params)
        case 'CUBIUM':
            cubium(data_controller, params)
        case 'CUBIUM2':
            cubium2(data_controller, params)
        case 'KANE_MELE':
            Kane_Mele(data_controller, params)
        case _:
            print('Not a predefined model.')
            import sys

            sys.exit()
