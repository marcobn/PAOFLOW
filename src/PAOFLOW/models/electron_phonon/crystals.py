"""Model crystals for electron-phonon coupling: semiconductors with an indirect gap and two metals.

Every function returns an :class:`~PAOFLOW.models.electron_phonon.model.ElectronPhononModel` with
tight-binding electrons whose hoppings scale with the bond length, springs between the same atoms
(ad-hoc force constants) and so the vertex :math:`\\partial H/\\partial u` of the same crystal.
They are the models of the electron-phonon lecture notes (PAOFLOW wiki, ``epc_tutorial``):

* :func:`indirect_semiconductor_model`: s + p semiconductor with 2, 3 or 4 bands (1D, 2D, 3D), a
  gap from Gamma to the zone edge and acoustic phonons: scattering rates, mobility and
  phonon-assisted absorption;
* :func:`cubic_metal_model`: one s band on the simple-cubic lattice: Eliashberg function, Tc,
  Migdal-Eliashberg gap, resistivity, self-energy;
* :func:`light_atom_metal_model`: a heavy and a light atom with strongly bond-dependent hoppings
  (an MgB2-like toy): optical phonons, a two-peak Eliashberg function, the isotope effect.
"""

from __future__ import annotations

import numpy as np

from .model import ElectronPhononModel


def indirect_semiconductor_model(
    n_bands: int = 2,
    indirect_gap: float = 1.0,
    t_c: float = -0.25,
    t_sigma: float = 0.4,
    t_pi: float = 0.1,
    kane_p: float = 1.0,
    alat: float = 10.0,
    mass: float = 28.0,
    KL: float = 1.2,
    KT: float = 0.35,
    eta: float = 2.0,
) -> ElectronPhononModel:
    """s + p semiconductor with an indirect gap and 2, 3 or 4 bands (1D chain, square, simple cubic).

    Parameters
    ----------
    n_bands : {2, 3, 4}, optional
        Number of bands: one s conduction band and ``n_bands - 1`` p valence bands; the lattice
        is a chain (2), a square lattice (3) or a simple-cubic one (4).
    indirect_gap : float, optional
        Indirect gap from the valence top at Gamma to the conduction bottom at the zone corner (eV).
    t_c : float, optional
        Hopping of the s conduction band (eV); negative puts its minimum at the zone corner.
    t_sigma : float, optional
        Hopping of the p valence bands along their axis, V_pp_sigma (eV).
    t_pi : float, optional
        Hopping of the p valence bands across their axis, V_pp_pi (eV).
    kane_p : float, optional
        Interband s-p coupling P (eV), V_sp_sigma = P / 2: the optical matrix element.
    alat : float, optional
        Lattice parameter (bohr).
    mass : float, optional
        Atomic mass (amu).
    KL : float, optional
        Longitudinal spring constant of the nearest neighbours (eV/bohr^2).
    KT : float, optional
        Transverse spring constant of the nearest neighbours (eV/bohr^2).
    eta : float, optional
        Exponent of the bond-length scaling of the hoppings, t ~ d^-eta (Harrison: 2).

    Returns
    -------
    ElectronPhononModel
        One atom per cell, nearest-neighbour hoppings and springs, ``2 (n_bands - 1)`` electrons.

    Raises
    ------
    ValueError
        If ``n_bands`` is not 2, 3 or 4.

    Notes
    -----
    The electrons are :func:`PAOFLOW.models.models.parabolic_model` without asymmetry written in
    Slater-Koster form (:math:`V_{ss\\sigma} = -t_c`, :math:`V_{sp\\sigma} = P/2`,
    :math:`V_{pp\\sigma} = t_\\sigma`, :math:`V_{pp\\pi} = t_\\pi`).  With :math:`t_c < 0` the
    conduction band has its minimum at X, M or R and the gap is indirect; the direct gap at
    Gamma is ``indirect_gap + 4 d |t_c|`` in ``d`` dimensions.  Every hopping scales as
    :math:`(d_0/d)^\\eta`, so a strain :math:`\\varepsilon` shifts the conduction band by the
    deformation potential :math:`\\Xi = 2\\eta V_{ss\\sigma}` per axis.
    """
    if int(n_bands) not in (2, 3, 4):
        raise ValueError('n_bands must be 2, 3 or 4')
    dimension = int(n_bands) - 1
    kinds = ['s', 'px', 'py', 'pz'][: int(n_bands)]
    direct_gap = indirect_gap + 4.0 * dimension * abs(t_c)
    onsite_s = direct_gap + 2.0 * dimension * t_c  # conduction band at Gamma: the direct gap
    onsite_p = -2.0 * (t_sigma + (dimension - 1) * t_pi)  # valence top at Gamma: zero
    integrals = {'sss': -t_c, 'sps': 0.5 * kane_p, 'pps': t_sigma, 'ppp': t_pi}
    lattice = np.diag([1.0 if axis < dimension else 10.0 for axis in range(3)])
    return ElectronPhononModel(
        name='indirect_%d' % int(n_bands),
        lattice=lattice,
        alat=alat,
        positions=np.zeros((1, 3)),
        masses=np.array([mass]),
        orbitals=[(0, kind) for kind in kinds],
        onsite=np.array([onsite_s] + [onsite_p] * dimension),
        bonds=[{'atoms': (0, 0), 'distance': alat, 'V': integrals, 'eta': eta}],
        springs=[{'atoms': (0, 0), 'distance': alat, 'KL': KL, 'KT': KT}],
        nelec=2.0 * dimension,
        dimension=dimension,
        neighbour_search=1,
    )


def cubic_metal_model(
    t: float = -1.0,
    t2: float = -0.15,
    alat: float = 6.0,
    mass: float = 100.0,
    eta: float = 6.0,
    nelec: float = 0.8,
    KL: float = 0.6,
    KT: float = 0.12,
    KL2: float = 0.15,
) -> ElectronPhononModel:
    """One s band on the simple-cubic lattice: a monatomic metal.

    Parameters
    ----------
    t : float, optional
        Nearest-neighbour hopping (eV).
    t2 : float, optional
        Second-neighbour hopping (eV).
    alat : float, optional
        Lattice parameter (bohr).
    mass : float, optional
        Atomic mass (amu).
    eta : float, optional
        Exponent of the bond-length scaling of both hoppings, t ~ d^-eta: the strength of the coupling.
    nelec : float, optional
        Electrons per atom (both spins, between 0 and 2).
    KL : float, optional
        Longitudinal spring constant of the nearest neighbours (eV/bohr^2).
    KT : float, optional
        Transverse spring constant of the nearest neighbours (eV/bohr^2).
    KL2 : float, optional
        Longitudinal spring constant of the second neighbours (eV/bohr^2).

    Returns
    -------
    ElectronPhononModel

    Raises
    ------
    ValueError
        If ``nelec`` is not between 0 and 2.

    Notes
    -----
    The coupling grows as :math:`g \\propto \\eta` and :math:`\\lambda \\propto \\eta^2` at fixed
    phonons; the defaults give :math:`\\lambda \\approx 0.76` and :math:`T_c \\approx 10` K.
    """
    if not 0.0 < nelec < 2.0:
        raise ValueError('nelec must be between 0 and 2 (one s band)')
    return ElectronPhononModel(
        name='cubic_metal',
        lattice=np.eye(3),
        alat=alat,
        positions=np.zeros((1, 3)),
        masses=np.array([mass]),
        orbitals=[(0, 's')],
        onsite=np.array([0.0]),
        bonds=[
            {'atoms': (0, 0), 'distance': alat, 'V': {'sss': t}, 'eta': eta},
            {'atoms': (0, 0), 'distance': alat * np.sqrt(2), 'V': {'sss': t2}, 'eta': eta},
        ],
        springs=[
            {'atoms': (0, 0), 'distance': alat, 'KL': KL, 'KT': KT},
            {'atoms': (0, 0), 'distance': alat * np.sqrt(2), 'KL': KL2, 'KT': 0.0},
        ],
        nelec=nelec,
        dimension=3,
        neighbour_search=1,
    )


def light_atom_metal_model(
    t_ab: float = -1.0,
    t_aa: float = -0.3,
    eps_b: float = 1.0,
    alat: float = 7.0,
    mass_a: float = 90.0,
    mass_b: float = 11.0,
    eta_ab: float = 8.0,
    eta_aa: float = 3.0,
    nelec: float = 1.3,
    K_ab: float = 1.0,
    KT_ab: float = 0.25,
    K_aa: float = 0.4,
    K_bb: float = 0.15,
    dim: int = 2,
) -> ElectronPhononModel:
    """A heavy and a light atom with one s orbital each: optical phonons that couple strongly.

    Parameters
    ----------
    t_ab : float, optional
        Hopping between the heavy atom A and the light atom B (eV).
    t_aa : float, optional
        Hopping between neighbouring A atoms (eV).
    eps_b : float, optional
        On-site energy of the B orbital, relative to A (eV).
    alat : float, optional
        Lattice parameter (bohr).
    mass_a : float, optional
        Mass of the heavy atom A (amu).
    mass_b : float, optional
        Mass of the light atom B (amu).
    eta_ab : float, optional
        Exponent of the bond-length scaling of the A-B hopping: the coupling of the B modes.
    eta_aa : float, optional
        Exponent of the bond-length scaling of the A-A hopping.
    nelec : float, optional
        Electrons per cell (both spins, between 0 and 4).
    K_ab : float, optional
        Longitudinal spring constant of the A-B bond (eV/bohr^2).
    KT_ab : float, optional
        Transverse spring constant of the A-B bond (eV/bohr^2).
    K_aa : float, optional
        Longitudinal spring constant between A atoms (eV/bohr^2); the transverse one is a tenth.
    K_bb : float, optional
        Longitudinal spring constant between B atoms (eV/bohr^2); the transverse one is a tenth.
    dim : {2, 3}, optional
        Dimension: the centred square lattice (2) or the CsCl structure (3).

    Returns
    -------
    ElectronPhononModel
        A at the origin, B at the centre of the cell.

    Raises
    ------
    ValueError
        If ``dim`` is not 2 or 3 or ``nelec`` is not between 0 and 4.

    Notes
    -----
    The light atom's optical modes (60-67 meV with the defaults) carry most of the coupling,
    so :math:`T_c` depends on :math:`M_B`: the partial isotope coefficient
    :math:`\\alpha_B = -d\\ln T_c/d\\ln M_B`.
    """
    if int(dim) not in (2, 3):
        raise ValueError('dim must be 2 or 3')
    if not 0.0 < nelec < 4.0:
        raise ValueError('nelec must be between 0 and 4 (two s bands)')
    dimension = int(dim)
    lattice = np.diag([1.0 if axis < dimension else 10.0 for axis in range(3)])
    centre = np.array([0.5 if axis < dimension else 0.0 for axis in range(3)])
    bond_ab = alat * np.sqrt(dimension) / 2
    return ElectronPhononModel(
        name='light_atom_metal_%dd' % dimension,
        lattice=lattice,
        alat=alat,
        positions=np.array([np.zeros(3), centre]),
        masses=np.array([mass_a, mass_b]),
        orbitals=[(0, 's'), (1, 's')],
        onsite=np.array([0.0, eps_b]),
        bonds=[
            {'atoms': (0, 1), 'distance': bond_ab, 'V': {'sss': t_ab}, 'eta': eta_ab},
            {'atoms': (0, 0), 'distance': alat, 'V': {'sss': t_aa}, 'eta': eta_aa},
        ],
        springs=[
            {'atoms': (0, 1), 'distance': bond_ab, 'KL': K_ab, 'KT': KT_ab},
            {'atoms': (0, 0), 'distance': alat, 'KL': K_aa, 'KT': 0.1 * K_aa},
            {'atoms': (1, 1), 'distance': alat, 'KL': K_bb, 'KT': 0.1 * K_bb},
        ],
        nelec=nelec,
        dimension=dimension,
        neighbour_search=1,
    )
