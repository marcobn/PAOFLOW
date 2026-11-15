"""ARPES simulation: photoemission matrix elements and spectral functions from the PAO Hamiltonian.

The photoemission formalism (plane-wave final state, dipole matrix elements with per-orbital
radial integrals, free-electron :math:`k_z` from the photon energy and the inner potential,
polarisation and spin projections, resolution broadening) follows **chinook**,
R. P. Day, B. Zwartsenberg, I. S. Elfimov and A. Damascelli, *npj Quantum Materials* **4**, 54
(2019), https://github.com/qmlab-ubc-ca/chinook (MIT licence, Copyright (c) 2018 Ryan Patrick
Day).  The kinematics and the Slater-orbital screening rules are adapted from chinook (see
``THIRD_PARTY_NOTICES.md``); the matrix elements are re-formulated in the real-harmonic PAO basis
of PAOFLOW with its own Gaunt coefficients and radial functions, and the spectral function takes
a band- and k-resolved self-energy.  Please cite chinook when publishing results of this module.

Workflow: ``PAOFLOW(...)`` → ``projections`` → ``pao_hamiltonian`` → :class:`PhotoemissionBasis`
and :class:`BlochHamiltonian` from the data controller → :class:`ArpesExperiment` →
:func:`simulate` → :class:`IntensityMap`.
"""

from .basis import Orbital, PhotoemissionBasis
from .bloch import BlochHamiltonian
from .experiment import ArpesExperiment, IntensityMap, simulate
from .slab import build_slab
from .spectral import (
    ConstantSelfEnergy,
    GridSelfEnergy,
    fermi_function,
    kramers_kronig_self_energy,
    spectral_function,
)

__all__ = [
    'ArpesExperiment',
    'BlochHamiltonian',
    'ConstantSelfEnergy',
    'GridSelfEnergy',
    'IntensityMap',
    'Orbital',
    'PhotoemissionBasis',
    'build_slab',
    'fermi_function',
    'kramers_kronig_self_energy',
    'simulate',
    'spectral_function',
]
