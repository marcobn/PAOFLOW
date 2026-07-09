"""Electron-phonon coupling for PAOFLOW (Stage S7).

This subpackage builds the electron-phonon matrix elements

    g_mn^v(k, q) = <psi_{m, k+q} | d_{qv} V | psi_{n, k}>

by **finite-differencing the PAO Hamiltonian** with respect to atomic
displacements (a frozen-phonon approach that reuses the projection ->
``pao_hamiltonian`` -> ``HRs`` pipeline and the phonon module's supercell /
eigenvector machinery).

The workflow is staged:

* **P0 (this module set):** the package scaffold, the reference-cell
  displacement bookkeeping (:mod:`displacements`) and the two-phase QE input
  generation (:mod:`io`, :mod:`do_elphon`).
* **P1+:** rebuild the PAO Hamiltonian of every displaced supercell, central
  finite-difference ``dV_{kappa,alpha} = dH/du_{kappa,alpha}``, assemble
  ``g_mn^v(k, q)`` and the derived properties (Eliashberg ``alpha^2 F`` /
  ``lambda``, phonon-limited transport, ...).

Public entry point: :meth:`PAOFLOW.PAOFLOW.electron_phonon`.
"""

from .do_ao_eph import eliashberg_from_qe_coupling, vertex_from_qe_elphmat
from .eph_kq import eliashberg, eliashberg_from_modes
from .qe_elph_io import (
    el_ph_mat_to_cartesian,
    lambda_from_gamma,
    load_qe_coupling,
    read_elph_inp_lambda,
    read_lambda_in,
    read_qe_dyn,
    read_qe_el_ph_mat,
)
from .qe_matdyn import (
    interpolate_coupling,
    read_a2f_ifc,
    read_qe_ifc,
)

__all__ = [
    'eliashberg',
    'eliashberg_from_modes',
    'eliashberg_from_qe_coupling',
    'el_ph_mat_to_cartesian',
    'interpolate_coupling',
    'lambda_from_gamma',
    'load_qe_coupling',
    'read_a2f_ifc',
    'read_elph_inp_lambda',
    'read_lambda_in',
    'read_qe_dyn',
    'read_qe_el_ph_mat',
    'read_qe_ifc',
    'vertex_from_qe_elphmat',
]
