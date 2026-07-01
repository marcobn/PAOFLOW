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
