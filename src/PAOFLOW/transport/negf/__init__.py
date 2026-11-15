"""Finite-bias transport: non-equilibrium density, planar electrostatics and the self-consistent potential.

Modules:
- :mod:`~PAOFLOW.transport.negf.contour`: the equilibrium contour (with the Fermi poles) and the bias window;
- :mod:`~PAOFLOW.transport.negf.density_matrix`: orbital occupations and T(E) of the open conductor under bias;
- :mod:`~PAOFLOW.transport.negf.electrostatics`: layers and the planar Poisson solution between the electrodes;
- :mod:`~PAOFLOW.transport.negf.self_consistency`: the bias scan (linear ramp or self-consistent), Anderson mixing.

Entry point: :meth:`PAOFLOW.Transport.Transport.compute_current_self_consistent`.
"""
