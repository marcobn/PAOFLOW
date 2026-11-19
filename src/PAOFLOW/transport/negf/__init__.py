"""Finite-bias transport: non-equilibrium density, electrostatics and the self-consistent potential.

Modules:
- :mod:`~PAOFLOW.transport.negf.contour`: the equilibrium contour (with the Fermi poles) and the bias window;
- :mod:`~PAOFLOW.transport.negf.density_matrix`: orbital occupations and T(E) of the open conductor under bias;
- :mod:`~PAOFLOW.transport.negf.electrostatics`: the :class:`ElectrostaticModel` interface of the loop, and the
  planar layers with the planar Poisson solution between the electrodes;
- :mod:`~PAOFLOW.transport.negf.atomic_electrostatics`: one Gaussian charge per atom, three-dimensional Poisson
  equation (molecules, wires);
- :mod:`~PAOFLOW.transport.negf.poisson_3d`: Gaussian charges between grounded planes, periodic across the
  transverse cell (plane waves times sines);
- :mod:`~PAOFLOW.transport.negf.charge_model`: the Gaussian widths from the pseudo-atomic valence densities;
- :mod:`~PAOFLOW.transport.negf.self_consistency`: the bias scan (linear ramp or self-consistent), Anderson mixing.

Entry point: :meth:`PAOFLOW.Transport.Transport.compute_current_self_consistent`.
"""
