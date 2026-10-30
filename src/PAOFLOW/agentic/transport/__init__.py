"""Transport and mobility of PAOFLOW vs experimental data of any form.

Serves the repository workflow ``agentic/transport/`` (playbook, lessons, prompts,
templates). The data toolkit is shared with the thermo-optics workflow
(:mod:`PAOFLOW.agentic.thermo_optics`: ``datasets``, ``units``, ``ingest``,
``digitize``, ``metrics``, ``plotting``); this package holds the transport
adapters and baselines:

* :mod:`~PAOFLOW.agentic.transport.mobility` --
  :class:`~PAOFLOW.agentic.transport.mobility.MobilityModel`: phonon-limited
  mobilities (SERTA / MRTA), carrier densities, conductivity and Seebeck
  coefficient of ``transport/<tag>/mobility.dat`` (semiconductors);
* :mod:`~PAOFLOW.agentic.transport.resistivity` --
  :class:`~PAOFLOW.agentic.transport.resistivity.ResistivityModel`: the
  resistivity per scattering channel, the extended-Drude dielectric function and
  the thermal emissivity of ``transport/<tag>/resistivity.dat`` /
  ``emissivity.npz`` (metals), with the Eliashberg scalars of the run;
* :mod:`~PAOFLOW.agentic.transport.baselines` -- Bloch-Gruneisen, power-law
  and linear fits, the Matthiessen residual offset, Matthiessen's mobilities;
* :mod:`~PAOFLOW.agentic.transport.channels` -- channel lists for
  ``PAOFLOW.transport(scattering_channels=...)`` from the run outputs (e-ph +
  residual + Fermi liquid for metals; energy-resolved e-ph + impurities for
  semiconductors).

Both adapters open an electron-phonon system folder (``examples/elphon/<System>``
layout) with ``from_system(folder, output=...)`` and record its ``system.py`` for
``describe()``.
"""

from .mobility import COLUMNS, MobilityModel
from .resistivity import ResistivityModel

__all__ = ['COLUMNS', 'MobilityModel', 'ResistivityModel']
