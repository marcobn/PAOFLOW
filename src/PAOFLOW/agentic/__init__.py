"""Library code of the agentic workflows.

Each subpackage serves the workflow of the same name in the repository folder
``agentic/`` (playbook, lessons learned, prompts, templates; not installed):

* :mod:`PAOFLOW.agentic.qo_analysis` -- ``agentic/skeaf/``: pyskeaf / SKEAF
  extremal orbits vs measured quantum oscillations
  (:class:`~PAOFLOW.agentic.qo_analysis.QuantumOscillationAnalysis`);
* :mod:`PAOFLOW.agentic.thermo_optics` -- ``agentic/thermo_optics/``:
  temperature-dependent optical response (absorption, refractive index,
  emissivity, free carriers) vs experimental data of any form.
* :mod:`PAOFLOW.agentic.transport` -- ``agentic/transport/``: phonon-limited
  mobilities, carrier densities, conductivity and Seebeck of semiconductors;
  resistivity, infrared optics and emissivity of metals; the Boltzmann bridge
  with impurity and residual channels (adapters ``MobilityModel``,
  ``ResistivityModel``; ``baselines``; ``channels``).
* :mod:`PAOFLOW.agentic.superconductivity` -- ``agentic/superconductivity/``:
  the Eliashberg function and its moments, McMillan / Allen-Dynes /
  Migdal-Eliashberg Tc, gaps versus temperature, multi-gap distributions
  (adapters ``EliashbergModel``, ``GapModel``, ``AnisotropicGapModel``;
  ``baselines``).
* :mod:`PAOFLOW.agentic.arpes` -- ``agentic/arpes/``: simulated ARPES band
  maps, kinks, self-energies and phonon linewidths vs measured spectra and
  band-map images (``ArpesModel``, ``curves``, ``images``).
* :mod:`PAOFLOW.agentic.nlo` -- ``agentic/nlo/``: second-order optical
  responses (SHG, shift and injection currents) vs experiments and published
  calculations, with the point-group audit of the run (``NonlinearModel``,
  ``symmetry``).

The last four reuse the data toolkit of :mod:`PAOFLOW.agentic.thermo_optics`
(datasets with provenance, units, PDF/raster ingestion and digitising, metrics,
publication plots) and open electron-phonon system folders
(``examples/elphon/<System>`` layout) through ``from_system``.

The modules are ordinary library code and can be used without an agent.
"""
