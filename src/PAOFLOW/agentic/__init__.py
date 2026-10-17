"""Library code of the agentic workflows.

Each subpackage serves the workflow of the same name in the repository folder
``agentic/`` (playbook, lessons learned, prompts, templates; not installed):

* :mod:`PAOFLOW.agentic.qo_analysis` -- ``agentic/skeaf/``: pyskeaf / SKEAF
  extremal orbits vs measured quantum oscillations
  (:class:`~PAOFLOW.agentic.qo_analysis.QuantumOscillationAnalysis`);
* :mod:`PAOFLOW.agentic.thermo_optics` -- ``agentic/thermo_optics/``:
  temperature-dependent optical response (absorption, refractive index,
  emissivity, free carriers) vs experimental data of any form.

The modules are ordinary library code and can be used without an agent.
"""
