"""Temperature-dependent optical response of PAOFLOW vs experimental data of any form.

The workflow is system- and source-agnostic: experimental data arrive as tables,
free text, vector or scanned PDF figures, slide decks or images; they are turned
into one canonical long-format table with provenance
(:mod:`~PAOFLOW.agentic.thermo_optics.datasets`), the model is evaluated at the
measured conditions through an adapter (e.g.
:class:`~PAOFLOW.agentic.thermo_optics.paoflow_optics.OpticsModel`), and the two
are compared with explicit metrics and reference baselines
(:mod:`~PAOFLOW.agentic.thermo_optics.metrics`).

Modules:

* :mod:`~PAOFLOW.agentic.thermo_optics.datasets` -- canonical datasets with
  provenance (``measured``, ``transcribed``, ``vector-extracted``,
  ``digitised``, ``calculated``);
* :mod:`~PAOFLOW.agentic.thermo_optics.units` -- photon energy / wavelength /
  wavenumber and temperature conversions;
* :mod:`~PAOFLOW.agentic.thermo_optics.ingest` -- inventories of PDFs, page
  rendering, embedded images, words and vector paths, tables and slide decks;
* :mod:`~PAOFLOW.agentic.thermo_optics.digitize` -- axis calibration (linear or
  logarithmic), frame and tick detection, colour masks, marker centroids and
  curve tracing for raster figures;
* :mod:`~PAOFLOW.agentic.thermo_optics.metrics` -- interpolation, deviation
  statistics, rigid shifts, threshold crossings, Arrhenius fits, condition
  matching;
* :mod:`~PAOFLOW.agentic.thermo_optics.paoflow_optics` -- adapter for PAOFLOW's
  optical absorption and thermal-emissivity outputs;
* :mod:`~PAOFLOW.agentic.thermo_optics.plotting` -- publication style, a
  validated temperature ramp, vector + 600 dpi output.

The agent playbook, lessons learned, prompts and templates live outside the
package, in the repository folder ``agentic/thermo_optics/``.
"""

from .datasets import read_dataset, write_dataset

__all__ = ['read_dataset', 'write_dataset']
