"""ARPES of PAOFLOW (band maps, kinks, phonon linewidths) vs experimental data of any form.

Serves the repository workflow ``agentic/arpes/`` (playbook, lessons, prompts,
templates). The data toolkit is shared with the thermo-optics workflow
(:mod:`PAOFLOW.agentic.thermo_optics`: ``datasets``, ``units``, ``ingest``,
``digitize``, ``metrics``, ``plotting``); this package holds the ARPES
analysis:

* :mod:`~PAOFLOW.agentic.arpes.curves` -- MDC / EDC peaks and widths,
  Lorentzian fits, the dressed dispersion, Fermi momentum and velocity, the
  self-energy from a dispersion against a bare band, kink energy and mass
  enhancement; plain arrays, so they apply to model and measured maps alike;
* :mod:`~PAOFLOW.agentic.arpes.model` --
  :class:`~PAOFLOW.agentic.arpes.model.ArpesModel`: the adapter for
  ``arpes/<output>/`` of an electron-phonon system folder (bare and dressed
  maps, ``sigma_<kind>_<T>K.dat``, ``self_energy_fan_migdal.npz``);
* :mod:`~PAOFLOW.agentic.arpes.images` -- a calibrated band-map image as an
  intensity map in the same layout.
"""

from .model import ArpesModel

__all__ = ['ArpesModel']
