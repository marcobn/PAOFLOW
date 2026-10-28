"""Non-linear optics of PAOFLOW (shift / injection currents, SHG) vs experiments and literature.

Serves the repository workflow ``agentic/nlo/`` (playbook, lessons, prompts,
templates). The data toolkit is shared with the thermo-optics workflow
(:mod:`PAOFLOW.agentic.thermo_optics`); this package holds

* :mod:`~PAOFLOW.agentic.nlo.model` --
  :class:`~PAOFLOW.agentic.nlo.model.NonlinearModel`: the adapter for the
  ``<kind>_<abc>.dat`` files of :meth:`PAOFLOW.nonlinear_optics` (spectra,
  tensors at a photon energy, static values, peaks and onsets, point-group and
  Kleinman checks, ``d = chi/2`` and sheet / effective-3D conversions);
* :mod:`~PAOFLOW.agentic.nlo.symmetry` -- point-group operations of the crystal
  classes from generators, tensor symmetrisation, forbidden components, the
  relations between allowed components, symmetry and Kleinman errors.
"""

from .model import NonlinearModel

__all__ = ['NonlinearModel']
