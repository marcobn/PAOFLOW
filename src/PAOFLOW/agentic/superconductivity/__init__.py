"""Superconductivity of PAOFLOW (Eliashberg function, Tc, gaps) vs experimental data of any form.

Serves the repository workflow ``agentic/superconductivity/`` (playbook, lessons,
prompts, templates). The data toolkit is shared with the thermo-optics workflow
(:mod:`PAOFLOW.agentic.thermo_optics`: ``datasets``, ``units``, ``ingest``,
``digitize``, ``metrics``, ``plotting``); this package holds the adapters and
baselines:

* :mod:`~PAOFLOW.agentic.superconductivity.eliashberg` --
  :class:`~PAOFLOW.agentic.superconductivity.eliashberg.EliashbergModel`:
  :math:`\\alpha^2F(\\omega)` (and :math:`\\alpha^2F_{tr}`), :math:`\\lambda`,
  :math:`\\omega_{\\log}`, :math:`\\omega_2`, McMillan / Allen-Dynes Tc for any
  :math:`\\mu^*`, the cumulative and windowed coupling, the mode-resolved
  :math:`\\lambda_{\\mathbf{q}\\nu}` of ``superconductivity/output/eliashberg.npz``;
* :mod:`~PAOFLOW.agentic.superconductivity.gap` --
  :class:`~PAOFLOW.agentic.superconductivity.gap.GapModel` (isotropic
  Migdal-Eliashberg: :math:`\\Delta(T)`, gap edges, :math:`Z`, Tc, the gap
  ratio, the quasiparticle DOS) and
  :class:`~PAOFLOW.agentic.superconductivity.gap.AnisotropicGapModel`
  (:math:`\\Delta_{n\\mathbf{k}}(T)`, gap distributions and their peaks, sheet
  averages, :math:`\\lambda_{n\\mathbf{k}}`);
* :mod:`~PAOFLOW.agentic.superconductivity.baselines` -- the moments of
  :math:`\\alpha^2F`, McMillan and Allen-Dynes Tc and their inversion, the BCS
  gap and its universal :math:`\\Delta(T)`, the strong-coupling gap ratio.

The adapters open an electron-phonon system folder (``examples/elphon/<System>``
layout) with ``from_system(folder, output=...)`` and record its ``system.py`` for
``describe()``.
"""

from .eliashberg import EliashbergModel
from .gap import AnisotropicGapModel, GapModel

__all__ = ['AnisotropicGapModel', 'EliashbergModel', 'GapModel']
