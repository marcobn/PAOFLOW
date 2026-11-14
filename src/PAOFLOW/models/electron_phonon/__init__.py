"""Model crystals for electron-phonon coupling and their properties with PAOFLOW's engines.

Tight-binding electrons whose Slater-Koster hoppings depend on the bond length, harmonic phonons
from springs between the same atoms, and so the electron-phonon vertex of the same crystal
(:class:`ElectronPhononModel`).  The model crystals of the electron-phonon lecture notes
(:mod:`.crystals`), the coupling and the properties on uniform grids (:mod:`.properties`), and the
runs that write PAOFLOW's output files (:mod:`.workflows`, the ``ep_model`` workflow of
``paoflow-gen`` and ``paoflow-studio``).

Example
-------
>>> from PAOFLOW.models.electron_phonon import cubic_metal_model, workflows
>>> model = cubic_metal_model(eta=6.0)
>>> out = workflows.superconductivity(model, 'output', nk=40, nq=8)  # doctest: +SKIP
>>> workflows.migdal_eliashberg('output')  # doctest: +SKIP
"""

from . import properties, workflows
from .crystals import cubic_metal_model, indirect_semiconductor_model, light_atom_metal_model
from .model import ElectronPhononModel, slater_koster_hopping

#: The model crystals by name: (function, metal).
MODEL_CRYSTALS = {
    'indirect_semiconductor': (indirect_semiconductor_model, False),
    'cubic_metal': (cubic_metal_model, True),
    'light_atom_metal': (light_atom_metal_model, True),
}

__all__ = [
    'ElectronPhononModel',
    'MODEL_CRYSTALS',
    'cubic_metal_model',
    'indirect_semiconductor_model',
    'light_atom_metal_model',
    'properties',
    'slater_koster_hopping',
    'workflows',
]
