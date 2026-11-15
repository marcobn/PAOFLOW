"""Workflow layer of the electron-phonon examples and of ``paoflow-gen elphon``.

One :class:`ElphonSystem` per crystal (``system.py`` of the system folder) and one
physics-named function per property, called by the thin drivers of the workflow folders
(``phonons/``, ``coupling/``, ``superconductivity/``, ``optics/``, ``transport/``,
``arpes/``).  See ``docs/internals/Electron-Phonon-Coupling.md`` and
``examples/elphon/README.md``.
"""

from .coupling import long_range_path, quadrupole_inputs, quadrupole_run_dir, quadrupoles
from .optics import (
    absorption_label,
    direct_dielectric,
    emissivity_from_absorption,
    load_absorption,
    phonon_assisted_absorption,
    scissor_settings,
    slab_emissivity,
    write_absorption,
)
from .phonons import phonon_dispersion
from .superconductivity import (
    eliashberg,
    migdal_eliashberg,
    migdal_eliashberg_aniso,
    smearing_label,
    write_eliashberg,
)
from .system import Electrons, ElphonSystem, find_system_file, is_root, load_system
from .transport import (
    carrier_mobility,
    metal_resistivity,
    mobility_label,
    statistics_scissor,
    write_mobility,
)

__all__ = [
    'Electrons',
    'ElphonSystem',
    'absorption_label',
    'carrier_mobility',
    'direct_dielectric',
    'eliashberg',
    'emissivity_from_absorption',
    'find_system_file',
    'is_root',
    'load_absorption',
    'load_system',
    'long_range_path',
    'metal_resistivity',
    'migdal_eliashberg',
    'migdal_eliashberg_aniso',
    'mobility_label',
    'phonon_assisted_absorption',
    'phonon_dispersion',
    'quadrupole_inputs',
    'quadrupole_run_dir',
    'quadrupoles',
    'scissor_settings',
    'slab_emissivity',
    'smearing_label',
    'statistics_scissor',
    'write_absorption',
    'write_eliashberg',
    'write_mobility',
]
