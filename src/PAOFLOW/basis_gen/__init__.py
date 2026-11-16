"""Pseudo-atom basis generator.

Solves the radial (generalized, for ultrasoft / PAW) Schroedinger equation in a confining box
using the operator stored in a pseudopotential (local potential, projectors, strengths and
augmentation charges of a UPF file or of a VASP POTCAR dataset), to produce smooth radial
functions for arbitrary (n, l[, j]) channels: in memory (:func:`generate_basis_shells`, the
generator route of ``projections()``) or as ``BASIS_PS/<elem>/<shell>.dat`` tables
(:func:`generate_basis_for_pseudo`, ``paoflow-genbasis-ps``).
"""

from .driver import (
    GeneratedShell,
    generate_basis_for_directory,
    generate_basis_for_pseudo,
    generate_basis_shells,
)
from .radial import pseudize_shell, solve_radial_channel

__all__ = [
    'GeneratedShell',
    'generate_basis_for_directory',
    'generate_basis_for_pseudo',
    'generate_basis_shells',
    'pseudize_shell',
    'solve_radial_channel',
]
